#!/usr/bin/env python3

import os
import sys
import base64
import time
import json
import argparse
import threading
import tempfile
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from PIL import Image

# --- Anti-provenance final pass (independent variant of this project) -------
# The `antiprovenance` package makes it hard to prove an image was AI-generated:
# it purges provenance containers (C2PA/caBX, XMP, EXIF, text chunks), perturbs
# pixel-level watermarks (e.g. SynthID), optionally regenerates through a
# different encoder/decoder, and can verify the result against the OpenAI
# provenance API. Imported defensively so the project still runs if absent.
try:
    from antiprovenance import finalize_image as _ap_finalize_image
except ImportError:  # pragma: no cover - package not installed
    _ap_finalize_image = None

# Load .env from the project root (parent of src/)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

def c(color: str, text: str) -> str:
    return f"{color}{text}{C.RESET}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_MOCKUPS = [
    "basic.png",
    "double.png",
    "open.png",
    "clear.png",
]

BASE_DIR             = Path(__file__).resolve().parent.parent
ASSETS_DIR           = BASE_DIR / "assets"
MOCKUPS_DIR          = ASSETS_DIR / "mockups"
PROMPTS_DIR          = ASSETS_DIR / "prompts"
REFS_DIR             = ASSETS_DIR / "refs"
OUTPUTS_DIR          = BASE_DIR / "outputs"
OUTPUT_ARCHIVE       = OUTPUTS_DIR / "mockify_outputs.zip"
PROMPT_FILE          = PROMPTS_DIR / "texture_applyer.txt"
CHECKER_PROMPT_FILE  = PROMPTS_DIR / "ImageChecker.txt"

# --- Anti-provenance settings (override via .env or the environment) -------
# MOCKIFY_ANTIPROVENANCE: off | light | balanced | aggressive | max
ANTIPROVENANCE_STRENGTH = os.getenv("MOCKIFY_ANTIPROVENANCE", "balanced").strip().lower()
# MOCKIFY_VERIFY_PROVENANCE: 1/true/yes -> query the OpenAI provenance API
ANTIPROVENANCE_VERIFY = os.getenv("MOCKIFY_VERIFY_PROVENANCE", "0").strip().lower() in ("1", "true", "yes")
# MOCKIFY_REGENERATE: off | local | http (the only real lever against SynthID;
# "local" needs the optional torch/diffusers extras, "http" an img2img endpoint)
ANTIPROVENANCE_REGENERATE = os.getenv("MOCKIFY_REGENERATE", "off").strip().lower()
ANTIPROVENANCE_REGENERATE_STRENGTH = float(os.getenv("MOCKIFY_REGENERATE_STRENGTH", "0.75"))


def _fallback_strip_metadata(image_path: Path) -> dict:
    """Safety net when the antiprovenance package is unavailable.

    Re-encodes the PNG through Pillow, which drops ancillary chunks
    (C2PA/caBX, tEXt/iTXt, eXIf, tIME, ...). It cannot touch pixel-level
    watermarks such as SynthID, but it still removes the signed manifest.
    """
    try:
        with Image.open(image_path) as img:
            img.load()
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))
            clean.save(image_path, format="PNG")
        return {"fallback": True, "container_scrubbed": True}
    except Exception as exc:  # pragma: no cover
        return {"fallback": True, "error": str(exc)}


def finalize_deliverable(image_path: Path) -> dict:
    """Run the anti-provenance final pass on a freshly generated image.

    Uses the `antiprovenance` package when available (container scrub + pixel
    sanitize + optional regeneration/verification); otherwise falls back to a
    plain metadata strip so generated files never carry stale provenance.
    """
    image_path = Path(image_path)
    if _ap_finalize_image is None:
        return _fallback_strip_metadata(image_path)
    try:
        return _ap_finalize_image(
            image_path,
            strength=ANTIPROVENANCE_STRENGTH,
            regenerate=ANTIPROVENANCE_REGENERATE,
            regenerate_strength=ANTIPROVENANCE_REGENERATE_STRENGTH,
            verify=ANTIPROVENANCE_VERIFY,
        )
    except Exception as exc:
        return {"error": str(exc)}

# --- Multi mode specific paths ---
MULTI_MOCKUP_FILE    = MOCKUPS_DIR / "multi.png"
MULTI_MASK_FILE      = MOCKUPS_DIR / "multi_mask.png"
MULTI_APPLY_PROMPT_FILE = PROMPTS_DIR / "multi.txt"
BOARD_GENERATOR_PROMPT_FILE = PROMPTS_DIR / "board_generator.txt"

# --- Choice mode specific paths (selection graphic: "choose your case style") ---
CHOICE_MOCKUP_FILE       = MOCKUPS_DIR / "choice.png"
CHOICE_APPLY_PROMPT_FILE = PROMPTS_DIR / "choice.txt"

IMAGE_SIZE     = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024").strip()
MODEL          = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst").strip()
IMAGE_QUALITY  = os.getenv("OPENAI_IMAGE_QUALITY", "medium").strip()
CHECKER_MODEL  = os.getenv("OPENAI_CHECKER_MODEL", "gpt-5.6-luna").strip()
CHECKER_DETAIL = os.getenv("OPENAI_CHECKER_DETAIL", "original")
CHECKER_REASONING = os.getenv("OPENAI_CHECKER_REASONING", "low")

_PRICE_INPUT_PER_M  = 8.00
_PRICE_OUTPUT_PER_M = 30.00
_EST_INPUT_TOKENS   = 4160 * 2 + 500
_EST_OUTPUT_TOKENS  = 5056

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def get_file_size(path: Path) -> str:
    size_bytes = path.stat().st_size
    size_kb = size_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.2f} MB"
    return f"{size_kb:.2f} KB"


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def load_text_file(path: Path, label: str, required: bool = True) -> str:
    if required:
        validate_file(path, label)
    elif not path.exists():
        return ""
    content = path.read_text(encoding="utf-8").strip()
    if required and not content:
        raise ValueError(f"{label} is empty — please add content.")
    return content


def load_prompt() -> str:
    return load_text_file(PROMPT_FILE, "Prompt file")


def load_checker_prompt() -> str:
    return load_text_file(CHECKER_PROMPT_FILE, "ImageChecker prompt file")


def load_multi_apply_prompt() -> str:
    return load_text_file(MULTI_APPLY_PROMPT_FILE, "Multi apply prompt file (multi.txt)")


def load_choice_apply_prompt() -> str:
    return load_text_file(CHOICE_APPLY_PROMPT_FILE, "Choice apply prompt file (choice.txt)")


def load_board_generator_prompt() -> str:
    return load_text_file(BOARD_GENERATOR_PROMPT_FILE, "Board generator prompt file (board_generator.txt)")


def create_output_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def safe_stem(value: str) -> str:
    """Return a filesystem-safe stem while keeping Unicode letters and digits."""
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned.strip("_") or "image"


def create_run_dir(name: str, timestamp: str | None = None) -> Path:
    """Create one collision-safe, timestamped output directory for a run."""
    create_output_dir()
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUTS_DIR / f"{safe_stem(name)}_{timestamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = OUTPUTS_DIR / f"{base.name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def resolve_worker_count(total_jobs: int, requested: int | None = None) -> int:
    """Use one worker per job by default, capped at eight concurrent jobs."""
    if total_jobs < 1:
        return 0
    limit = requested if requested is not None else 8
    return min(total_jobs, limit, 8)


def archive_output_folders() -> tuple[Path, int, int]:
    """Append output folders to a verified ZIP, then remove only archived folders."""
    create_output_dir()
    run_folders = sorted(path for path in OUTPUTS_DIR.iterdir() if path.is_dir())
    if not run_folders:
        return OUTPUT_ARCHIVE, 0, 0

    outputs_root = OUTPUTS_DIR.resolve()
    for folder in run_folders:
        if folder.resolve().parent != outputs_root:
            raise RuntimeError(f"Unsafe output folder, archive cancelled: {folder}")

    temporary_archive = OUTPUTS_DIR / ".mockify_outputs.tmp.zip"
    temporary_archive.unlink(missing_ok=True)
    if OUTPUT_ARCHIVE.is_file():
        shutil.copy2(OUTPUT_ARCHIVE, temporary_archive)

    archived_files = 0
    try:
        mode = "a" if temporary_archive.exists() else "w"
        with zipfile.ZipFile(
            temporary_archive,
            mode=mode,
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            existing_names = set(archive.namelist())
            for folder in run_folders:
                for source in sorted(path for path in folder.rglob("*") if path.is_file()):
                    archive_name = source.relative_to(OUTPUTS_DIR).as_posix()
                    if archive_name in existing_names:
                        continue
                    archive.write(source, archive_name)
                    existing_names.add(archive_name)
                    archived_files += 1

        with zipfile.ZipFile(temporary_archive, "r") as archive:
            broken_entry = archive.testzip()
            if broken_entry is not None:
                raise RuntimeError(f"Invalid archive entry: {broken_entry}")

        temporary_archive.replace(OUTPUT_ARCHIVE)
        for folder in run_folders:
            shutil.rmtree(folder)
        return OUTPUT_ARCHIVE, len(run_folders), archived_files
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise

# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _PRICE_INPUT_PER_M + output_tokens * _PRICE_OUTPUT_PER_M) / 1_000_000

def estimated_cost() -> float:
    return compute_cost(_EST_INPUT_TOKENS, _EST_OUTPUT_TOKENS)

def format_cost(usd: float) -> str:
    return f"${usd:.4f}"

# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str):
        self.label   = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            print(f"\r  {c(C.CYAN, frame)}  {self.label} ", end="", flush=True)
            time.sleep(0.1)
            i += 1

    def start(self):
        self._thread.start()

    def stop(self, success: bool = True):
        self._stop.set()
        self._thread.join()
        icon = c(C.GREEN, "✓") if success else c(C.RED, "✗")
        print(f"\r  {icon}  {self.label}   ")

# ---------------------------------------------------------------------------
# Prompt construction (Single mode)
# ---------------------------------------------------------------------------

_MOCKUP_EXTRA: dict = {
    "basic.png": (
        "\n\nIMPORTANT — SINGLE MOCKUP:\n"
        "There is one Kindle case visible in this mockup.\n"
        "Apply the design so it fits perfectly on the printable surface.\n"
        "Respect the perspective and lighting of the case."
    ),
    "double.png": (
        "\n\nIMPORTANT — DOUBLE MOCKUP:\n"
        "There are TWO Kindle cases visible in this mockup.\n"
        "Apply the exact same artwork on BOTH cases.\n"
        "Respect the perspective and lighting of each case independently."
    ),
    "open.png": (
        "\n\nIMPORTANT — OPEN CASE MOCKUP:\n"
        "The case is shown open, with both the front and back covers visible.\n"
        "Apply the texture on BOTH sides equally — same design, same scale.\n"
        "Respect the perspective and lighting of each panel independently."
    ),
    "clear.png": (
        "\n\nIMPORTANT — CLEAR CASE MOCKUP:\n"
        "PRESERVE THE EXACT ORIGINAL LAYOUT OF THE MOCKUP. Do NOT resize, recompose, reframe, crop, or reposition any element.\n"
        "The transparent panel on the left and the white panel on the right must keep their EXACT original size, shape, and position.\n"
        "Apply the texture ONLY on the white/printable panel on the right.\n"
        "Do NOT paint, alter, or overlay the transparent panel in any way — it must remain fully transparent.\n"
        "The white panel must keep pixel-identical proportions to the original mockup.\n"
        "This is a texture fill task only. Zero layout changes allowed."
    ),
}

def build_prompt(base_prompt: str, mockup_name: str) -> str:
    role_assignment = (
        "IMAGE ROLE ASSIGNMENT (STRICT):\n"
        "  Image 1 -> MOCKUP BASE  (the Kindle case 3D template)\n"
        "  Image 2 -> TEXTURE DESIGN  (the artwork to transfer onto Image 1)\n\n"
        "Do not swap these roles. Do not merge them. Apply Image 2 onto Image 1 only.\n\n"
    )
    anti_creativity = (
        "REDUCE CREATIVE FREEDOM TO ZERO:\n"
        "  - You have no artistic license.\n"
        "  - You are a texture compositor, not an artist.\n"
        "  - Reproduce the texture mechanically.\n\n"
    )
    prompt = role_assignment + anti_creativity + base_prompt
    extra = _MOCKUP_EXTRA.get(mockup_name, "")
    if extra:
        prompt += extra
    return prompt


def build_multi_prompt(base_prompt: str) -> str:
    """
    Build the prompt used to apply the 9-design board onto multi.png.
    The multi mockup may show 9 cases in a scattered/overlapping arrangement
    (not necessarily a clean aligned grid) — the composition must be preserved
    exactly as-is, only the printable surfaces are touched.
    """
    role_assignment = (
        "IMAGE ROLE ASSIGNMENT (STRICT):\n"
        "  Image 1 -> MOCKUP BASE  (9 Kindle cases, in their own fixed positions/angles — "
        "this arrangement may be a clean grid OR a scattered/overlapping layout; whatever it is, "
        "it is FINAL and must not be changed)\n"
        "  Image 2 -> DESIGN BOARD  (a board containing exactly 9 distinct designs)\n\n"
        "Do not swap these roles. Do not merge them. Apply each of the 9 designs from Image 2 onto "
        "the 9 cases in Image 1, one unique design per case, with no duplicates and no blending. "
        "Match designs to cases in reading order (left-to-right, top-to-bottom in Image 2, mapped to "
        "left-to-right, top-to-bottom visual position in Image 1) unless the cases are clearly "
        "pre-arranged in a specific order.\n\n"
    )
    anti_creativity = (
        "REDUCE CREATIVE FREEDOM TO ZERO:\n"
        "  - You have no artistic license.\n"
        "  - You are a texture compositor, not an artist, not a photo retoucher, not a layout designer.\n"
        "  - Reproduce each of the 9 designs mechanically, exactly as provided, one per case.\n"
        "  - Do not mix designs between cases. Do not skip any of the 9 cases. Do not reuse a design.\n"
        "  - Do not rearrange, align, straighten, de-overlap, resize, rotate, or reposition any case. "
        "If the cases overlap or sit at irregular angles in Image 1, keep that exact arrangement.\n\n"
    )
    return role_assignment + anti_creativity + base_prompt


def build_choice_prompt(base_prompt: str) -> str:
    """
    Build the prompt used to apply the 9-design board onto choice.png — a clean,
    numbered 3x3 selection graphic ("Choose your Case style") used as a customer-
    facing thumbnail/carousel image. Unlike multi.png, this mockup has a clean
    aligned grid (no scattered overlap), numbered badges 1-9, and a title.
    """
    role_assignment = (
        "IMAGE ROLE ASSIGNMENT (STRICT):\n"
        "  Image 1 -> MOCKUP BASE  (a clean 3x3 grid of 9 numbered Kindle cases, with a title "
        "and decorative elements — this is a customer-facing 'choose your design' selection graphic)\n"
        "  Image 2 -> DESIGN BOARD  (a board containing exactly 9 distinct designs)\n\n"
        "Do not swap these roles. Do not merge them. Apply each of the 9 designs from Image 2 onto "
        "the matching numbered case in Image 1, one unique design per case, with no duplicates and "
        "no blending. Match designs to case numbers in reading order: design 1 (top-left of Image 2) "
        "-> case 1, design 2 -> case 2, and so on through design 9 -> case 9.\n\n"
    )
    anti_creativity = (
        "REDUCE CREATIVE FREEDOM TO ZERO:\n"
        "  - You have no artistic license.\n"
        "  - You are a texture compositor, not an artist, not a layout designer.\n"
        "  - Reproduce each of the 9 designs mechanically, exactly as provided, one per numbered case.\n"
        "  - Do not mix designs between cases. Do not skip any of the 9 cases. Do not reuse a design.\n"
        "  - Do not move, resize, rotate, or restyle the numbered badges, title text, or background.\n\n"
    )
    return role_assignment + anti_creativity + base_prompt


def build_board_generation_prompt(base_prompt: str, art_direction: str, has_refs: bool) -> str:
    """
    Build the prompt used to generate the 9-design board itself.
    """
    structure_lock = (
        "STRICT OUTPUT FORMAT:\n"
        "  - Output a single square canvas image on a pure white background.\n"
        "  - Arrange exactly 9 distinct designs in a 3x3 grid (3 rows, 3 columns) inside that square canvas.\n"
        "  - Each of the 9 cells MUST be a TALL RECTANGLE (portrait orientation), matching the "
        "proportions of a Kindle e-reader case — approximately 3:4 width-to-height ratio "
        "(clearly taller than wide). Do NOT use square cells. Do NOT use landscape cells. "
        "All 9 cells must share the exact same tall rectangular shape and size.\n"
        "  - Since the cells are taller than wide, leave even white margins above and below the "
        "3x3 grid so the whole grid fits inside the square canvas without cropping or distorting "
        "the rectangles.\n"
        "  - Each of the 9 designs must be a separate, self-contained pattern/illustration, "
        "fully visible within its own rectangle, with even spacing/margins between cells, "
        "matching the reference layout (white background, 9 tall rectangles, 3x3 grid).\n"
        "  - Do not add text, labels, numbers, borders, or watermarks.\n\n"
    )
    prompt = structure_lock + base_prompt
    if art_direction:
        prompt += f"\n\nART DIRECTION FOR THE 9 DESIGNS:\n{art_direction.strip()}\n"
    if has_refs:
        prompt += (
            "\n\nREFERENCE IMAGES PROVIDED:\n"
            "Use the additional reference image(s) as style/inspiration for the 9 designs "
            "(palette, motifs, mood). Do not simply copy a single reference into all 9 cells — "
            "create 9 distinct but coherent designs inspired by the references and the art direction above."
        )
    return prompt

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(image_base64: str, output_path: Path, finalize: bool = True):
    output_path.write_bytes(base64.b64decode(image_base64))
    if finalize:
        return finalize_deliverable(output_path)
    return None

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

# ---------------------------------------------------------------------------
# Image quality checker (Single mode only)
# ---------------------------------------------------------------------------

def check_image_quality(
    client: OpenAI,
    image_path: Path,
    texture_path: Path,
    show_progress: bool = True,
) -> dict:
    """
    Send BOTH the original texture AND the generated mockup to the vision checker.
    Image 1 = original texture (reference), Image 2 = generated mockup result.
    Returns parsed JSON dict with keys: has_problem, problems, severity, details.
    """
    checker_prompt = load_checker_prompt()
    texture_b64 = image_to_base64(texture_path)
    result_b64  = image_to_base64(image_path)

    spinner = Spinner("Checking image quality ...") if show_progress else None
    if spinner:
        spinner.start()
    try:
        response = client.chat.completions.create(
            model=CHECKER_MODEL,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGE 1 — ORIGINAL TEXTURE (reference design that must be applied):",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{texture_b64}",
                                "detail": CHECKER_DETAIL,
                            },
                        },
                        {
                            "type": "text",
                            "text": "IMAGE 2 — GENERATED MOCKUP RESULT (what was actually produced):",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{result_b64}",
                                "detail": CHECKER_DETAIL,
                            },
                        },
                        {
                            "type": "text",
                            "text": checker_prompt,
                        },
                    ],
                }
            ],
        )
    except Exception:
        if spinner:
            spinner.stop(success=False)
        raise

    if spinner:
        spinner.stop(success=True)

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ImageChecker returned invalid JSON: {e}\nRaw: {raw_text!r}")

    return result


def print_check_result(check: dict) -> None:
    severity = check.get("severity", "none")
    details  = check.get("details", "")
    problems = check.get("problems", [])

    if not check.get("has_problem", False):
        print(f"  {c(C.GREEN, '  ✓ QC PASS')}  {c(C.GREY, details)}")
    else:
        sev_color = C.YELLOW if severity == "minor" else C.RED
        sev_label = c(sev_color, f"[{severity.upper()}]")
        print(f"  {c(C.RED, '  ✗ QC FAIL')}  {sev_label}  {c(C.GREY, details)}")
        if problems:
            issues = "  ".join(c(C.YELLOW, p) for p in problems)
            print(f"  {c(C.GREY, '     Issues:')}  {issues}")


def ask_skip_or_retry(mockup_name: str) -> bool:
    """
    Show a 5-second countdown. If the user types S or R before it expires,
    honour their choice. Otherwise auto-select Retry.
    Returns True = skip (keep bad image), False = retry.
    """
    TIMEOUT = 5

    print()
    print(f"  {c(C.YELLOW, '⚠')}  Quality issue detected on {c(C.BOLD, mockup_name)}")
    print(f"  {c(C.GREY, 'S = skip  |  R = retry  |  auto-retry in')} ", end="", flush=True)

    user_input: list = []   # filled by reader thread
    stop_event = threading.Event()

    def _reader():
        try:
            val = input()  # blocks until Enter
            user_input.append(val.strip().upper())
        except (EOFError, OSError):
            pass
        finally:
            stop_event.set()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    for remaining in range(TIMEOUT, 0, -1):
        if stop_event.is_set():
            break
        print(f"\r  {c(C.GREY, 'S = skip  |  R = retry  |  auto-retry in')} {c(C.CYAN, str(remaining) + 's')} ", end="", flush=True)
        stop_event.wait(timeout=1.0)

    print()  # newline after countdown

    if not user_input:
        # Timeout expired — auto retry
        print(f"  {c(C.CYAN, '→')} No input — auto-retrying.\n")
        return False  # retry

    choice = user_input[0]
    if choice == "S":
        print(f"  {c(C.YELLOW, '→')} Keeping image as-is.\n")
        return True   # skip
    else:
        # R or anything else → retry
        print(f"  {c(C.CYAN, '→')} Retrying.\n")
        return False  # retry

# ---------------------------------------------------------------------------
# Core generation (image edit — used by Single, Multi-apply, Multi-board-with-refs)
# ---------------------------------------------------------------------------

def call_image_edit(client: OpenAI, image_paths: list, prompt: str) -> tuple:
    """
    Calls images.edit with one or more input images and returns (b64_json, usage).
    image_paths: list of Path objects, opened in order.
    """
    opened = [open(p, "rb") for p in image_paths]
    try:
        response = client.images.edit(
            model=MODEL,
            image=opened,
            prompt=prompt,
            size=IMAGE_SIZE,
            quality=IMAGE_QUALITY,
        )
    finally:
        for f in opened:
            f.close()

    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise RuntimeError("API returned no image data.")
    return image_base64, getattr(response, "usage", None)


def _parse_image_size(size_str: str) -> tuple:
    """Parses 'WxH' strings like '1024x1024' into (width, height) ints."""
    w_str, h_str = size_str.lower().split("x")
    return int(w_str), int(h_str)


def normalize_mockup_and_mask(mockup_path: Path, mask_path: Path, target_size: str) -> tuple:
    """
    Resizes the mockup and its mask to the EXACT same target dimensions, in a
    synchronized way, so the API never has to resize them independently
    (which would desync the mask from the mockup and blur/soften mask edges).

    - Mockup: resized with LANCZOS (quality-preserving for photos).
    - Mask: resized with NEAREST (keeps the alpha channel perfectly binary —
      no blurred/soft edges that would let the model bleed outside the
      intended editable zones).

    Returns (resized_mockup_path, resized_mask_path) — temp files that the
    caller is responsible for cleaning up.
    """
    target_w, target_h = _parse_image_size(target_size)

    mockup = Image.open(mockup_path).convert("RGB")
    mask   = Image.open(mask_path).convert("RGBA")

    if mockup.size != mask.size:
        raise ValueError(
            f"Mockup size {mockup.size} and mask size {mask.size} differ — "
            "they must be generated from the same base image."
        )

    mockup_resized = mockup.resize((target_w, target_h), Image.LANCZOS)
    mask_resized   = mask.resize((target_w, target_h), Image.NEAREST)

    # Force the alpha channel back to strictly binary (0/255) in case the
    # source mask had any soft edges to begin with.
    import numpy as np
    mask_arr = np.array(mask_resized)
    mask_arr[..., 3] = np.where(mask_arr[..., 3] > 127, 255, 0).astype("uint8")
    mask_resized = Image.fromarray(mask_arr, mode="RGBA")

    tmp_dir = Path(tempfile.mkdtemp(prefix="mockify_"))
    resized_mockup_path = tmp_dir / "mockup_resized.png"
    resized_mask_path   = tmp_dir / "mask_resized.png"
    mockup_resized.save(resized_mockup_path)
    mask_resized.save(resized_mask_path)

    return resized_mockup_path, resized_mask_path


def call_image_edit_with_mask(client: OpenAI, image_paths: list, mask_path: Path, prompt: str) -> tuple:
    """
    Calls images.edit with one or more input images PLUS an inpainting mask.
    The mask applies to the FIRST image in image_paths (per OpenAI API behavior):
    transparent (alpha=0) areas are editable, opaque (alpha=255) areas are protected.
    image_paths[0] must be the base image the mask was made for (e.g. the multi mockup),
    and MUST already be the same pixel dimensions as mask_path (use
    normalize_mockup_and_mask beforehand to guarantee this).
    Returns (b64_json, usage).
    """
    opened = [open(p, "rb") for p in image_paths]
    mask_file = open(mask_path, "rb")
    try:
        response = client.images.edit(
            model=MODEL,
            image=opened,
            mask=mask_file,
            prompt=prompt,
            size=IMAGE_SIZE,
            quality=IMAGE_QUALITY,
        )
    finally:
        for f in opened:
            f.close()
        mask_file.close()

    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise RuntimeError("API returned no image data.")
    return image_base64, getattr(response, "usage", None)


def call_image_generate(client: OpenAI, prompt: str) -> tuple:
    """
    Calls images.generate (text-to-image, no input images) and returns (b64_json, usage).
    """
    response = client.images.generate(
        model=MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        quality=IMAGE_QUALITY,
    )
    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise RuntimeError("API returned no image data.")
    return image_base64, getattr(response, "usage", None)


def usage_to_result(usage, elapsed: float, output_path: Path) -> dict:
    if usage and hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        input_tok  = usage.input_tokens
        output_tok = usage.output_tokens
        cost       = compute_cost(input_tok, output_tok)
        source     = "real"
    else:
        input_tok  = _EST_INPUT_TOKENS
        output_tok = _EST_OUTPUT_TOKENS
        cost       = estimated_cost()
        source     = "estimated"

    return {
        "cost_usd":      cost,
        "cost_source":   source,
        "input_tokens":  input_tok,
        "output_tokens": output_tok,
        "elapsed":       elapsed,
        "output_path":   output_path,
    }


def generate_mockup(
    client: OpenAI,
    mockup_path: Path,
    texture_path: Path,
    mockup_name: str,
    output_path: Path,
    extra_instructions: str = "",
    show_progress: bool = True,
) -> dict:
    base_prompt = load_prompt()
    prompt = build_prompt(base_prompt, mockup_name)
    if extra_instructions:
        prompt += extra_instructions

    spinner = Spinner(f"Calling API for {c(C.BOLD, mockup_name)} ...") if show_progress else None
    if spinner:
        spinner.start()

    t_start = time.monotonic()
    try:
        image_base64, usage = call_image_edit(client, [mockup_path, texture_path], prompt)
    except Exception:
        if spinner:
            spinner.stop(success=False)
        raise

    elapsed = time.monotonic() - t_start
    if spinner:
        spinner.stop(success=True)

    ap_report = save_image(image_base64, output_path)
    result = usage_to_result(usage, elapsed, output_path)
    result["antiprovenance"] = ap_report
    return result

# ---------------------------------------------------------------------------
# Generation + QC loop (Single mode — first pass + one retry)
# ---------------------------------------------------------------------------

def generate_with_qc(
    client: OpenAI,
    mockup_path: Path,
    texture_path: Path,
    mockup_name: str,
    output_path: Path,
    interactive_qc: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Generate a mockup, run QC, handle retry logic.
    Returns the final result dict (may include qc_status key).
    Raises on hard failure.
    """

    # --- First generation ---
    result = generate_mockup(
        client, mockup_path, texture_path, mockup_name, output_path,
        show_progress=show_progress,
    )

    # --- QC check ---
    check = check_image_quality(client, output_path, texture_path, show_progress=show_progress)
    if show_progress:
        print_check_result(check)
        print()

    if not check.get("has_problem", False):
        result["qc_status"] = "pass"
        return result

    # --- Problem detected: ask user ---
    # Concurrent workers must not compete for stdin. They auto-retry QC failures.
    skip = ask_skip_or_retry(mockup_name) if interactive_qc else False

    if skip:
        result["qc_status"] = "skipped"
        return result

    # --- Build correction instructions from QC errors ---
    problems = check.get("problems", [])
    details  = check.get("details", "")

    _PROBLEM_FIXES = {
        "texture_recreated":      "Do NOT redraw or reimagine the texture — copy it EXACTLY as provided. Every color, shape, character, and element must match the original texture pixel-for-pixel in style and composition.",
        "texture_not_applied":    "The texture MUST be fully applied on the case surface — do not leave any bare or original surface visible.",
        "wrong_proportions":      "Do NOT stretch or squish the texture — preserve its original aspect ratio exactly.",
        "texture_bleeding":       "Keep the texture strictly within the printable area — do not let it bleed onto the background, table, or shadow zones.",
        "low_coverage":           "Cover the ENTIRE printable surface with the texture — no large empty or untextured areas.",
        "misalignment":           "Align the texture precisely with the case edges — no offset, no rotation, no gap.",
        "color_corruption":       "Preserve the original colors of the texture faithfully — no black patches, no color shifts, no artifacts.",
        "layout_changed":         "Do NOT change the composition, framing, scale, or crop of the mockup — reproduce the original layout exactly.",
        "transparent_panel_painted": "Do NOT paint or fill the transparent panel — it must remain fully transparent and unmodified.",
    }

    fix_lines = []
    for p in problems:
        fix = _PROBLEM_FIXES.get(p)
        if fix:
            fix_lines.append(f"  - {fix}")
    if details and details.lower() != "ok":
        fix_lines.append(f"  - Previous attempt failed with: {details}")

    avoid_block = ""
    if fix_lines:
        avoid_block = (
            "\n\nCRITICAL — CORRECTIONS FOR THIS RETRY (previous attempt had quality issues):\n"
            + "\n".join(fix_lines)
            + "\nFix ALL of the above. This is mandatory."
        )
        if show_progress:
            print(f"  {c(C.CYAN, '  ℹ')} Injecting {len(fix_lines)} correction(s) into retry prompt.")

    # --- Delete the bad image before retry ---
    if output_path.exists():
        output_path.unlink()
        if show_progress:
            print(f"  {c(C.GREY, '  🗑')} Deleted failed image: {c(C.DIM, output_path.name)}\n")

    # --- Retry once (same output path, corrected prompt) ---
    retry_result = generate_mockup(
        client, mockup_path, texture_path, mockup_name, output_path,
        extra_instructions=avoid_block,
        show_progress=show_progress,
    )

    # QC on retry
    check2 = check_image_quality(client, output_path, texture_path, show_progress=show_progress)
    if show_progress:
        print_check_result(check2)
        print()

    if not check2.get("has_problem", False):
        retry_result["output_path"] = output_path
        retry_result["qc_status"]   = "pass_on_retry"
        retry_result["cost_usd"]   += result["cost_usd"]
        retry_result["elapsed"]    += result["elapsed"]
        return retry_result
    else:
        # Retry also failed — delete the retry image too, then stop
        if output_path.exists():
            output_path.unlink()
            if show_progress:
                print(f"  {c(C.GREY, '  🗑')} Deleted retry image: {c(C.DIM, output_path.name)}")
        if show_progress:
            print(f"  {c(C.RED, '  ✗ Retry also failed QC — stopping this mockup.')}\n")
        raise RuntimeError(
            f"QC failed twice for {mockup_name}: {check2.get('details', 'unknown error')}"
        )

# ---------------------------------------------------------------------------
# Interactive mode selector (Single / Multi)
# ---------------------------------------------------------------------------

def ask_run_mode() -> str:
    width = 56
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Choose a mode'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, '1')}  {c(C.CYAN + C.BOLD, 'Single')}  — one texture, applied to chosen mockups")
    print(f"  {c(C.GREY, '2')}  {c(C.CYAN + C.BOLD, 'Multi')}   — a 3x3 board of 9 designs, applied to the multi mockup")
    print(f"  {c(C.GREY, '3')}  {c(C.CYAN + C.BOLD, 'Compress')} — archive completed output folders into one ZIP")
    print()

    while True:
        try:
            raw = input(f"  {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw in ("1", "single", "Single", "SINGLE"):
            return "single"
        if raw in ("2", "multi", "Multi", "MULTI"):
            return "multi"
        if raw in ("3", "compress", "Compress", "archive", "Archive"):
            return "archive"
        print(f"  {c(C.RED, 'Invalid input:')} type 1 (Single), 2 (Multi) or 3 (Compress).")


# ---------------------------------------------------------------------------
# Interactive mockup selector (Single mode)
# ---------------------------------------------------------------------------

MOCKUP_LABELS: dict = {
    "basic.png":  "Basic     — single case, front view",
    "double.png": "Double    — two cases side by side",
    "open.png":   "Open      — case open, both sides",
    "clear.png":  "Clear     — transparent + white panel",
}

def ask_mockup_selection() -> list:
    width = 56
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Select mockups to generate'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, '0')}  {c(C.CYAN + C.BOLD, 'ALL')}  — generate all mockups")
    print()
    for i, name in enumerate(ALL_MOCKUPS, start=1):
        label  = MOCKUP_LABELS.get(name, name)
        exists = (MOCKUPS_DIR / name).exists()
        status = c(C.GREEN, "✓") if exists else c(C.RED, "✗ missing")
        print(f"  {c(C.GREY, str(i))}  {c(C.YELLOW, label)}  {status}")
    print()
    print(f"  {c(C.GREY, 'Type numbers separated by spaces, e.g.:  1 3')}")
    print(f"  {c(C.GREY, 'Or press Enter alone to generate ALL.')}")
    print()

    while True:
        try:
            raw = input(f"  {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw == "" or raw == "0":
            return list(ALL_MOCKUPS)

        parts    = raw.split()
        selected = []
        valid    = True
        for p in parts:
            if not p.isdigit():
                print(f"  {c(C.RED, 'Invalid input:')} {p!r} — use numbers only.")
                valid = False
                break
            idx = int(p)
            if idx < 1 or idx > len(ALL_MOCKUPS):
                print(f"  {c(C.RED, 'Out of range:')} {idx} — pick between 1 and {len(ALL_MOCKUPS)}.")
                valid = False
                break
            name = ALL_MOCKUPS[idx - 1]
            if name not in selected:
                selected.append(name)

        if valid and selected:
            return selected

# ---------------------------------------------------------------------------
# Interactive Multi-mode setup
# ---------------------------------------------------------------------------

def ask_multi_board_source(output_dir: Path) -> Path:
    """
    Ask the user whether they already have a ready-made 9-design board,
    or whether the board should be generated. Returns the Path to a
    9-design board image (existing or freshly generated).
    """
    width = 56
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Multi mode: 9-design board'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, 'Do you already have a ready-made 9-design board image?')}")
    print(f"  {c(C.GREY, '1')}  {c(C.CYAN + C.BOLD, 'Yes')}  — I will give you the file path")
    print(f"  {c(C.GREY, '2')}  {c(C.CYAN + C.BOLD, 'No')}   — generate it (art direction + optional reference images)")
    print()

    while True:
        try:
            raw = input(f"  {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw in ("1", "yes", "y", "Y"):
            return ask_existing_board_path()
        if raw in ("2", "no", "n", "N"):
            return run_board_generation(output_dir)
        print(f"  {c(C.RED, 'Invalid input:')} type 1 (Yes) or 2 (No).")


def ask_existing_board_path() -> Path:
    while True:
        try:
            raw = input(f"  {c(C.GREY, 'Path to the 9-design board image')} {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        path = Path(raw).expanduser().resolve()
        try:
            validate_file(path, "Board image")
            return path
        except Exception as exc:
            print(f"  {c(C.RED, 'Error:')} {exc}")


def ask_reference_images() -> list:
    """
    Optionally collect 0-9 reference image paths. Press Enter alone to skip/finish.
    """
    print()
    print(f"  {c(C.GREY, 'Optional: reference images for style/inspiration (0 to 9).')}")
    print(f"  {c(C.GREY, 'Enter one path at a time, press Enter alone when done.')}")
    refs = []
    while len(refs) < 9:
        try:
            raw = input(f"  {c(C.GREY, 'Ref image')} {len(refs) + 1} {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw == "":
            break

        path = Path(raw).expanduser().resolve()
        try:
            validate_file(path, "Reference image")
            refs.append(path)
        except Exception as exc:
            print(f"  {c(C.RED, 'Error:')} {exc} — skipped.")

    return refs


def ask_art_direction(required: bool) -> str:
    print()
    label = "Describe the art direction for the 9 designs" + ("" if required else " (optional)")
    print(f"  {c(C.GREY, label)}")
    print(f"  {c(C.GREY, 'You can paste multiple lines (blank lines allowed). Type END on its own line when done.')}")
    while True:
        lines: list = []
        try:
            while True:
                line = input(f"  {c(C.BOLD, '>')} " if not lines else "  ")
                if line.strip().upper() == "END":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        raw = "\n".join(lines).strip()

        if raw or not required:
            return raw
        print(f"  {c(C.RED, 'Art direction is required when no reference images are provided.')}")


def run_board_generation(output_dir: Path) -> Path:
    """
    Drives the interactive flow to generate a fresh 9-design board:
    optional reference images + art direction text -> API call -> saved board.
    """
    refs = ask_reference_images()
    art_direction = ask_art_direction(required=(len(refs) == 0))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = load_board_generator_prompt()
    prompt = build_board_generation_prompt(base_prompt, art_direction, has_refs=bool(refs))

    board_path = output_dir / "board.png"

    print()
    spinner = Spinner("Generating 9-design board ...")
    spinner.start()

    t_start = time.monotonic()
    try:
        if refs:
            image_base64, usage = call_image_edit(client, refs, prompt)
        else:
            image_base64, usage = call_image_generate(client, prompt)
    except Exception:
        spinner.stop(success=False)
        raise

    elapsed = time.monotonic() - t_start
    spinner.stop(success=True)

    save_image(image_base64, board_path, finalize=False)
    result = usage_to_result(usage, elapsed, board_path)

    cost_label = (
        c(C.GREEN, format_cost(result["cost_usd"]))
        if result["cost_source"] == "real"
        else c(C.YELLOW, "~" + format_cost(result["cost_usd"]) + " (estimated)")
    )
    print(f"  {c(C.GREEN, '  ✓')} Board saved  {c(C.CYAN, board_path.name)}  {c(C.GREY, get_file_size(board_path))}")
    print(f"  {c(C.GREY, '  ⏱')}  {elapsed:.1f}s   {c(C.GREY, '💰')} {cost_label}")
    print()

    return board_path


def ask_multi_apply_targets() -> set:
    """
    NEW: After obtaining the board (generated or provided), let the user
    choose what to apply it to:
      1 = multi.png only
      2 = choice.png only
      3 = both
      4 = nothing (just keep/generate the board, don't apply it anywhere)
    Returns a set subset of {"multi", "choice"} (empty set if "nothing" chosen).
    """
    width = 56
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  What do you want to generate?'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, '1')}  {c(C.CYAN + C.BOLD, 'Multi')}    — apply the board onto {MULTI_MOCKUP_FILE.name} only")
    print(f"  {c(C.GREY, '2')}  {c(C.CYAN + C.BOLD, 'Choice')}   — apply the board onto {CHOICE_MOCKUP_FILE.name} only")
    print(f"  {c(C.GREY, '3')}  {c(C.CYAN + C.BOLD, 'Both')}     — generate multi.png AND choice.png")
    print(f"  {c(C.GREY, '4')}  {c(C.CYAN + C.BOLD, 'Nothing')}  — just keep the board, don't apply it anywhere")
    print()

    while True:
        try:
            raw = input(f"  {c(C.BOLD, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw == "1":
            return {"multi"}
        if raw == "2":
            return {"choice"}
        if raw == "3":
            return {"multi", "choice"}
        if raw == "4":
            return set()
        print(f"  {c(C.RED, 'Invalid input:')} type 1, 2, 3 or 4.")

# ---------------------------------------------------------------------------
# Multi mode — apply board onto multi.png mockup (no QC)
# ---------------------------------------------------------------------------

def generate_multi_mockup(client: OpenAI, board_path: Path, output_path: Path) -> dict:
    validate_file(MULTI_MOCKUP_FILE, "Multi mockup (multi.png)")

    base_prompt = load_multi_apply_prompt()
    prompt = build_multi_prompt(base_prompt)

    use_mask = MULTI_MASK_FILE.exists() and MULTI_MASK_FILE.stat().st_size > 0
    label = "multi.png (masked)" if use_mask else "multi.png"
    spinner = Spinner(f"Calling API for {c(C.BOLD, label)} ...")
    spinner.start()

    tmp_mockup_path = None
    tmp_mask_path   = None

    t_start = time.monotonic()
    try:
        if use_mask:
            # Resize mockup + mask TOGETHER to IMAGE_SIZE before sending them,
            # so the API never resizes them independently (which would desync
            # the mask from the mockup and soften its edges).
            tmp_mockup_path, tmp_mask_path = normalize_mockup_and_mask(
                MULTI_MOCKUP_FILE, MULTI_MASK_FILE, IMAGE_SIZE
            )
            # Mask applies to the FIRST image -> resized mockup must be image_paths[0]
            image_base64, usage = call_image_edit_with_mask(
                client, [tmp_mockup_path, board_path], tmp_mask_path, prompt
            )
        else:
            image_base64, usage = call_image_edit(client, [MULTI_MOCKUP_FILE, board_path], prompt)
    except Exception:
        spinner.stop(success=False)
        raise
    finally:
        if tmp_mockup_path and tmp_mockup_path.exists():
            tmp_mockup_path.unlink()
        if tmp_mask_path and tmp_mask_path.exists():
            tmp_mask_path.unlink()
            try:
                tmp_mask_path.parent.rmdir()
            except OSError:
                pass

    elapsed = time.monotonic() - t_start
    spinner.stop(success=True)

    ap_report = save_image(image_base64, output_path)
    result = usage_to_result(usage, elapsed, output_path)
    result["antiprovenance"] = ap_report
    return result

# ---------------------------------------------------------------------------
# Choice mode — apply board onto choice.png mockup (clean numbered grid, no QC)
# ---------------------------------------------------------------------------

def generate_choice_mockup(client: OpenAI, board_path: Path, output_path: Path) -> dict:
    validate_file(CHOICE_MOCKUP_FILE, "Choice mockup (choice.png)")

    base_prompt = load_choice_apply_prompt()
    prompt = build_choice_prompt(base_prompt)

    spinner = Spinner(f"Calling API for {c(C.BOLD, 'choice.png')} ...")
    spinner.start()

    t_start = time.monotonic()
    try:
        image_base64, usage = call_image_edit(client, [CHOICE_MOCKUP_FILE, board_path], prompt)
    except Exception:
        spinner.stop(success=False)
        raise

    elapsed = time.monotonic() - t_start
    spinner.stop(success=True)

    ap_report = save_image(image_base64, output_path)
    result = usage_to_result(usage, elapsed, output_path)
    result["antiprovenance"] = ap_report
    return result

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 30

def progress_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total) if total > 0 else 0
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{c(C.CYAN, bar)}] {current}/{total}"

def print_run_header(texture_path: Path, mockups: list, threads: int) -> None:
    width = 56
    line  = "─" * width
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Product Mockup Generator'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, 'Texture')}   {c(C.CYAN, str(texture_path.name))}  {c(C.GREY, get_file_size(texture_path))}")
    print(f"  {c(C.GREY, 'Model')}     {c(C.CYAN, MODEL)}")
    print(f"  {c(C.GREY, 'Checker')}   {c(C.CYAN, CHECKER_MODEL)}")
    print(f"  {c(C.GREY, 'Size')}      {c(C.CYAN, IMAGE_SIZE)}")
    print(f"  {c(C.GREY, 'Threads')}   {c(C.CYAN, str(threads))}")
    selected_names = "  ".join(c(C.YELLOW, m.replace(".png", "")) for m in mockups)
    print(f"  {c(C.GREY, 'Mockups')}   {selected_names}  {c(C.GREY, '(' + str(len(mockups)) + ' total)')}")
    est_total = format_cost(estimated_cost() * len(mockups))
    print(f"  {c(C.GREY, 'Est. cost')} {c(C.YELLOW, '~' + est_total)}")
    print(f"  {c(C.GREY, line)}")
    print()

def print_mockup_header(i: int, total: int, mockup_name: str, mockup_path: Path) -> None:
    bar = progress_bar(i - 1, total)
    print(f"  {bar}  {c(C.BOLD, '[' + str(i) + '/' + str(total) + ']')} {c(C.YELLOW, mockup_name)}")
    print(f"  {c(C.GREY, '  Mockup')}  {c(C.DIM, str(mockup_path))}  {c(C.GREY, get_file_size(mockup_path))}")

def print_antiprovenance_result(report: dict | None) -> None:
    if report is None:
        return
    if report.get("fallback"):
        if report.get("error"):
            print(f"  {c(C.YELLOW, '  ⚑')}  {c(C.GREY, 'anti-provenance fallback failed: ' + report['error'])}")
        else:
            print(f"  {c(C.GREEN, '  ⚑')}  {c(C.GREY, 'anti-provenance: metadata stripped (package not installed)')}")
        return
    if report.get("error"):
        print(f"  {c(C.YELLOW, '  ⚑')}  {c(C.GREY, 'anti-provenance failed: ' + str(report['error']))}")
        return

    sanitize = report.get("sanitize") or {}
    parts = [f"strength={sanitize.get('strength', ANTIPROVENANCE_STRENGTH)}"]
    if sanitize.get("psnr_db") is not None:
        parts.append(f"PSNR={sanitize['psnr_db']} dB")
    if sanitize.get("pixels_modified"):
        parts.append("pixels perturbed")
    before = sanitize.get("provenance_chunks_before") or []
    if before:
        parts.append(f"purged {'/'.join(before)}")

    regenerate = report.get("regenerate") or {}
    if regenerate.get("applied"):
        parts.append(f"regenerated via {regenerate.get('backend', regenerate.get('mode'))}")

    provenance = report.get("provenance")
    if provenance is not None:
        results = provenance.get("results") or {}
        if results:
            parts.append("verify: " + ", ".join(f"{k}={v}" for k, v in results.items()))
        elif provenance.get("error"):
            parts.append("verify: " + str(provenance["error"]))

    print(f"  {c(C.GREEN, '  ⚑')}  {c(C.GREY, 'anti-provenance')}  {c(C.DIM, '  '.join(parts))}")


def print_mockup_result(result: dict) -> None:
    qc_status  = result.get("qc_status", "")
    qc_icons   = {
        "pass":          c(C.GREEN,  "✓ QC"),
        "pass_on_retry": c(C.CYAN,   "✓ QC (retry)"),
        "skipped":       c(C.YELLOW, "⚠ QC skipped"),
        "no_qc":         c(C.GREY,   "— no QC (multi mode)"),
        "":              c(C.GREY,   "— no QC"),
    }
    qc_display = qc_icons.get(qc_status, c(C.GREY, qc_status))

    cost_label = (
        c(C.GREEN, format_cost(result["cost_usd"]))
        if result["cost_source"] == "real"
        else c(C.YELLOW, "~" + format_cost(result["cost_usd"]) + " (estimated)")
    )
    elapsed  = result["elapsed"]
    size     = get_file_size(result["output_path"])
    name     = result["output_path"].name
    tok_info = "(in={}  out={})".format(result["input_tokens"], result["output_tokens"])
    print(f"  {c(C.GREEN, '  ✓')} Saved  {c(C.CYAN, name)}  {c(C.GREY, size)}  {qc_display}")
    print(f"  {c(C.GREY, '  ⏱')}  {elapsed:.1f}s   {c(C.GREY, '💰')} {cost_label}  {c(C.GREY, tok_info)}")
    print_antiprovenance_result(result.get("antiprovenance"))
    print()

def print_summary(
    results: list,
    errors: list,
    wall_time: float | None = None,
    output_dir: Path | None = None,
) -> None:
    total      = len(results) + len(errors)
    ok         = len(results)
    total_cost = sum(r["cost_usd"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    has_est    = any(r["cost_source"] == "estimated" for r in results)

    qc_pass    = sum(1 for r in results if r.get("qc_status") == "pass")
    qc_retry   = sum(1 for r in results if r.get("qc_status") == "pass_on_retry")
    qc_skipped = sum(1 for r in results if r.get("qc_status") == "skipped")
    no_qc      = sum(1 for r in results if r.get("qc_status") == "no_qc")

    line = "─" * 52
    print(f"  {c(C.GREY, line)}")
    print(f"  {c(C.BOLD, 'Summary')}")
    print()
    print(f"  {c(C.GREY, 'Generated')}   {c(C.GREEN, str(ok))}/{total}")
    if total == 0:
        print(f"  {c(C.GREY, 'QC results')}  {c(C.GREY, '— (nothing applied, board only)')}")
    elif no_qc:
        print(f"  {c(C.GREY, 'QC results')}  {c(C.GREY, str(no_qc) + ' (no QC — multi mode)')}")
    else:
        print(f"  {c(C.GREY, 'QC results')}  "
              f"{c(C.GREEN, str(qc_pass))} pass  "
              f"{c(C.CYAN, str(qc_retry))} retry-pass  "
              f"{c(C.YELLOW, str(qc_skipped))} skipped")
    print(f"  {c(C.GREY, 'Total time')}  {c(C.CYAN, str(round(total_time, 1)) + 's')}")
    if wall_time is not None:
        print(f"  {c(C.GREY, 'Wall time')}   {c(C.CYAN, str(round(wall_time, 1)) + 's')}")

    cost_str = format_cost(total_cost)
    if has_est:
        print(f"  {c(C.GREY, 'Total cost')}  {c(C.YELLOW, '~' + cost_str + ' (includes estimates)')}")
    else:
        print(f"  {c(C.GREY, 'Total cost')}  {c(C.GREEN, cost_str)}")

    if errors:
        print()
        print(f"  {c(C.RED, 'Errors:')}")
        for name, msg in errors:
            print(f"    {c(C.RED, '✗')} {name}: {c(C.DIM, msg)}")

    print()
    print(f"  {c(C.GREY, 'Output dir')}  {c(C.CYAN, str(output_dir or OUTPUTS_DIR))}")
    print()

# ---------------------------------------------------------------------------
# Main generation loop — Single mode
# ---------------------------------------------------------------------------

def _run_single_job(
    texture_path: Path,
    mockup_name: str,
    run_dir: Path,
    parallel: bool,
) -> dict:
    """Run one independent generation/QC job; safe to call from a worker thread."""
    mockup_path = MOCKUPS_DIR / mockup_name
    validate_file(mockup_path, "Mockup '{}'".format(mockup_name))

    output_name = f"{safe_stem(texture_path.stem)}_{safe_stem(Path(mockup_name).stem)}.png"
    output_path = run_dir / output_name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    return generate_with_qc(
        client,
        mockup_path,
        texture_path,
        mockup_name,
        output_path,
        interactive_qc=not parallel,
        show_progress=not parallel,
    )


def run_single_mode(texture_path: Path, threads: int | None = None) -> None:
    validate_file(texture_path, "Texture")
    create_output_dir()

    mockups = ask_mockup_selection()

    run_dir = create_run_dir(texture_path.stem)
    total = len(mockups)
    worker_count = resolve_worker_count(total, threads)

    print_run_header(texture_path, mockups, worker_count)

    results: list = []
    errors:  list = []
    wall_start = time.monotonic()

    if worker_count == 1:
        for i, mockup_name in enumerate(mockups, start=1):
            mockup_path = MOCKUPS_DIR / mockup_name
            print_mockup_header(i, total, mockup_name, mockup_path)

            try:
                result = _run_single_job(texture_path, mockup_name, run_dir, parallel=False)
                results.append(result)
                print_mockup_result(result)
            except Exception as exc:
                print(f"  {c(C.RED, '  ✗ Failed:')} {exc}\n")
                errors.append((mockup_name, str(exc)))
    else:
        print(f"  {c(C.CYAN, '>>')} Starting {total} jobs with {worker_count} threads ...\n")
        order = {name: i for i, name in enumerate(mockups, start=1)}

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mockify") as executor:
            futures = {
                executor.submit(_run_single_job, texture_path, name, run_dir, True): name
                for name in mockups
            }

            for future in as_completed(futures):
                mockup_name = futures[future]
                mockup_path = MOCKUPS_DIR / mockup_name
                print_mockup_header(order[mockup_name], total, mockup_name, mockup_path)
                try:
                    result = future.result()
                    results.append(result)
                    print_mockup_result(result)
                except Exception as exc:
                    print(f"  {c(C.RED, '  ✗ Failed:')} {exc}\n")
                    errors.append((mockup_name, str(exc)))

    print_summary(
        results,
        errors,
        wall_time=time.monotonic() - wall_start,
        output_dir=run_dir,
    )

# ---------------------------------------------------------------------------
# Main generation loop — Multi mode (no QC)
# ---------------------------------------------------------------------------

def run_multi_mode() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    run_dir = create_run_dir("multi")

    # Step 1: obtain the 9-design board (existing file or freshly generated)
    board_path = ask_multi_board_source(run_dir)

    # Step 2: choose what to apply the board to — multi only / choice only /
    # both / nothing (just keep the board as-is).
    targets = ask_multi_apply_targets()

    board_stem = board_path.stem

    results: list = []
    errors:  list = []

    width = 56
    line  = "─" * width

    if not targets:
        # "Nothing" chosen — board already saved/provided, nothing more to do.
        print()
        print(f"  {c(C.GREY, 'No mockup selected — keeping the board as-is.')}")
        print(f"  {c(C.GREY, 'Board file')}  {c(C.CYAN, str(board_path))}")
        # The board is the deliverable here: run the anti-provenance final pass
        # (only when it was generated into our outputs dir, never a user file).
        if board_path.parent == run_dir:
            print_antiprovenance_result(finalize_deliverable(board_path))
        print_summary(results, errors, output_dir=run_dir)
        return

    # --- multi.png ---
    if "multi" in targets:
        print()
        print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
        print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Multi mockup generation'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
        print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
        print()
        print(f"  {c(C.GREY, 'Board')}     {c(C.CYAN, board_path.name)}  {c(C.GREY, get_file_size(board_path))}")
        print(f"  {c(C.GREY, 'Mockup')}    {c(C.CYAN, MULTI_MOCKUP_FILE.name)}")
        mask_status = c(C.GREEN, f"enabled ({MULTI_MASK_FILE.name})") if (MULTI_MASK_FILE.exists() and MULTI_MASK_FILE.stat().st_size > 0) else c(C.YELLOW, "disabled (no mask file found — geometry not locked)")
        print(f"  {c(C.GREY, 'Mask')}      {mask_status}")
        print(f"  {c(C.GREY, 'Model')}     {c(C.CYAN, MODEL)}")
        print(f"  {c(C.GREY, 'Size')}      {c(C.CYAN, IMAGE_SIZE)}")
        print(f"  {c(C.GREY, 'QC')}        {c(C.GREY, 'disabled (multi mode)')}")
        print(f"  {c(C.GREY, line)}")
        print()

        output_name = f"{safe_stem(board_stem)}_multi.png"
        output_path = run_dir / output_name

        try:
            result = generate_multi_mockup(client, board_path, output_path)
            result["qc_status"] = "no_qc"
            results.append(result)
            print_mockup_result(result)
        except Exception as exc:
            print(f"  {c(C.RED, '  ✗ Failed:')} {exc}\n")
            errors.append(("multi.png", str(exc)))

    # --- choice.png ---
    if "choice" in targets:
        print()
        print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
        print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Choice selection graphic'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
        print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
        print()
        print(f"  {c(C.GREY, 'Board')}     {c(C.CYAN, board_path.name)}  {c(C.GREY, get_file_size(board_path))}")
        print(f"  {c(C.GREY, 'Mockup')}    {c(C.CYAN, CHOICE_MOCKUP_FILE.name)}")
        print(f"  {c(C.GREY, 'Model')}     {c(C.CYAN, MODEL)}")
        print(f"  {c(C.GREY, 'Size')}      {c(C.CYAN, IMAGE_SIZE)}")
        print(f"  {c(C.GREY, 'QC')}        {c(C.GREY, 'disabled (choice mode)')}")
        print(f"  {c(C.GREY, line)}")
        print()

        choice_output_name = f"{safe_stem(board_stem)}_choice.png"
        choice_output_path = run_dir / choice_output_name

        try:
            choice_result = generate_choice_mockup(client, board_path, choice_output_path)
            choice_result["qc_status"] = "no_qc"
            results.append(choice_result)
            print_mockup_result(choice_result)
        except Exception as exc:
            print(f"  {c(C.RED, '  ✗ Failed:')} {exc}\n")
            errors.append(("choice.png", str(exc)))

    print_summary(results, errors, output_dir=run_dir)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate product mockups with optional parallel API calls."
    )
    parser.add_argument(
        "texture",
        nargs="?",
        help="Texture image path (required for Single mode).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Maximum concurrent mockup jobs (1-8; automatic by default).",
    )
    parser.add_argument(
        "--archive-outputs",
        action="store_true",
        help="Archive and verify output folders, then remove the archived folders.",
    )
    args = parser.parse_args()
    if args.threads is not None and not 1 <= args.threads <= 8:
        parser.error("--threads must be between 1 and 8")
    return args


def main() -> None:
    args = parse_args()

    try:
        if args.archive_outputs:
            archive_path, folder_count, file_count = archive_output_folders()
            print(f"Archive updated: {archive_path}")
            print(f"Folders archived: {folder_count}  Files added: {file_count}")
            return

        mode = ask_run_mode()

        if mode == "archive":
            archive_path, folder_count, file_count = archive_output_folders()
            print(f"Archive updated: {archive_path}")
            print(f"Folders archived: {folder_count}  Files added: {file_count}")
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"  {c(C.RED, 'Error:')} OPENAI_API_KEY is not set (check your .env file).", file=sys.stderr)
            sys.exit(1)

        if mode == "single":
            if not args.texture:
                print(f"  {c(C.YELLOW, 'Usage:')} python src/main.py <texture_path> [--threads N]")
                sys.exit(1)

            texture_path = Path(args.texture).resolve()
            if not texture_path.exists():
                print(f"  {c(C.RED, 'Error:')} Texture not found: {texture_path}", file=sys.stderr)
                sys.exit(1)

            run_single_mode(texture_path, threads=args.threads)

        else:  # multi
            run_multi_mode()

    except KeyboardInterrupt:
        print(f"\n  {c(C.YELLOW, 'Aborted.')}", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"  {c(C.RED, 'Unexpected error:')} {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
