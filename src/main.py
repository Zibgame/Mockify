#!/usr/bin/env python3

import os
import sys
import base64
import time
import json
import threading
from pathlib import Path
from datetime import datetime

from openai import OpenAI

# Load .env from the project root (parent of src/)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
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

BASE_DIR          = Path(__file__).resolve().parent.parent
ASSETS_DIR        = BASE_DIR / "assets"
MOCKUPS_DIR       = ASSETS_DIR / "mockups"
PROMPTS_DIR       = ASSETS_DIR / "prompts"
OUTPUTS_DIR       = BASE_DIR / "outputs"
PROMPT_FILE       = PROMPTS_DIR / "texture_applyer.txt"
CHECKER_PROMPT_FILE = PROMPTS_DIR / "ImageChecker.txt"

IMAGE_SIZE = "1024x1024"
MODEL      = "gpt-image-2"
CHECKER_MODEL = "gpt-4o"

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


def load_prompt() -> str:
    validate_file(PROMPT_FILE, "Prompt file")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def load_checker_prompt() -> str:
    validate_file(CHECKER_PROMPT_FILE, "ImageChecker prompt file")
    content = CHECKER_PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("ImageChecker.txt is empty — please add a quality-check prompt.")
    return content


def create_output_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR

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
# Prompt construction
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

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(image_base64: str, output_path: Path) -> None:
    output_path.write_bytes(base64.b64decode(image_base64))

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

# ---------------------------------------------------------------------------
# Image quality checker
# ---------------------------------------------------------------------------

def check_image_quality(client: OpenAI, image_path: Path, texture_path: Path) -> dict:
    """
    Send BOTH the original texture AND the generated mockup to GPT-4o Vision.
    Image 1 = original texture (reference), Image 2 = generated mockup result.
    Returns parsed JSON dict with keys: has_problem, problems, severity, details.
    """
    checker_prompt = load_checker_prompt()
    texture_b64 = image_to_base64(texture_path)
    result_b64  = image_to_base64(image_path)

    spinner = Spinner("Checking image quality ...")
    spinner.start()
    try:
        response = client.chat.completions.create(
            model=CHECKER_MODEL,
            max_tokens=512,
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
                                "detail": "high",
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
                                "detail": "high",
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
        spinner.stop(success=False)
        raise

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
# Core generation
# ---------------------------------------------------------------------------

def generate_mockup(
    client: OpenAI,
    mockup_path: Path,
    texture_path: Path,
    mockup_name: str,
    output_path: Path,
    extra_instructions: str = "",
) -> dict:
    base_prompt = load_prompt()
    prompt = build_prompt(base_prompt, mockup_name)
    if extra_instructions:
        prompt += extra_instructions

    spinner = Spinner(f"Calling API for {c(C.BOLD, mockup_name)} ...")
    spinner.start()

    t_start = time.monotonic()
    try:
        with open(mockup_path, "rb") as mockup_file, \
             open(texture_path, "rb") as texture_file:
            response = client.images.edit(
                model=MODEL,
                image=[mockup_file, texture_file],
                prompt=prompt,
                size=IMAGE_SIZE,
            )
    except Exception:
        spinner.stop(success=False)
        raise

    elapsed = time.monotonic() - t_start
    spinner.stop(success=True)

    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise RuntimeError("API returned no image data.")

    save_image(image_base64, output_path)

    usage = getattr(response, "usage", None)
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
        "elapsed":       elapsed,
        "cost_usd":      cost,
        "cost_source":   source,
        "input_tokens":  input_tok,
        "output_tokens": output_tok,
        "output_path":   output_path,
    }

# ---------------------------------------------------------------------------
# Generation + QC loop (first pass + one retry)
# ---------------------------------------------------------------------------

def generate_with_qc(
    client: OpenAI,
    mockup_path: Path,
    texture_path: Path,
    mockup_name: str,
    output_path: Path,
) -> dict:
    """
    Generate a mockup, run QC, handle retry logic.
    Returns the final result dict (may include qc_status key).
    Raises on hard failure.
    """

    # --- First generation ---
    result = generate_mockup(client, mockup_path, texture_path, mockup_name, output_path)

    # --- QC check ---
    check = check_image_quality(client, output_path, texture_path)
    print_check_result(check)
    print()

    if not check.get("has_problem", False):
        result["qc_status"] = "pass"
        return result

    # --- Problem detected: ask user ---
    skip = ask_skip_or_retry(mockup_name)

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
        print(f"  {c(C.CYAN, '  ℹ')} Injecting {len(fix_lines)} correction(s) into retry prompt.")

    # --- Delete the bad image before retry ---
    if output_path.exists():
        output_path.unlink()
        print(f"  {c(C.GREY, '  🗑')} Deleted failed image: {c(C.DIM, output_path.name)}\n")

    # --- Retry once (same output path, corrected prompt) ---
    retry_result = generate_mockup(
        client, mockup_path, texture_path, mockup_name, output_path,
        extra_instructions=avoid_block,
    )

    # QC on retry
    check2 = check_image_quality(client, output_path, texture_path)
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
            print(f"  {c(C.GREY, '  🗑')} Deleted retry image: {c(C.DIM, output_path.name)}")
        print(f"  {c(C.RED, '  ✗ Retry also failed QC — stopping this mockup.')}\n")
        raise RuntimeError(
            f"QC failed twice for {mockup_name}: {check2.get('details', 'unknown error')}"
        )

# ---------------------------------------------------------------------------
# Interactive mockup selector
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
# CLI helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 30

def progress_bar(current: int, total: int) -> str:
    filled = int(BAR_WIDTH * current / total) if total > 0 else 0
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{c(C.CYAN, bar)}] {current}/{total}"

def print_run_header(texture_path: Path, mockups: list) -> None:
    width = 56
    line  = "─" * width
    print()
    print(f"  {c(C.BOLD + C.MAGENTA, '╔' + '═' * width + '╗')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '║')}  {c(C.BOLD + C.WHITE, 'MOCKIFY  —  Kindle Case Mockup Generator'):<{width - 2}}{c(C.BOLD + C.MAGENTA, '║')}")
    print(f"  {c(C.BOLD + C.MAGENTA, '╚' + '═' * width + '╝')}")
    print()
    print(f"  {c(C.GREY, 'Texture')}   {c(C.CYAN, str(texture_path.name))}  {c(C.GREY, get_file_size(texture_path))}")
    print(f"  {c(C.GREY, 'Model')}     {c(C.CYAN, MODEL)}")
    print(f"  {c(C.GREY, 'Checker')}   {c(C.CYAN, CHECKER_MODEL)}")
    print(f"  {c(C.GREY, 'Size')}      {c(C.CYAN, IMAGE_SIZE)}")
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

def print_mockup_result(result: dict) -> None:
    qc_status  = result.get("qc_status", "")
    qc_icons   = {
        "pass":          c(C.GREEN,  "✓ QC"),
        "pass_on_retry": c(C.CYAN,   "✓ QC (retry)"),
        "skipped":       c(C.YELLOW, "⚠ QC skipped"),
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
    print()

def print_summary(results: list, errors: list) -> None:
    total      = len(results) + len(errors)
    ok         = len(results)
    total_cost = sum(r["cost_usd"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    has_est    = any(r["cost_source"] == "estimated" for r in results)

    qc_pass    = sum(1 for r in results if r.get("qc_status") == "pass")
    qc_retry   = sum(1 for r in results if r.get("qc_status") == "pass_on_retry")
    qc_skipped = sum(1 for r in results if r.get("qc_status") == "skipped")

    line = "─" * 52
    print(f"  {c(C.GREY, line)}")
    print(f"  {c(C.BOLD, 'Summary')}")
    print()
    print(f"  {c(C.GREY, 'Generated')}   {c(C.GREEN, str(ok))}/{total}")
    print(f"  {c(C.GREY, 'QC results')}  "
          f"{c(C.GREEN, str(qc_pass))} pass  "
          f"{c(C.CYAN, str(qc_retry))} retry-pass  "
          f"{c(C.YELLOW, str(qc_skipped))} skipped")
    print(f"  {c(C.GREY, 'Total time')}  {c(C.CYAN, str(round(total_time, 1)) + 's')}")

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
    print(f"  {c(C.GREY, 'Output dir')}  {c(C.CYAN, str(OUTPUTS_DIR))}")
    print()

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all_mockups(texture_path: Path, mockups: list) -> None:
    validate_file(texture_path, "Texture")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    create_output_dir()

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    texture_stem = texture_path.stem
    total        = len(mockups)

    print_run_header(texture_path, mockups)

    results: list = []
    errors:  list = []

    for i, mockup_name in enumerate(mockups, start=1):
        mockup_path = MOCKUPS_DIR / mockup_name
        print_mockup_header(i, total, mockup_name, mockup_path)

        try:
            validate_file(mockup_path, "Mockup '{}'".format(mockup_name))
            output_name = "{}_{}_{}.png".format(texture_stem, Path(mockup_name).stem, timestamp)
            output_path = OUTPUTS_DIR / output_name

            result = generate_with_qc(client, mockup_path, texture_path, mockup_name, output_path)
            results.append(result)
            print_mockup_result(result)

        except Exception as exc:
            print(f"  {c(C.RED, '  ✗ Failed:')} {exc}\n")
            errors.append((mockup_name, str(exc)))

    print_summary(results, errors)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"  {c(C.YELLOW, 'Usage:')} python src/main.py <texture_path>")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"  {c(C.RED, 'Error:')} OPENAI_API_KEY is not set (check your .env file).", file=sys.stderr)
        sys.exit(1)

    texture_path = Path(sys.argv[1]).resolve()
    if not texture_path.exists():
        print(f"  {c(C.RED, 'Error:')} Texture not found: {texture_path}", file=sys.stderr)
        sys.exit(1)

    try:
        selected = ask_mockup_selection()
        generate_all_mockups(texture_path=texture_path, mockups=selected)
    except KeyboardInterrupt:
        print(f"\n  {c(C.YELLOW, 'Aborted.')}", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"  {c(C.RED, 'Unexpected error:')} {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()