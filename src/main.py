#!/usr/bin/env python3

import os
import sys
import base64
import argparse
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
# Configuration
# ---------------------------------------------------------------------------

MOCKUPS = [
    "basic.png",
]

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MOCKUPS_DIR = ASSETS_DIR / "mockups"
PROMPTS_DIR = ASSETS_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
PROMPT_FILE = PROMPTS_DIR / "texture_applyer.txt"

IMAGE_SIZE = "1024x1024"
MODEL = "gpt-image-1"

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


def create_output_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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
    if mockup_name == "double.png":
        prompt += (
            "\n\nIMPORTANT:\n"
            "Apply the artwork on BOTH Kindle cases visible in the mockup.\n"
            "Both Kindle cases must use the exact same design.\n"
            "Respect perspective and lighting independently.\n"
        )
    return prompt


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def print_debug(mockup_path: Path, texture_path: Path) -> None:
    print("\n========== DEBUG ==========")
    print(f"MOCKUP:  Path: {mockup_path}  Size: {get_file_size(mockup_path)}")
    print(f"TEXTURE: Path: {texture_path}  Size: {get_file_size(texture_path)}")
    print("ROLE ASSIGNMENT:")
    print("  Image 1 -> MOCKUP BASE")
    print("  Image 2 -> TEXTURE DESIGN")
    print("===========================\n")


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(image_base64: str, output_path: Path) -> None:
    output_path.write_bytes(base64.b64decode(image_base64))
    print(f"  ✓ Saved: {output_path}  ({get_file_size(output_path)})")


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_mockup(
    client: OpenAI,
    mockup_path: Path,
    texture_path: Path,
    mockup_name: str,
    output_path: Path,
) -> None:
    print_debug(mockup_path, texture_path)
    print(f"  → Generating mockup for: {mockup_name} ...")

    base_prompt = load_prompt()
    prompt = build_prompt(base_prompt, mockup_name)

    with open(mockup_path, "rb") as mockup_file, \
         open(texture_path, "rb") as texture_file:

        response = client.images.edit(
            model=MODEL,
            image=[mockup_file, texture_file],
            prompt=prompt,
            size=IMAGE_SIZE,
        )

    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise RuntimeError("API returned no image data.")

    save_image(image_base64, output_path)


def generate_all_mockups(texture_path: Path) -> None:
    validate_file(texture_path, "Texture")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_dir = create_output_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    texture_stem = texture_path.stem

    print(f"\n{'='*50}")
    print(f"  Kindle Case Mockup Generator")
    print(f"  Texture : {texture_path}")
    print(f"  Mockups : {len(MOCKUPS)}")
    print(f"{'='*50}\n")

    errors = []

    for i, mockup_name in enumerate(MOCKUPS, start=1):
        mockup_path = MOCKUPS_DIR / mockup_name
        print(f"[{i}/{len(MOCKUPS)}] Processing: {mockup_name}")

        try:
            validate_file(mockup_path, f"Mockup '{mockup_name}'")

            output_name = f"{texture_stem}_{Path(mockup_name).stem}_{timestamp}.png"
            output_path = output_dir / output_name

            generate_mockup(client, mockup_path, texture_path, mockup_name, output_path)

        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            errors.append((mockup_name, str(exc)))

    print(f"\n{'='*50}")
    print(f"  Done.  {len(MOCKUPS) - len(errors)}/{len(MOCKUPS)} mockups generated.")
    if errors:
        print("  Errors:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 src/main.py <texture_path>")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set (check your .env file).", file=sys.stderr)
        sys.exit(1)

    try:
        generate_all_mockups(texture_path=Path(sys.argv[1]).resolve())
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()