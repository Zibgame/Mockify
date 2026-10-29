"""Interface en ligne de commande : `python -m antiprovenance ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .pipeline import finalize_image, sanitize_image
from .pixel import PROFILES
from .png import list_png_chunks
from .verify import check_provenance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="antiprovenance",
        description="Rend une image générée par IA plus difficile à authentifier (métadonnées + pixels).",
    )
    p.add_argument("images", nargs="+", help="une ou plusieurs images")
    p.add_argument("--strength", default="balanced", choices=list(PROFILES),
                   help="profil de perturbation pixel (défaut: balanced)")
    p.add_argument("--format", default=None, choices=["png", "jpeg", "jpg", "webp"],
                   help="format de sortie (défaut: inchangé)")
    p.add_argument("--regenerate", default="off", choices=["off", "local", "http"],
                   help="régénération par un modèle tiers (défaut: off)")
    p.add_argument("--regenerate-strength", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=None, help="graine (résultats reproductibles)")
    p.add_argument("--verify", action="store_true", help="vérifie la provenance après coup (API)")
    p.add_argument("--audit", action="store_true", help="liste seulement les chunks PNG de provenance")
    p.add_argument("--json", action="store_true", help="sortie JSON brute")
    p.add_argument("--version", action="version", version=f"antiprovenance {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for raw in args.images:
        path = Path(raw)
        if args.audit:
            found = [c for c in list_png_chunks(path)
                     if c in ("caBX", "c2pa", "tEXt", "zTXt", "iTXt", "eXIf", "tIME", "XMP ", "xmp ")]
            print(f"{path}: provenance chunks = {found or 'aucun'}")
            continue

        if args.regenerate != "off" or args.verify:
            report = finalize_image(
                path, strength=args.strength, regenerate=args.regenerate,
                regenerate_strength=args.regenerate_strength, verify=args.verify,
                save_format=args.format,
            )
        else:
            report = {"sanitize": sanitize_image(
                path, strength=args.strength, save_format=args.format, seed=args.seed)}

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            san = report["sanitize"]
            after = san["provenance_chunks_after"]
            psnr_txt = f", PSNR {san['psnr_db']} dB" if san.get("psnr_db") else ""
            print(f"{path}: {san['strength']}{psnr_txt}, "
                  f"provenance retirée={san['provenance_chunks_before'] or 'aucune'}, "
                  f"restante={after or 'aucune'}")
            prov = report.get("provenance")
            if prov:
                if prov.get("ok"):
                    print("   vérification:", prov["results"])
                else:
                    print("   vérification: échec —", prov.get("error"))
    return 0
