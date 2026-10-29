"""
Orchestration : la passe complète appliquée à une image.

`finalize_image()` est le point d'entrée haut niveau : (régénération optionnelle)
→ perturbation pixel → scrub conteneur → (vérification optionnelle). C'est ce
qu'un projet de génération d'images appelle sur chaque fichier produit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import png
from .pixel import Strength, get_strength, pixel_sanitize, psnr

__all__ = ["sanitize_image", "finalize_image"]

_FORMAT_ALIASES = {"jpg": "jpeg"}


def sanitize_image(
    image_path: Path | str,
    strength: str | Strength = "balanced",
    save_format: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Applique la perturbation pixel puis le nettoyage conteneur.

    `save_format` : None (inchangé) ou "png" / "jpeg" / "webp".
    `seed` : force la graine (résultats reproductibles) au lieu du profil.
    Retourne un rapport détaillé (dict sérialisable JSON).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    s = get_strength(strength)
    if seed is not None:
        s.seed = seed
    rng = np.random.default_rng(s.seed)

    img = Image.open(image_path)
    mode = img.mode
    has_alpha = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)
    img.load()
    rgba = img.convert("RGBA") if has_alpha else img.convert("RGB")
    arr = np.asarray(rgba).astype(np.float32)

    fmt = (save_format or image_path.suffix.lstrip(".")).lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)

    provenance_before: list[str] = []
    if image_path.suffix.lower() == ".png":
        try:
            provenance_before = png.has_provenance_chunks(image_path)
        except ValueError:
            provenance_before = []

    report: dict = {
        "strength": s.name,
        "format": fmt,
        "has_alpha": bool(has_alpha),
        "pixels_modified": False,
        "psnr_db": None,
        "jpeg_quality": s.jpeg_quality,
        "dither_bits": s.dither_bits,
        "provenance_chunks_before": provenance_before,
        "provenance_chunks_after": [],
    }

    # --- Couche pixel ---
    if s.name != "off":
        rgb = arr[:, :, :3]
        sanitized = pixel_sanitize(rgb, s, rng)
        report["pixels_modified"] = True
        val = psnr(rgb, sanitized)
        report["psnr_db"] = round(float(val), 2) if np.isfinite(val) else None
        if has_alpha:
            arr = np.dstack([sanitized, arr[:, :, 3]]).astype(np.float32)
        else:
            arr = sanitized.astype(np.float32)

    # --- Écriture ---
    out_arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGBA" if has_alpha else "RGB")

    target = image_path
    if save_format is not None and fmt != image_path.suffix.lstrip(".").lower():
        target = image_path.with_suffix("." + ("jpg" if fmt == "jpeg" else fmt))

    if fmt == "png":
        out_img.save(target, format="PNG", optimize=True)
    elif fmt == "jpeg":
        out_img.convert("RGB").save(target, format="JPEG", quality=95, subsampling=0)
    elif fmt == "webp":
        out_img.save(target, format="WEBP", quality=98, method=6)
    else:
        out_img.save(target)

    # --- Couche conteneur (PNG uniquement) ---
    if target.suffix.lower() == ".png":
        png.scrub_png_container(target)
        report["provenance_chunks_after"] = png.has_provenance_chunks(target)
    else:
        report["provenance_chunks_after"] = []

    report["output_path"] = str(target)
    if target != image_path and image_path.exists():
        image_path.unlink()
    return report


def finalize_image(
    image_path: Path | str,
    strength: str | Strength = "balanced",
    regenerate: str = "off",
    regenerate_strength: float = 0.75,
    verify: bool = False,
    save_format: Optional[str] = None,
) -> dict:
    """
    Passe complète, dans l'ordre :

      1. régénération optionnelle (le seul levier réel contre SynthID) ;
      2. perturbation pixel + scrub conteneur (`sanitize_image`) ;
      3. vérification de provenance optionnelle (appel API).

    Retourne {"sanitize": {...}, "regenerate": {...}, "provenance": {...}|None}.
    """
    from .regenerate import regenerate_image
    from .verify import check_provenance

    result: dict = {"regenerate": None, "sanitize": None, "provenance": None}

    result["regenerate"] = regenerate_image(
        Path(image_path), mode=regenerate or "off", strength=regenerate_strength)

    result["sanitize"] = sanitize_image(image_path, strength=strength, save_format=save_format)

    if verify:
        result["provenance"] = check_provenance(result["sanitize"]["output_path"])

    return result
