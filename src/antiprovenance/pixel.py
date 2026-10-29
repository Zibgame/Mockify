"""
Couche pixel : perturbation qui dégrade les tatouages invisibles encodés dans
l'image elle-même (type SynthID).

Les tatouages modernes sont des perturbations étalées dans le domaine
fréquentiel (accent sur le canal vert), entraînées à survivre au bruit, au
redimensionnement et à la recompression. La chaîne ci-dessous combine donc :

  a. micro-cadrage géométrique sous-pixel (rotation/échelle/translation) ;
  b. désynchronisation sous-pixel des canaux (vert ciblé en priorité) ;
  c. dé-bruitage + re-bruitage du résidu haute fréquence ;
  d. recompression JPEG en mémoire (attaque fréquentielle) ;
  e. quantification dithered (point faible documenté du décodeur).

Ces opérations **augmentent la difficulté de détection** ; elles ne garantissent
pas la suppression du tatouage (voir README pour les mesures empiriques).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

__all__ = ["Strength", "PROFILES", "get_strength", "pixel_sanitize", "psnr"]


@dataclass
class Strength:
    """Paramètres de la chaîne pixel. Les valeurs par défaut = profil 'balanced'."""
    name: str = "balanced"
    # (a) micro-cadrage géométrique — coûteux en qualité, gain marginal d'après la
    # littérature (SynthID survit aux perturbations géométriques). Réservé aux
    # profils élevés comme assurance supplémentaire.
    rotate_deg: float = 0.0           # ±
    scale_sigma: float = 0.0          # ± autour de 1.0
    translate_px: float = 0.0         # ±
    # (b) désynchronisation sous-pixel du canal vert (tatouage green-dominant)
    green_desync_px: float = 0.35     # ±
    other_desync_px: float = 0.12     # ± sur R/B
    # (c) dé-bruitage + re-bruitage du résidu HF
    denoise_sigma: float = 0.55
    residual_gain: float = 1.00
    noise_sigma: float = 1.80
    # (d) recompression JPEG en mémoire
    jpeg_quality: Optional[int] = 96
    # (e) quantification dithered (8 = désactivé)
    dither_bits: int = 7
    # graine (None => aléatoire à chaque appel)
    seed: Optional[int] = None


PROFILES: dict[str, Strength] = {
    "off": Strength(
        name="off", rotate_deg=0.0, scale_sigma=0.0, translate_px=0.0,
        green_desync_px=0.0, other_desync_px=0.0, denoise_sigma=0.0,
        noise_sigma=0.0, jpeg_quality=None, dither_bits=8,
    ),
    "light": Strength(
        name="light", rotate_deg=0.0, scale_sigma=0.0, translate_px=0.0,
        green_desync_px=0.15, other_desync_px=0.05, denoise_sigma=0.40,
        noise_sigma=1.0, jpeg_quality=98, dither_bits=8,
    ),
    "balanced": Strength(),
    "aggressive": Strength(
        name="aggressive", rotate_deg=0.15, scale_sigma=0.003, translate_px=0.8,
        green_desync_px=0.50, other_desync_px=0.18, denoise_sigma=0.70,
        residual_gain=0.95, noise_sigma=3.0, jpeg_quality=92, dither_bits=6,
    ),
    "max": Strength(
        name="max", rotate_deg=0.25, scale_sigma=0.006, translate_px=1.2,
        green_desync_px=0.70, other_desync_px=0.25, denoise_sigma=0.85,
        residual_gain=0.90, noise_sigma=4.5, jpeg_quality=90, dither_bits=5,
    ),
}


def get_strength(name: str | Strength | None) -> Strength:
    """Résout un nom de profil (ou un Strength) en une copie indépendante."""
    if isinstance(name, Strength):
        return Strength(**asdict(name))
    key = (name or "balanced").strip().lower()
    if key not in PROFILES:
        key = "balanced"
    return Strength(**asdict(PROFILES[key]))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR (dB) entre deux images float [0,255]."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def _subpixel_affine(arr: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """Rotation + échelle + translation sous-pixel, taille préservée (bord réfléchi padé)."""
    h, w = arr.shape[:2]
    pad = 16
    padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    cx, cy = (w + 2 * pad) / 2.0, (h + 2 * pad) / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    warped = cv2.warpAffine(
        padded, m, (w + 2 * pad, h + 2 * pad),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped[pad:pad + h, pad:pad + w]


def _channel_shift(channel: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translation sous-pixel d'un canal unique."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return channel
    h, w = channel.shape
    pad = 8
    padded = cv2.copyMakeBorder(channel, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        padded, m, (w + 2 * pad, h + 2 * pad),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101,
    )
    return shifted[pad:pad + h, pad:pad + w]


def _denoise_renoise(arr: np.ndarray, sigma: float, residual_gain: float,
                     noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Dé-bruitage puis réinjection du résidu + bruit indépendant."""
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    residual = arr - blur
    noise = rng.normal(0.0, noise_sigma, arr.shape).astype(np.float32)
    return blur + residual * residual_gain + noise


def _jpeg_roundtrip(arr: np.ndarray, quality: int) -> np.ndarray:
    """Encode/décode JPEG en mémoire (4:4:4, sans sous-échantillonnage chroma)."""
    bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
        params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
    ok, enc = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise RuntimeError("JPEG re-encode failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)


def _dither_quantize(arr: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    """Quantification à `bits` bits/canal avec dither triangulaire."""
    if bits >= 8:
        return arr
    levels = (1 << bits) - 1
    step = 255.0 / levels
    dither = (rng.random(arr.shape).astype(np.float32) - rng.random(arr.shape).astype(np.float32))
    val = arr + dither * (step * 0.5)
    return (np.round(np.clip(val, 0.0, 255.0) / step) * step).astype(np.float32)


def pixel_sanitize(arr: np.ndarray, strength: Strength | str | None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Chaîne pixel complète. `arr` = float32 RGB [0,255]. Retourne float32 [0,255].
    """
    s = get_strength(strength)
    rng = rng or np.random.default_rng(s.seed)
    out = arr.astype(np.float32)

    if s.rotate_deg or s.scale_sigma or s.translate_px:
        angle = rng.uniform(-s.rotate_deg, s.rotate_deg)
        scale = 1.0 + rng.uniform(-s.scale_sigma, s.scale_sigma)
        tx = rng.uniform(-s.translate_px, s.translate_px)
        ty = rng.uniform(-s.translate_px, s.translate_px)
        out = _subpixel_affine(out, angle, scale, tx, ty)

    if s.green_desync_px or s.other_desync_px:
        chans = []
        for i in range(3):
            amp = s.green_desync_px if i == 1 else s.other_desync_px
            dx = rng.uniform(-amp, amp) if amp else 0.0
            dy = rng.uniform(-amp, amp) if amp else 0.0
            chans.append(_channel_shift(out[:, :, i], dx, dy))
        out = np.stack(chans, axis=2)

    if s.denoise_sigma > 0 or s.noise_sigma > 0:
        out = _denoise_renoise(out, s.denoise_sigma, s.residual_gain, s.noise_sigma, rng)

    if s.jpeg_quality:
        out = _jpeg_roundtrip(np.clip(out, 0, 255), s.jpeg_quality)

    if s.dither_bits < 8:
        out = _dither_quantize(np.clip(out, 0, 255), s.dither_bits, rng)

    return np.clip(out, 0.0, 255.0)
