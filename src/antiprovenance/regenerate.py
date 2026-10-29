#!/usr/bin/env python3
"""
Étape optionnelle de régénération (repaint) — le seul levier qui neutralise
réellement le tatouage pixel SynthID en conservant l'image.

Constat empirique (vérifié via l'API content_provenance_checks d'OpenAI) :
toutes les transformations déterministes testées (bruit, redimensionnement,
double recompression JPEG, quantification dithered jusqu'à 3 bits, attaques
ciblées canal vert) laissent l'outcome `synthid = detected`. Seule une
régénération par un couple encodeur/décodeur DIFFÉRENT (donc un autre modèle
génératif, pas Gemini et pas OpenAI) blanchit l'image. La binarisation 1 bit le
supprime aussi, mais détruit l'image.

Ce module fournit deux backends interchangeables :

  * `local` — diffusers (Stable Diffusion img2img). Requiert torch + diffusers
              et, en pratique, un GPU.
  * `http`  — n'importe quel service img2img exposé par HTTP (SD WebUI, ComfyUI,
              un endpoint maison, une API tierce...).

Les deux sont DÉSACTIVÉS par défaut : sans backend configuré, la fonction
retourne proprement sans rien modifier.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

_DEFAULT_PROMPT = (
    "high quality product photo, photorealistic, natural lighting, "
    "sharp details, faithful to the original composition"
)


def regenerate_image(
    image_path: Path,
    mode: str = "off",
    prompt: Optional[str] = None,
    strength: float = 0.75,
    model_id: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> dict:
    """
    Régénère `image_path` via un modèle tiers. Écrase le fichier en place et
    retourne un rapport. `mode` ∈ {"off", "local", "http"}.
    """
    mode = (mode or "off").strip().lower()
    report: dict = {"mode": mode, "applied": False, "reason": ""}

    if mode == "off":
        report["reason"] = "disabled"
        return report
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    prompt = prompt or _DEFAULT_PROMPT

    if mode == "local":
        out = _regenerate_local(image_path, prompt, strength, model_id)
    elif mode == "http":
        out = _regenerate_http(image_path, prompt, strength, endpoint)
    else:
        report["reason"] = f"unknown mode {mode!r}"
        return report

    # détruit tout métadonnée du backend et réécrit sans EXIF/C2PA
    clean = out.convert("RGB")
    clean.save(image_path, format="PNG", optimize=True)

    report["applied"] = True
    report["reason"] = "ok"
    report["backend"] = mode
    report["strength"] = strength
    return report


# ---------------------------------------------------------------------------
# Backend local (diffusers)
# ---------------------------------------------------------------------------

def _regenerate_local(image_path: Path, prompt: str, strength: float,
                      model_id: Optional[str]) -> Image.Image:
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RuntimeError(
            "Backend 'local' indisponible : torch/diffusers ne sont pas installés.\n"
            "  pip install torch diffusers accelerate safetensors\n"
            "Ou utilisez le backend 'http'."
        ) from exc

    model_id = model_id or os.getenv(
        "MOCKIFY_REGENERATE_MODEL",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    src = Image.open(image_path).convert("RGB")
    result = pipe(
        prompt=prompt,
        image=src,
        strength=float(strength),
        guidance_scale=float(os.getenv("MOCKIFY_REGENERATE_CFG", "7.0")),
        num_inference_steps=int(os.getenv("MOCKIFY_REGENERATE_STEPS", "40")),
    ).images[0]
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Backend HTTP générique
# ---------------------------------------------------------------------------

def _regenerate_http(image_path: Path, prompt: str, strength: float,
                     endpoint: Optional[str]) -> Image.Image:
    endpoint = endpoint or os.getenv("MOCKIFY_REGENERATE_URL")
    if not endpoint:
        raise RuntimeError(
            "Backend 'http' sans URL : définissez MOCKIFY_REGENERATE_URL "
            "(endpoint img2img renvoyant l'image produite)."
        )

    boundary = "----mockifyimg2img"
    payload = image_path.read_bytes()
    fields = {
        "prompt": prompt,
        "strength": str(strength),
        "denoising_strength": str(strength),
    }
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="image.png"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode() + payload + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    token = os.getenv("MOCKIFY_REGENERATE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(endpoint, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=int(os.getenv("MOCKIFY_REGENERATE_TIMEOUT", "300"))) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read()

    if ctype.startswith("image/"):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    # réponse JSON : base64 sous une clé "image" / "b64_json" / "images"[0]
    data = json.loads(raw.decode("utf-8", "ignore"))
    if isinstance(data, dict):
        for key in ("image", "b64_json", "images"):
            if key in data:
                val = data[key]
                if isinstance(val, list) and val:
                    val = val[0]
                    if isinstance(val, dict):
                        val = val.get("image") or val.get("b64_json")
                if isinstance(val, str):
                    if val.startswith("data:image"):
                        val = val.split(",", 1)[1]
                    return Image.open(io.BytesIO(base64.b64decode(val))).convert("RGB")
    raise RuntimeError("Réponse img2img non reconnue (ni image binaire ni base64 JSON).")
