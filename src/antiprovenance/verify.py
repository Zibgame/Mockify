#!/usr/bin/env python3
"""
Vérification de provenance via l'API OpenAI `content_provenance_checks`.

Renvoie, pour une image, le verdict des signaux supportés :
  * `c2pa`   — manifeste signé (chunk PNG `caBX`)
  * `synthid` — tatouage pixel invisible

Sert de test objectif dans le pipeline : après la passe anti-provenance on sait
exactement quel signal subsiste, sans se fier à une heuristique.

Nécessite OPENAI_API_KEY. L'API exige le SDK openai >= 2.52.0 côté SDK officiel ;
ici on fait l'appel REST directement, donc aucune contrainte de version.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_ENDPOINT = "https://api.openai.com/v1/content_provenance_checks"
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp"}


def check_provenance(image_path: Path, api_key: Optional[str] = None,
                     timeout: int = 60) -> dict:
    """
    Retourne {"ok": bool, "results": {type: outcome}, "raw": {...}}.
    `ok=False` si l'appel échoue (clé absente, rate limit, réseau...).
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return {"ok": False, "error": "OPENAI_API_KEY missing", "results": {}, "raw": None}

    p = Path(image_path)
    mime = _MIME.get(p.suffix.lower(), "application/octet-stream")
    boundary = "----mockifyprov"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + p.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        _ENDPOINT, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300]
        return {"ok": False, "error": f"HTTP {exc.code}: {detail}", "results": {}, "raw": None}
    except Exception as exc:  # réseau, timeout...
        return {"ok": False, "error": str(exc), "results": {}, "raw": None}

    results = {entry.get("type"): entry.get("outcome") for entry in data.get("results", [])}
    return {"ok": True, "results": results, "raw": data}


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        res = check_provenance(Path(arg))
        print(f"== {arg}")
        if res["ok"]:
            for kind, outcome in res["results"].items():
                print(f"    {kind:8s} {outcome}")
        else:
            print(f"    ERROR {res['error']}")
