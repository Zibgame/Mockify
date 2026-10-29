# antiprovenance

Module autonome pour **rendre l'origine IA d'une image beaucoup plus difficile à
établir**. Conçu pour être appelé en fin de pipeline de génération d'images, et
réutilisable tel quel dans n'importe quel projet.

Il traite les deux couches de provenance réellement présentes dans les images des
générateurs modernes :

| Couche | Support | Traitement |
| --- | --- | --- |
| **C2PA** (Content Credentials) | chunk PNG `caBX` (coffre JUMBF signé) | **supprimé à 100 %** |
| **Tatouage pixel** (type SynthID) | perturbation invisible dans les pixels | perturbé (et régénérable) |

## Installation

```bash
# depuis ce dépôt
pip install -e src

# ou : copier le dossier antiprovenance/ dans votre projet et l'importer
```

Dépendances : `numpy`, `pillow`, `opencv-python`. Optionnel : `openai` (pour la
vérification), `torch` + `diffusers` (pour la régénération locale).

## Utilisation

```python
from antiprovenance import finalize_image

rapport = finalize_image(
    "output.png",
    strength="balanced",   # off | light | balanced | aggressive | max
    regenerate="off",      # off | local | http
    verify=True,           # interroge l'API de provenance
)
print(rapport["sanitize"]["psnr_db"])
print(rapport["provenance"]["results"])   # {'c2pa': 'not_detected', 'synthid': ...}
```

API bas niveau :

```python
from antiprovenance import sanitize_image, scrub_png_container, has_provenance_chunks

sanitize_image("output.png", strength="aggressive")   # pixels + conteneur
scrub_png_container("output.png")                     # conteneur seul (pixels intacts)
has_provenance_chunks("output.png")                   # -> [] quand c'est propre
```

Ligne de commande :

```bash
python -m antiprovenance image.png --strength balanced --verify
python -m antiprovenance image.png --audit          # liste les chunks de provenance
antiprovenance image.png --strength aggressive      # si installé
```

## Profils de perturbation pixel

| Profil | PSNR typique | Effet visuel |
| --- | --- | --- |
| `off` | ∞ | aucun traitement pixel |
| `light` | ~44 dB | imperceptible |
| `balanced` (défaut) | ~40 dB | imperceptible |
| `aggressive` | ~32 dB | léger flou |
| `max` | ~28 dB | dégradation visible |

Chaîne appliquée (`pixel.py`) : désynchronisation sous-pixel du canal vert,
dé-bruitage + re-bruitage du résidu haute fréquence, recompression JPEG en
mémoire, quantification dithered, et — profils élevés — micro-cadrage
géométrique.

## Résultats mesurés (vérité terrain, API `content_provenance_checks`)

| Image | `c2pa` | `synthid` |
| --- | --- | --- |
| générée brute | detected | detected |
| après `balanced` | not_detected | detected |
| après `aggressive` | not_detected | detected |
| binarisation 1 bit | not_detected | not_detected |

**Ce qu'il faut retenir, sans enjolivure :**

- La suppression **C2PA est totale** dès `light`.
- Le tatouage pixel (SynthID) **survit à toutes les transformations
  déterministes testées** (bruit gaussien, resize 0.5×/0.25×, double JPEG,
  requantification jusqu'à 3 bits, attaque ciblée canal vert, CLAHE). Seule la
  binarisation 1 bit le supprime — en détruisant l'image (PSNR ~13 dB). C'est
  cohérent avec la littérature (encodeur/décodeur entraîné contre ces
  transformations).
- Le seul levier qui blanchit le tatouage **en conservant l'image** est la
  **régénération** par un modèle tiers (couple encodeur/décodeur différent) :
  `regenerate="local"` ou `"http"`.

## Régénération

```text
MOCKIFY_REGENERATE=local|http|off            (variable du projet appelant)
MOCKIFY_REGENERATE_STRENGTH=0.75
MOCKIFY_REGENERATE_MODEL=...                 (backend local)
MOCKIFY_REGENERATE_URL=...                   (backend http, endpoint img2img)
MOCKIFY_REGENERATE_TOKEN=...                 (optionnel)
```

- `local` : `diffusers` img2img (GPU conseillé).
- `http` : tout endpoint img2img (SD WebUI, ComfyUI, API maison) renvoyant
  l'image (binaire ou base64 JSON).

## Vérification de provenance

```python
from antiprovenance import check_provenance
check_provenance("output.png", api_key=None)   # lit OPENAI_API_KEY
# -> {"ok": True, "results": {"c2pa": "not_detected", "synthid": "detected"}, "raw": {...}}
```

Utilise l'API REST directement : aucune contrainte de version du SDK `openai`.

## Structure

```text
antiprovenance/
  __init__.py     API publique
  png.py          couche conteneur (chunks, scrub)
  pixel.py        profils + transformations pixel
  pipeline.py     sanitize_image / finalize_image
  regenerate.py   backends de regénération (local / http)
  verify.py       vérification via l'API de provenance
  cli.py          interface ligne de commande
  __main__.py     python -m antiprovenance
```
