"""
antiprovenance
==============

Boîte à outils autonome pour **rendre l'origine IA d'une image beaucoup plus
difficile à établir** : suppression des métadonnées de provenance signées (C2PA),
perturbation des tatouages invisibles encodés dans les pixels (type SynthID),
régénération optionnelle par un modèle tiers, et vérification objective.

Pensé comme un module réutilisable : aucune dépendance à un projet particulier.
Conçu pour être appelé en fin de pipeline de génération d'images.

Fonction principale ::

    from antiprovenance import finalize_image

    rapport = finalize_image(
        "output.png",
        strength="balanced",        # off | light | balanced | aggressive | max
        regenerate="off",           # off | local | http
        verify=True,                # vérifie via l'API de provenance
    )

Dépendances : numpy, pillow, opencv-python (et openai uniquement pour `verify`).

Avertissement d'exactitude : la suppression de C2PA est totale ; celle du
tatouage pixel SynthID **ne l'est pas** par les transformations déterministes.
Voir le README du package pour les mesures empiriques.
"""

from .pipeline import finalize_image, sanitize_image
from .pixel import PROFILES, Strength, get_strength, pixel_sanitize, psnr
from .png import has_provenance_chunks, list_png_chunks, scrub_png_container
from .regenerate import regenerate_image
from .verify import check_provenance

__version__ = "1.0.0"

__all__ = [
    "finalize_image",
    "sanitize_image",
    "regenerate_image",
    "check_provenance",
    "scrub_png_container",
    "list_png_chunks",
    "has_provenance_chunks",
    "pixel_sanitize",
    "get_strength",
    "psnr",
    "Strength",
    "PROFILES",
    "__version__",
]
