"""
Couche conteneur : suppression des métadonnées de provenance/traçabilité d'un PNG.

Le manifeste C2PA produit par les générateurs d'images voyage dans un chunk PNG
non standard (`caBX`, superbox JUMBF signée). On réécrit le fichier en ne
conservant que les chunks d'image et de transparence, avec CRC recalculés. Les
pixels (`IDAT`) sont recopiés à l'octet — cette étape ne modifie jamais l'image.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import Iterator

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Chunks critiques + couleur/transparence légitimes (jamais de provenance).
CHUNKS_TO_KEEP = {
    b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS",
    b"cHRM", b"gAMA", b"iCCP", b"sRGB",
}

# Chunks typiques porteurs de provenance / traçabilité / identité d'éditeur.
PROVENANCE_CHUNK_TYPES = {
    b"caBX",   # C2PA (Content Credentials, coffre JUMBF)
    b"c2pa",   # variante de nommage
    b"tEXt", b"zTXt", b"iTXt",   # texte libre (Software, Comment, parameters...)
    b"eXIf",   # EXIF
    b"tIME",   # horodatage
    b"XMP ", b"xmp ",  # XMP
}


def iter_chunks(data: bytes, image_path: Path | str = "<bytes>") -> Iterator[tuple[bytes, bytes, int]]:
    """Itère (chunk_type, chunk_data, length) en validant signatures et CRC."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a valid PNG: {image_path}")
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"Truncated PNG chunk in: {image_path}")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ValueError(f"Invalid PNG chunk length in: {image_path}")
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length:chunk_end])[0]
        crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if crc != expected_crc:
            raise ValueError(
                f"Invalid PNG checksum for {chunk_type.decode('ascii', 'replace')} in: {image_path}"
            )
        yield chunk_type, chunk_data, length
        offset = chunk_end
        if chunk_type == b"IEND":
            break


def list_png_chunks(image_path: Path | str) -> list[str]:
    """Types de chunks présents dans le PNG (audit)."""
    data = Path(image_path).read_bytes()
    return [t.decode("latin-1") for t, _, _ in iter_chunks(data, image_path)]


def has_provenance_chunks(image_path: Path | str) -> list[str]:
    """Chunks de provenance/traçabilité encore présents (vide = propre)."""
    present = {t.encode("latin-1") for t in list_png_chunks(image_path)}
    return sorted(t.decode("latin-1") for t in (present & PROVENANCE_CHUNK_TYPES))


def scrub_png_container(image_path: Path | str, drop_color_chunks: bool = False) -> list[str]:
    """
    Réécrit le PNG sur disque sans aucun chunk ancillaire. Pixels intacts.

    `drop_color_chunks` : retire aussi cHRM/gAMA/iCCP/sRGB (utile si l'on veut
    un fichier minimal ; à laisser False pour préserver la fidélité couleur).
    Retourne la liste des types de chunks retirés.
    """
    image_path = Path(image_path)
    data = image_path.read_bytes()
    keep = set(CHUNKS_TO_KEEP)
    if drop_color_chunks:
        keep -= {b"cHRM", b"gAMA", b"iCCP", b"sRGB"}

    kept: list[bytes] = []
    removed: list[str] = []
    saw = {b"IHDR": False, b"IDAT": False, b"IEND": False}

    for chunk_type, chunk_data, length in iter_chunks(data, image_path):
        if chunk_type in saw:
            saw[chunk_type] = True
        if chunk_type in keep:
            crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
            kept.append(struct.pack(">I", length) + chunk_type + chunk_data + struct.pack(">I", crc))
        else:
            removed.append(chunk_type.decode("latin-1"))

    if not all(saw.values()):
        raise ValueError(f"Incomplete PNG image: {image_path}")

    tmp = image_path.with_suffix(image_path.suffix + ".tmp")
    try:
        tmp.write_bytes(PNG_SIGNATURE + b"".join(kept))
        os.replace(tmp, image_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return removed
