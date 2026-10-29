from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from antiprovenance.pipeline import sanitize_image
from antiprovenance.pixel import get_strength, psnr
from antiprovenance.png import PNG_SIGNATURE, iter_chunks, list_png_chunks, scrub_png_container


class PngContainerTests(unittest.TestCase):
    def test_iter_chunks_rejects_non_png_and_bad_crc(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not a valid PNG"):
            list(iter_chunks(b"not a png"))

        chunk_type = b"IEND"
        bad_png = PNG_SIGNATURE + struct.pack(">I", 0) + chunk_type + b"" + struct.pack(">I", 0)
        with self.assertRaisesRegex(ValueError, "checksum"):
            list(iter_chunks(bad_png))

    def test_scrub_removes_text_and_keeps_valid_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Comment", "temporary metadata")
            Image.new("RGB", (8, 8), (20, 40, 60)).save(path, pnginfo=metadata)

            self.assertIn("tEXt", list_png_chunks(path))
            removed = scrub_png_container(path)

            self.assertIn("tEXt", removed)
            self.assertNotIn("tEXt", list_png_chunks(path))
            with Image.open(path) as image:
                self.assertEqual(image.size, (8, 8))
                self.assertEqual(image.getpixel((0, 0)), (20, 40, 60))


class PixelPipelineTests(unittest.TestCase):
    def test_strength_profiles_are_independent_copies(self) -> None:
        first = get_strength("balanced")
        second = get_strength("balanced")
        first.noise_sigma = 99
        self.assertNotEqual(first.noise_sigma, second.noise_sigma)
        self.assertEqual(get_strength("does-not-exist").name, "balanced")

    def test_psnr_for_identical_and_different_arrays(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        self.assertEqual(psnr(image, image), float("inf"))
        self.assertGreater(psnr(image, image + 1), 40)

    def test_off_profile_preserves_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
            Image.fromarray(pixels, mode="RGB").save(path)

            report = sanitize_image(path, strength="off", seed=123)

            with Image.open(path) as image:
                result = np.asarray(image)
            np.testing.assert_array_equal(result, pixels)
            self.assertFalse(report["pixels_modified"])
            self.assertEqual(report["format"], "png")


if __name__ == "__main__":
    unittest.main()
