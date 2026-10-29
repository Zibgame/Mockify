from __future__ import annotations

import base64
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import main as mockify


class FileUtilityTests(unittest.TestCase):
    def test_validate_and_load_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("  hello  ", encoding="utf-8")

            mockify.validate_file(path, "Prompt")
            self.assertEqual(mockify.load_text_file(path, "Prompt"), "hello")
            self.assertEqual(mockify.load_text_file(path.with_name("missing.txt"), "Optional", False), "")

    def test_validate_file_rejects_missing_empty_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                mockify.validate_file(root / "missing", "Input")
            with self.assertRaises(ValueError):
                mockify.validate_file(root, "Input")

            empty = root / "empty.txt"
            empty.touch()
            with self.assertRaises(ValueError):
                mockify.validate_file(empty, "Input")

    def test_get_file_size_formats_kb_and_mb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"x" * 2048)
            self.assertEqual(mockify.get_file_size(path), "2.00 KB")
            path.write_bytes(b"x" * (2 * 1024 * 1024))
            self.assertEqual(mockify.get_file_size(path), "2.00 MB")


class OutputManagementTests(unittest.TestCase):
    def test_safe_stem_and_automatic_worker_count(self) -> None:
        self.assertEqual(mockify.safe_stem("  Mon design (bleu)!  "), "Mon_design__bleu")
        self.assertEqual(mockify.safe_stem("***"), "image")
        self.assertEqual(mockify.resolve_worker_count(4), 4)
        self.assertEqual(mockify.resolve_worker_count(12), 8)
        self.assertEqual(mockify.resolve_worker_count(4, 2), 2)
        self.assertEqual(mockify.resolve_worker_count(0), 0)

    def test_create_run_dir_avoids_name_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            with patch.object(mockify, "OUTPUTS_DIR", outputs):
                first = mockify.create_run_dir("Design rose!", "20261029_120000")
                second = mockify.create_run_dir("Design rose!", "20261029_120000")

            self.assertEqual(first.name, "Design_rose_20261029_120000")
            self.assertEqual(second.name, "Design_rose_20261029_120000_2")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_archive_is_verified_appended_and_removes_only_run_folders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            archive_path = outputs / "mockify_outputs.zip"
            first_run = outputs / "first_run"
            first_run.mkdir(parents=True)
            (first_run / "image.png").write_bytes(b"first")

            with (
                patch.object(mockify, "OUTPUTS_DIR", outputs),
                patch.object(mockify, "OUTPUT_ARCHIVE", archive_path),
            ):
                archive, folders, files = mockify.archive_output_folders()
                self.assertEqual((archive, folders, files), (archive_path, 1, 1))
                self.assertFalse(first_run.exists())

                second_run = outputs / "second_run"
                second_run.mkdir()
                (second_run / "choice.png").write_bytes(b"second")
                archive, folders, files = mockify.archive_output_folders()

            self.assertEqual((archive, folders, files), (archive_path, 1, 1))
            self.assertFalse(second_run.exists())
            with zipfile.ZipFile(archive_path, "r") as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertEqual(
                    set(zipped.namelist()),
                    {"first_run/image.png", "second_run/choice.png"},
                )


class PromptTests(unittest.TestCase):
    def test_single_prompt_assigns_roles_and_adds_mockup_rules(self) -> None:
        prompt = mockify.build_prompt("BASE", "double.png")
        self.assertIn("Image 1 -> MOCKUP BASE", prompt)
        self.assertIn("Image 2 -> TEXTURE DESIGN", prompt)
        self.assertIn("TWO Kindle cases", prompt)
        self.assertIn("BASE", prompt)

    def test_unknown_mockup_gets_common_prompt_only(self) -> None:
        prompt = mockify.build_prompt("BASE", "unknown.png")
        self.assertIn("BASE", prompt)
        self.assertNotIn("IMPORTANT —", prompt)

    def test_board_prompt_reflects_optional_inputs(self) -> None:
        prompt = mockify.build_board_generation_prompt("BASE", "Pastel flowers", True)
        self.assertIn("exactly 9 distinct designs", prompt)
        self.assertIn("Pastel flowers", prompt)
        self.assertIn("REFERENCE IMAGES PROVIDED", prompt)


class CostAndResultTests(unittest.TestCase):
    def test_compute_and_format_cost(self) -> None:
        expected = mockify._PRICE_INPUT_PER_M + mockify._PRICE_OUTPUT_PER_M
        self.assertEqual(mockify.compute_cost(1_000_000, 1_000_000), expected)
        self.assertEqual(mockify.format_cost(1.23456), "$1.2346")

    def test_usage_to_result_uses_real_usage(self) -> None:
        usage = SimpleNamespace(input_tokens=100, output_tokens=200)
        result = mockify.usage_to_result(usage, 1.5, Path("result.png"))
        self.assertEqual(result["cost_source"], "real")
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 200)
        self.assertEqual(result["elapsed"], 1.5)

    def test_usage_to_result_falls_back_to_estimate(self) -> None:
        result = mockify.usage_to_result(None, 2.0, Path("result.png"))
        self.assertEqual(result["cost_source"], "estimated")
        self.assertEqual(result["cost_usd"], mockify.estimated_cost())


class ImageUtilityTests(unittest.TestCase):
    def test_image_base64_round_trip(self) -> None:
        payload = b"mock-image-data"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            target = Path(directory) / "target.bin"
            source.write_bytes(payload)

            encoded = mockify.image_to_base64(source)
            self.assertEqual(base64.b64decode(encoded), payload)
            self.assertIsNone(mockify.save_image(encoded, target, finalize=False))
            self.assertEqual(target.read_bytes(), payload)

    def test_parse_image_size(self) -> None:
        self.assertEqual(mockify._parse_image_size("1024x768"), (1024, 768))
        self.assertEqual(mockify._parse_image_size("10X20"), (10, 20))
        with self.assertRaises((ValueError, AttributeError)):
            mockify._parse_image_size("invalid")

    def test_normalize_mockup_and_mask_resizes_and_binarizes_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mockup = root / "mockup.png"
            mask = root / "mask.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(mockup)

            mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
            mask_data[..., :3] = 255
            mask_data[..., 3] = np.array(
                [[0, 64, 128, 255], [0, 64, 128, 255], [0, 64, 128, 255], [0, 64, 128, 255]],
                dtype=np.uint8,
            )
            Image.fromarray(mask_data, mode="RGBA").save(mask)

            resized_mockup, resized_mask = mockify.normalize_mockup_and_mask(mockup, mask, "8x6")
            generated_dir = resized_mockup.parent
            try:
                with Image.open(resized_mockup) as image:
                    self.assertEqual(image.size, (8, 6))
                with Image.open(resized_mask) as image:
                    self.assertEqual(image.size, (8, 6))
                    alpha = np.asarray(image.getchannel("A"))
                    self.assertTrue(set(np.unique(alpha)).issubset({0, 255}))
            finally:
                shutil.rmtree(generated_dir)

    def test_normalize_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mockup = root / "mockup.png"
            mask = root / "mask.png"
            Image.new("RGB", (4, 4)).save(mockup)
            Image.new("RGBA", (5, 4)).save(mask)
            with self.assertRaisesRegex(ValueError, "differ"):
                mockify.normalize_mockup_and_mask(mockup, mask, "8x8")


if __name__ == "__main__":
    unittest.main()
