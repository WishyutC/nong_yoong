from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_sidecar_utils.py"
SPEC = importlib.util.spec_from_file_location("qwen_sidecar_utils", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenSidecarTextTests(unittest.TestCase):
    def test_clean_text_removes_markdown_and_emoji_but_keeps_lines(self) -> None:
        cleaned = MODULE.clean_text("**เฟส:** ไฟฟ้า 😊\n2. **แรงดัน:** ทดสอบ")

        self.assertEqual(cleaned, "เฟส: ไฟฟ้า\n2. แรงดัน: ทดสอบ")

    def test_long_formatted_reply_is_split_without_losing_ending(self) -> None:
        text = (
            "แน่นอนค่ะ มาดูคำศัพท์ไฟฟ้ากันนะคะ\n"
            "1. เฟส คือสายไฟที่ส่งพลังงานสลับกัน ทำให้กระแสไหลต่อเนื่อง\n"
            "2. แรงดัน คือแรงที่ผลักดันให้กระแสไฟฟ้าไหล\n"
            "3. กระแส คือปริมาณไฟฟ้าที่ไหลผ่านวงจร\n"
            "ง่ายๆ แค่นี้ค่ะ มีคำศัพท์อื่นที่อยากรู้เพิ่มอีกไหมคะ"
        )

        chunks = MODULE.split_text_chunks(text, max_chars=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertTrue(chunks[-1].endswith("ไหมคะ"))
        self.assertIn("แรงดัน", " ".join(chunks))
        self.assertFalse(any(chunk.endswith(("1.", "2.", "3.")) for chunk in chunks))

    def test_unspaced_thai_text_obeys_hard_limit(self) -> None:
        chunks = MODULE.split_text_chunks("ก" * 181, max_chars=60)

        self.assertEqual([len(chunk) for chunk in chunks], [60, 60, 60, 1])

    def test_end_padding_uses_exact_silence_sample_count(self) -> None:
        self.assertEqual(MODULE.silence_sample_count(24_000, 500), 12_000)

    def test_invalid_silence_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.silence_sample_count(0, 500)
        with self.assertRaises(ValueError):
            MODULE.silence_sample_count(24_000, -1)


if __name__ == "__main__":
    unittest.main()
