"""Exported tree must not import monorepo packages."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SKIP_PARTS = {".git", "__pycache__"}
_SKIP_FILES = {"test_standalone_imports.py"}


class StandaloneImportTests(unittest.TestCase):
    def test_tree_has_no_monorepo_import_paths(self) -> None:
        forbidden = (
            "harness" + ".smit_harness",
            "harness" + ".common",
            "trace" + "_generation",
        )
        hits: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            if _SKIP_PARTS.intersection(path.parts) or path.name in _SKIP_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
