"""Every source file carries the Apache 2.0 header."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_file_has_the_license_header():
    missing = [str(p.relative_to(ROOT)) for p in (ROOT / "papilon").rglob("*.py")
               if "Licensed under the Apache License, Version 2.0" not in p.read_text()[:1200]]
    assert not missing, f"add the license header to: {missing}"


def test_license_and_notice_are_in_place():
    assert "Apache License" in (ROOT / "LICENSE").read_text()[:200]
    notice = (ROOT / "NOTICE").read_text()
    assert "Brian Curry" in notice and "trademarks" in notice
