from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def joined(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_PATHS = [
    ROOT / "docs" / joined("star", "_", "playbook"),
    ROOT / "docs" / joined("code", "_", "analysis"),
    ROOT / "scripts" / joined("sync", "_", "upstreams.py"),
    ROOT / "references",
]

PUBLIC_TEXT_PATHS = [
    ROOT / "README.md",
    ROOT / "DATA_CARD.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "RELEASE_NOTES.md",
    *sorted((ROOT / "docs").glob("**/*.md")),
    *sorted((ROOT / ".github").glob("**/*.yml")),
]

PUBLIC_TEXT_BLOCKLIST = [
    joined("star", "_", "playbook"),
    joined("code", "_", "analysis"),
    joined("sync", "_", "upstreams"),
    joined("future", " ", "adapter"),
    joined("Not", "Implemented"),
    joined("place", "holder"),
]

MOJIBAKE_PATTERNS = [
    chr(0xFFFD),
    chr(0x940E),
    chr(0x6CD1),
    chr(0x7F01),
    chr(0x95C2),
    chr(0x95BF),
    chr(0x9207),
    chr(0x745B),
    chr(0x951F),
]


def fail(message: str) -> None:
    raise SystemExit(f"repo quality check failed: {message}")


def check_forbidden_paths() -> None:
    for path in FORBIDDEN_PATHS:
        relative = path.relative_to(ROOT).as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.stdout.strip():
            fail(f"public repo should not track {relative}")


def check_text_blocklist() -> None:
    for path in PUBLIC_TEXT_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in PUBLIC_TEXT_BLOCKLIST:
            if pattern in text:
                fail(f"{path.relative_to(ROOT).as_posix()} contains internal marker `{pattern}`")
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                fail(f"{path.relative_to(ROOT).as_posix()} may contain mojibake `{pattern}`")


def iter_readme_paths() -> list[str]:
    text = README.read_text(encoding="utf-8")
    markdown_targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
    image_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    inline_code_paths = re.findall(
        r"`((?:docs|images|scripts|configs|dataset|model|trainer|webui)/[^`]+|eval_vla\.py|requirements\.txt)`",
        text,
    )
    return markdown_targets + image_targets + inline_code_paths


def check_readme_local_paths() -> None:
    if not README.exists():
        fail("README.md is missing")
    for raw_target in iter_readme_paths():
        target = raw_target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if " " in target and not Path(target).suffix:
            continue
        candidate = ROOT / target
        if not candidate.exists():
            fail(f"README references missing path `{target}`")


def main() -> int:
    check_forbidden_paths()
    check_text_blocklist()
    check_readme_local_paths()
    print("repo quality check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
