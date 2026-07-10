from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

FORBIDDEN_TRACKED_PATHS = (
    "docs/star_playbook",
    "docs/code_analysis",
    "scripts/sync_upstreams.py",
    "references",
)
SKIP_PREFIXES = (
    "docs/archive/v1.0/",
    "outputs/",
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "references/",
)
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_SCAN_BYTES = 2 * 1024 * 1024

ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/(?:[^\s`'\"<>]|\\ )+"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\[^\s`'\"<>]+"),
)
CREDENTIAL_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("GitHub classic token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,255}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
)


def fail(errors: list[str]) -> None:
    if not errors:
        return
    for error in errors:
        print(f"repo quality error: {error}", file=sys.stderr)
    raise SystemExit(1)


def repository_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))


def is_skipped(relative: str) -> bool:
    return relative.startswith(SKIP_PREFIXES)


def tracked(relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_forbidden_paths(files: list[str]) -> list[str]:
    errors: list[str] = []
    for forbidden in FORBIDDEN_TRACKED_PATHS:
        if any(path == forbidden or path.startswith(forbidden + "/") for path in files if tracked(path)):
            errors.append(f"public repository must not track `{forbidden}`")
    return errors


def check_text(files: list[str]) -> list[str]:
    errors: list[str] = []
    for relative in files:
        if is_skipped(relative):
            continue
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"`{relative}` is not valid UTF-8")
            continue
        if "\ufffd" in text:
            errors.append(f"`{relative}` contains the Unicode replacement character")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            match = pattern.search(text)
            if match:
                errors.append(f"`{relative}` contains a workstation-specific absolute path: `{match.group(0)}`")
                break
        for label, pattern in CREDENTIAL_PATTERNS:
            if pattern.search(text):
                errors.append(f"`{relative}` contains a value shaped like a {label}")
    return errors


def readme_targets() -> list[str]:
    if not README.is_file():
        return []
    text = README.read_text(encoding="utf-8")
    return re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)


def check_readme_links() -> list[str]:
    if not README.is_file():
        return ["README.md is missing"]
    errors: list[str] = []
    for raw in readme_targets():
        target = raw.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (ROOT / target).exists():
            errors.append(f"README references missing local path `{target}`")
    return errors


def main() -> int:
    files = repository_files()
    errors = check_forbidden_paths(files)
    errors.extend(check_text(files))
    errors.extend(check_readme_links())
    fail(errors)
    print(f"repo quality check passed ({len(files)} repository files inspected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
