from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the local environment can run MiniMind-VLA.")
    parser.add_argument("--out", default="outputs/environment_check.md", help="Markdown report path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def add_row(rows: list[dict[str, str]], check: str, status: str, detail: str, next_action: str) -> None:
    rows.append(
        {
            "check": check,
            "status": status,
            "detail": detail,
            "next_action": next_action,
        }
    )


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_python(rows: list[dict[str, str]]) -> None:
    version = sys.version_info
    version_text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= MIN_PYTHON:
        add_row(rows, "python", "pass", f"Python {version_text}", "Continue to dependency checks.")
    else:
        add_row(rows, "python", "fail", f"Python {version_text}", "Use Python 3.10 or newer.")


def check_imports(rows: list[dict[str, str]]) -> None:
    dependencies = [
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
    ]
    for module_name, package_name in dependencies:
        try:
            importlib.import_module(module_name)
        except ImportError:
            add_row(
                rows,
                f"import {module_name}",
                "fail",
                f"{package_name} is not importable",
                "Run `pip install -r requirements.txt`.",
            )
        else:
            add_row(
                rows,
                f"import {module_name}",
                "pass",
                f"{package_name} {package_version(package_name)}",
                "Dependency is available.",
            )


def check_required_files(rows: list[dict[str, str]]) -> None:
    required = [
        "requirements.txt",
        "configs/act_pusht_cpu_smoke.yaml",
        "configs/act_pusht_baseline.yaml",
        "configs/act_pusht_ablation_chunk_size.yaml",
        "trainer/train_act_pusht.py",
        "eval_vla.py",
        "scripts/run_cpu_smoke.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        add_row(
            rows,
            "repo files",
            "fail",
            "Missing " + ", ".join(missing),
            "Check that the repository was cloned completely.",
        )
    else:
        add_row(rows, "repo files", "pass", f"{len(required)} required files found", "Continue to smoke testing.")


def check_output_writable(rows: list[dict[str, str]]) -> None:
    output_dir = ROOT / "outputs"
    test_dir = output_dir / ".environment_check_tmp"
    test_file = test_dir / "write_test.txt"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file.write_text("ok\n", encoding="utf-8")
        if test_file.read_text(encoding="utf-8") != "ok\n":
            raise OSError("readback mismatch")
    except OSError as exc:
        add_row(
            rows,
            "outputs writable",
            "fail",
            str(exc),
            "Check filesystem permissions for the repository directory.",
        )
    else:
        add_row(rows, "outputs writable", "pass", f"Can write under {relative(output_dir)}", "Generated run artifacts can be saved.")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def check_gitignore(rows: list[dict[str, str]]) -> None:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        add_row(rows, "artifact ignore rules", "warn", ".gitignore is missing", "Avoid committing checkpoints or outputs.")
        return
    text = gitignore.read_text(encoding="utf-8")
    needed = ["outputs/*", "*.pt", "*.ckpt", "*.safetensors"]
    missing = [pattern for pattern in needed if pattern not in text]
    if missing:
        add_row(
            rows,
            "artifact ignore rules",
            "warn",
            "Missing " + ", ".join(missing),
            "Keep large generated artifacts out of Git.",
        )
    else:
        add_row(rows, "artifact ignore rules", "pass", "Runtime outputs and checkpoints are ignored", "Safe to run local experiments.")


def status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}[status]


def overall_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "fail"
    return max((row["status"] for row in rows), key=status_rank)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def build_report(rows: list[dict[str, str]]) -> str:
    overall = overall_status(rows)
    lines: list[str] = [
        "# MiniMind-VLA Environment Check",
        "",
        f"Overall: `{overall}`",
        "",
        "This report checks whether the current machine is ready to run the public MiniMind-VLA commands.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "python scripts/run_cpu_smoke.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    check_python(rows)
    check_imports(rows)
    check_required_files(rows)
    check_output_writable(rows)
    check_gitignore(rows)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows), encoding="utf-8")

    status = overall_status(rows)
    print(f"environment check: {status}")
    print(f"environment report: {out_path}")
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
