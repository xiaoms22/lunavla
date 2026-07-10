from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release-assets"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ALPHA_CONFIGS = (
    Path("configs/v2/numpy_baseline.yaml"),
    Path("configs/v2/transformer_chunk_cpu.yaml"),
    Path("configs/v2/transformer_visual_cpu.yaml"),
    Path("configs/v2/transformer_visual_state_only_cpu.yaml"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean LunaVLA v2 release evidence profile.")
    parser.add_argument(
        "--profile",
        required=True,
        choices=("alpha", "language", "vision", "stable"),
    )
    parser.add_argument("--expected-sha", required=True)
    return parser.parse_args(argv)


def run(command: Sequence[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_origin(remote: str) -> str:
    """Normalize Git's equivalent HTTPS/SSH spelling without broadening ownership."""

    value = remote.strip()
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def verify_source(expected_sha: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha):
        raise ValueError("--expected-sha must be a 40-character lowercase Git SHA")
    actual = run(("git", "rev-parse", "HEAD"), capture=True)
    if actual != expected_sha:
        raise RuntimeError(f"checked-out Git SHA {actual} does not match {expected_sha}")
    status = run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        capture=True,
    )
    if status:
        raise RuntimeError("release evidence requires a clean Git checkout")
    remote = canonical_origin(
        run(("git", "remote", "get-url", "origin"), capture=True)
    )
    if remote not in {
        "https://github.com/xiaoms22/lunavla",
        "git@github.com:xiaoms22/lunavla",
    }:
        raise RuntimeError(f"refusing release evidence from unexpected origin: {remote}")


def prepare_release_root() -> None:
    if RELEASE_ROOT.exists() or (ROOT / "outputs" / "v2").exists():
        raise FileExistsError("release-assets/ and outputs/v2/ must not exist before a release run")
    RELEASE_ROOT.mkdir(parents=True)


def alpha_quality_gate() -> None:
    commands: tuple[tuple[str, ...], ...] = (
        ("uv", "lock", "--check"),
        (sys.executable, "scripts/lock_v2_cpu.py", "--check"),
        (sys.executable, "scripts/validate_configs.py"),
        (sys.executable, "scripts/check_repo_quality.py"),
        (sys.executable, "scripts/render_readme_results.py", "--check"),
        ("ruff", "check", "."),
        ("mypy", "dataset", "model", "trainer", "lunavla", "eval_vla.py"),
        (sys.executable, "-m", "pytest", "-q"),
    )
    for command in commands:
        run(command)


def alpha_runs(expected_sha: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in ALPHA_CONFIGS:
        run(
            (
                sys.executable,
                "-m",
                "lunavla.cli",
                "train",
                config_path.as_posix(),
                "--require-device",
                "cpu",
                "--require-clean",
            )
        )
        config = json.loads(
            run(
                (
                    sys.executable,
                    "-c",
                    (
                        "import json,sys,yaml; "
                        "print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
                    ),
                    config_path.as_posix(),
                ),
                capture=True,
            )
        )
        output_dir = ROOT / str(config["artifacts"]["output_dir"])
        run((sys.executable, "-m", "lunavla.cli", "verify-run", str(output_dir)))
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["git_sha"] != expected_sha:
            raise RuntimeError(f"manifest source mismatch: {manifest_path}")
        if manifest["git_dirty"] or manifest["source_diff_sha256"] is not None:
            raise RuntimeError(f"dirty manifest is not release evidence: {manifest_path}")
        rows.append(
            {
                "config": config_path.as_posix(),
                "output_dir": output_dir.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "policy_id": manifest["policy_id"],
                "task_id": manifest["task_id"],
                "claim_allowed": False,
            }
        )
    return rows


def build_distributions() -> list[Path]:
    destination = RELEASE_ROOT / "dist"
    run((sys.executable, "-m", "build", "--outdir", str(destination)))
    distributions = sorted(path for path in destination.iterdir() if path.is_file())
    if not distributions:
        raise RuntimeError("package build produced no distributions")
    run(("twine", "check", *(str(path) for path in distributions)))
    return distributions


def installed_requirements() -> str:
    """Return a deterministic freeze without assuming pip exists in the uv venv."""

    versions: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        value = (name, distribution.version)
        previous = versions.get(canonical)
        if previous is not None and previous[1] != value[1]:
            raise RuntimeError(
                f"conflicting installed versions for {canonical}: "
                f"{previous[1]} and {value[1]}"
            )
        versions[canonical] = value
    if not versions:
        raise RuntimeError("the release environment contains no distributions")
    return "".join(
        f"{versions[key][0]}=={versions[key][1]}\n" for key in sorted(versions)
    )


def sbom_command(requirements: Path, output: Path) -> tuple[str, ...]:
    return (
        "cyclonedx-py",
        "requirements",
        str(requirements),
        "--output-reproducible",
        "--output-file",
        str(output),
    )


def write_environment_and_sbom() -> None:
    requirements = RELEASE_ROOT / "environment-requirements.txt"
    requirements.write_text(installed_requirements(), encoding="utf-8")
    run(sbom_command(requirements, RELEASE_ROOT / "sbom.json"))


def write_candidate(
    *,
    profile: str,
    expected_sha: str,
    runs: Iterable[dict[str, object]],
    distributions: Sequence[Path],
) -> Path:
    from importlib.metadata import version

    path = RELEASE_ROOT / "release-candidate.json"
    payload = {
        "schema_version": 1,
        "profile": profile,
        "expected_sha": expected_sha,
        "package_version": version("lunavla"),
        "authoritative_device": "cpu",
        "modality_effect_claims": False,
        "runs": list(runs),
        "distributions": [
            {
                "path": item.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(item),
            }
            for item in distributions
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_evidence_archive() -> Path:
    archive = RELEASE_ROOT / "lunavla-v2-alpha-evidence.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        stream.add(ROOT / "outputs" / "v2", arcname="outputs/v2")
        for name in ("release-candidate.json", "environment-requirements.txt", "sbom.json"):
            stream.add(RELEASE_ROOT / name, arcname=f"release-assets/{name}")
        stream.add(RELEASE_ROOT / "dist", arcname="release-assets/dist")
    return archive


def write_checksums() -> Path:
    candidates = sorted(
        path
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    target = RELEASE_ROOT / "SHA256SUMS"
    target.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}\n"
            for path in candidates
        ),
        encoding="utf-8",
    )
    return target


def run_alpha(expected_sha: str) -> None:
    prepare_release_root()
    try:
        alpha_quality_gate()
        runs = alpha_runs(expected_sha)
        distributions = build_distributions()
        write_environment_and_sbom()
        write_candidate(
            profile="alpha",
            expected_sha=expected_sha,
            runs=runs,
            distributions=distributions,
        )
        write_evidence_archive()
        write_checksums()
    except Exception:
        shutil.rmtree(RELEASE_ROOT, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_sha = str(args.expected_sha)
    verify_source(expected_sha)
    if args.profile != "alpha":
        raise RuntimeError(
            f"profile {args.profile!r} is gated until its EvidenceDesign implementation lands"
        )
    run_alpha(expected_sha)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "expected_sha": expected_sha,
                "release_assets": str(RELEASE_ROOT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
