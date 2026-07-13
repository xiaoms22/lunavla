from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from lunavla.v3 import (
    PyPIFileRecordV1,
    PyPIPublishRecordV1,
    StablePrePublishCandidateV1,
    StableReleaseCandidateV1,
    TrustedPublisherIdentityV1,
    verify_release_candidate_assets,
)

PROJECT = "lunavla"
VERSION = "3.0.0"
TAG = "v3.0.0"
PUBLISHER = TrustedPublisherIdentityV1(
    owner="xiaoms22",
    repository="lunavla",
    workflow="v3-pypi-release.yml",
    environment="pypi",
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return value


def _get_json(url: str, *, allow_not_found: bool = False) -> Mapping[str, Any] | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if allow_not_found and error.code == 404:
            return None
        raise
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON endpoint did not return a mapping: {url}")
    return value


def verify_version_absent(api_root: str) -> None:
    payload = _get_json(f"{api_root.rstrip('/')}/pypi/{PROJECT}/json", allow_not_found=True)
    if payload is None:
        return
    releases = payload.get("releases")
    if not isinstance(releases, Mapping):
        raise ValueError("PyPI project JSON is missing releases")
    if VERSION in releases and releases[VERSION]:
        raise ValueError("PyPI already contains lunavla 3.0.0; overwrite is forbidden")


def verify_prepublish(candidate_path: str | Path, asset_root: str | Path, expected_sha: str) -> None:
    candidate = cast(
        StablePrePublishCandidateV1,
        StablePrePublishCandidateV1.from_mapping(_mapping(candidate_path)),
    )
    if candidate.git_sha != expected_sha or candidate.expected_tag != TAG:
        raise ValueError("pre-publish candidate does not bind the requested stable tag/SHA")
    verify_release_candidate_assets(candidate, asset_root)


def _publisher_from_provenance(value: Mapping[str, Any]) -> TrustedPublisherIdentityV1:
    bundles = value.get("attestation_bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("PyPI provenance has no attestation bundles")
    matches: list[TrustedPublisherIdentityV1] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            continue
        publisher = bundle.get("publisher")
        attestations = bundle.get("attestations")
        if not isinstance(publisher, Mapping) or not isinstance(attestations, list) or not attestations:
            continue
        if publisher.get("kind") != "GitHub":
            continue
        repository = publisher.get("repository")
        if repository != "xiaoms22/lunavla":
            continue
        matches.append(
            TrustedPublisherIdentityV1(
                owner="xiaoms22",
                repository="lunavla",
                workflow=str(publisher.get("workflow", "")),
                environment=str(publisher.get("environment", "")),
            )
        )
    if matches != [PUBLISHER]:
        raise ValueError("PyPI provenance does not contain the exact trusted publisher identity")
    return matches[0]


def build_publication(
    candidate_path: str | Path,
    asset_root: str | Path,
    output_dir: str | Path,
    *,
    api_root: str,
    integrity_root: str,
) -> Path:
    pre = cast(
        StablePrePublishCandidateV1,
        StablePrePublishCandidateV1.from_mapping(_mapping(candidate_path)),
    )
    root = Path(asset_root).resolve()
    verify_release_candidate_assets(pre, root)
    project = _get_json(f"{api_root.rstrip('/')}/pypi/{PROJECT}/{VERSION}/json")
    if project is None:
        raise ValueError("PyPI publication is missing")
    urls = project.get("urls")
    if not isinstance(urls, list):
        raise ValueError("PyPI release JSON is missing files")
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "dist").iterdir())
        if path.is_file()
    }
    if set(expected) != {"lunavla-3.0.0-py3-none-any.whl", "lunavla-3.0.0.tar.gz"}:
        raise ValueError("signed payload does not contain the exact stable distributions")
    files: list[PyPIFileRecordV1] = []
    provenances: dict[str, Mapping[str, Any]] = {}
    for item in urls:
        if not isinstance(item, Mapping) or item.get("filename") not in expected:
            continue
        filename = str(item["filename"])
        digests = item.get("digests")
        if not isinstance(digests, Mapping) or digests.get("sha256") != expected[filename]:
            raise ValueError(f"PyPI digest mismatch for {filename}")
        url = str(item.get("url", ""))
        provenance = _get_json(
            f"{integrity_root.rstrip('/')}/integrity/{PROJECT}/{VERSION}/{urllib.parse.quote(filename)}/provenance"
        )
        if provenance is None:
            raise ValueError(f"PyPI provenance is missing for {filename}")
        _publisher_from_provenance(provenance)
        provenance_bytes = _canonical(provenance)
        provenances[filename] = provenance
        files.append(
            PyPIFileRecordV1(
                filename=filename,
                sha256=expected[filename],
                size_bytes=int(item.get("size", 0)),
                file_url=url,
                provenance_sha256=_sha256(provenance_bytes),
            )
        )
    if len(files) != 2:
        raise ValueError("PyPI release does not contain the exact wheel and sdist")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    provenance_payload = {name: provenances[name] for name in sorted(provenances)}
    provenance_bytes = _canonical(provenance_payload)
    (out / "pypi-provenance.json").write_bytes(provenance_bytes)
    receipt_core = {
        "schema_version": 1,
        "project": PROJECT,
        "version": VERSION,
        "project_url": f"https://pypi.org/project/{PROJECT}/{VERSION}/",
        "publisher": PUBLISHER.to_dict(),
        "files": [item.to_dict() for item in sorted(files, key=lambda value: value.filename)],
    }
    receipt_bytes = _canonical(receipt_core)
    (out / "pypi-publication.json").write_bytes(receipt_bytes)
    record = PyPIPublishRecordV1(
        project=PROJECT,
        version=VERSION,
        files=tuple(files),
        publisher=PUBLISHER,
        publish_receipt_sha256=_sha256(receipt_bytes),
        attestation_sha256=_sha256(provenance_bytes),
        project_url=f"https://pypi.org/project/{PROJECT}/{VERSION}/",
        published=True,
    )
    payload = pre.to_dict()
    payload["pypi_published"] = True
    payload["pypi_publish_record"] = record.to_dict()
    final = StableReleaseCandidateV1.from_mapping(payload)
    destination = out / "stable-release-candidate.json"
    destination.write_bytes(_canonical(final.to_dict()))
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_v3_pypi_release")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("candidate")
    preflight.add_argument("--asset-root", required=True)
    preflight.add_argument("--expected-sha", required=True)
    preflight.add_argument("--api-root", default="https://pypi.org")
    verify = commands.add_parser("verify-publication")
    verify.add_argument("candidate")
    verify.add_argument("--asset-root", required=True)
    verify.add_argument("--out", required=True)
    verify.add_argument("--api-root", default="https://pypi.org")
    verify.add_argument("--integrity-root", default="https://pypi.org")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        verify_prepublish(args.candidate, args.asset_root, args.expected_sha)
        verify_version_absent(args.api_root)
        print(json.dumps({"valid": True, "project": PROJECT, "version": VERSION}, sort_keys=True))
        return 0
    destination = build_publication(
        args.candidate,
        args.asset_root,
        args.out,
        api_root=args.api_root,
        integrity_root=args.integrity_root,
    )
    print(json.dumps({"valid": True, "candidate": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
