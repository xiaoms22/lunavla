from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict[str, object]:
    path = ROOT / ".github" / "workflows" / name
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_fast_ci_maintains_main_and_v1x_without_collecting_v2_tests() -> None:
    payload = _workflow("ci.yml")
    triggers = payload["on"]
    assert set(triggers["push"]["branches"]) == {"main", "v1.x"}
    assert set(triggers["pull_request"]["branches"]) == {"main", "v1.x"}

    steps = payload["jobs"]["tests"]["steps"]
    test_step = next(step for step in steps if step.get("name") == "Run test suite with coverage gate")
    command = str(test_step["run"])
    assert "python -m pytest tests " in command
    assert "tests_v2" not in command


def test_codeql_covers_maintained_and_v2_integration_branches() -> None:
    payload = _workflow("codeql.yml")
    triggers = payload["on"]
    expected = {"main", "v1.x", "v2"}
    assert set(triggers["push"]["branches"]) == expected
    assert set(triggers["pull_request"]["branches"]) == expected
