from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    python = sys.executable
    run([python, "scripts/validate_configs.py", "configs/act_pusht_cpu_smoke.yaml"])
    run([python, "trainer/train_act_pusht.py", "--config", "configs/act_pusht_cpu_smoke.yaml"])
    run([python, "eval_vla.py", "--checkpoint", "outputs/cpu_smoke/checkpoint.pt", "--episodes", "3", "--save-rollouts"])
    run([python, "scripts/summarize_results.py", "--run-dir", "outputs/cpu_smoke"])
    run([python, "scripts/web_demo_vla.py", "--run-dir", "outputs/cpu_smoke"])
    run([python, "scripts/generate_project_report.py", "--run-dir", "outputs/cpu_smoke"])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", "outputs/cpu_smoke"])

    expected = [
        ROOT / "outputs/cpu_smoke/checkpoint.pt",
        ROOT / "outputs/cpu_smoke/eval_summary.json",
        ROOT / "outputs/cpu_smoke/summary_report.md",
        ROOT / "outputs/cpu_smoke/web_demo.html",
        ROOT / "outputs/cpu_smoke/project_report.md",
        ROOT / "outputs/cpu_smoke/resume_pack.md",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing smoke artifacts: " + ", ".join(missing))
    print("CPU smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
