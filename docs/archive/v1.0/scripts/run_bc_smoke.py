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
    config = "configs/bc_pusht_cpu_smoke.yaml"
    run_dir = "outputs/bc_pusht_cpu_smoke"
    checkpoint = f"{run_dir}/checkpoint.pt"

    run([python, "scripts/validate_configs.py", config])
    run([python, "trainer/train_bc_pusht.py", "--config", config])
    run([python, "eval_vla.py", "--checkpoint", checkpoint, "--episodes", "5", "--save-rollouts"])
    run([python, "scripts/summarize_results.py", "--run-dir", run_dir])
    run([python, "scripts/web_demo_vla.py", "--run-dir", run_dir])
    run([python, "scripts/generate_project_report.py", "--run-dir", run_dir, "--title", "LunaVLA BC Smoke Report"])
    run([python, "scripts/diagnose_run.py", "--run-dir", run_dir])

    expected = [
        ROOT / checkpoint,
        ROOT / run_dir / "eval_summary.json",
        ROOT / run_dir / "summary_report.md",
        ROOT / run_dir / "project_report.md",
        ROOT / run_dir / "run_diagnostic.md",
        ROOT / run_dir / "web_demo.html",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing BC smoke artifacts: " + ", ".join(missing))
    print(f"BC smoke test passed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
