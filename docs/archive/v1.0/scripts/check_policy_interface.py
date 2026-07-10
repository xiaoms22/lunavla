from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import ACTPolicyWrapper, BCMLPPolicy, MiniVLAPolicy, load_policy


def main() -> int:
    policy = ACTPolicyWrapper(input_dim=4, action_dim=2, chunk_size=3, seed=7)
    inputs = np.ones((2, 4), dtype=np.float32)
    targets = np.zeros((2, 6), dtype=np.float32)
    losses = policy.forward({"inputs": inputs, "targets": targets})
    action_chunk = policy.predict_action(inputs[0])

    if "loss" not in losses or losses["loss"] < 0:
        raise SystemExit("policy.forward must return a non-negative loss")
    if action_chunk.shape != (6,):
        raise SystemExit(f"predict_action returned {action_chunk.shape}, expected (6,)")

    tmp_dir = ROOT / "outputs" / "policy_interface_check"
    checkpoint_path = policy.save_pretrained(tmp_dir, metadata={"check": "policy_interface"})
    loaded, metadata = MiniVLAPolicy.from_pretrained(tmp_dir)
    loaded_action = loaded.predict_action(inputs[0])

    if checkpoint_path.name != "checkpoint.pt":
        raise SystemExit("save_pretrained should write checkpoint.pt inside a run directory")
    if metadata.get("check") != "policy_interface":
        raise SystemExit("from_pretrained did not restore metadata")
    if getattr(loaded, "policy_name", "unknown") != "act":
        raise SystemExit("from_pretrained did not restore policy_name")
    if not np.allclose(action_chunk, loaded_action):
        raise SystemExit("loaded policy prediction does not match saved policy")

    bc_policy = BCMLPPolicy(input_dim=4, action_dim=2, chunk_size=1, hidden_dim=8, seed=11)
    bc_losses = bc_policy.forward({"inputs": inputs, "targets": np.zeros((2, 2), dtype=np.float32)})
    bc_action = bc_policy.predict_action(inputs[0])
    if "loss" not in bc_losses or bc_losses["loss"] < 0:
        raise SystemExit("BC policy forward must return a non-negative loss")
    if bc_action.shape != (2,):
        raise SystemExit(f"BC predict_action returned {bc_action.shape}, expected (2,)")
    bc_dir = ROOT / "outputs" / "policy_interface_check_bc"
    bc_checkpoint = bc_policy.save_pretrained(bc_dir, metadata={"check": "bc_policy_interface"})
    loaded_bc, bc_metadata = load_policy(bc_checkpoint)
    if getattr(loaded_bc, "policy_name", "unknown") != "bc_mlp":
        raise SystemExit("load_policy did not dispatch to BCMLPPolicy")
    if bc_metadata.get("check") != "bc_policy_interface":
        raise SystemExit("BC from_pretrained did not restore metadata")

    print("policy interface check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
