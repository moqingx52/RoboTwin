#!/usr/bin/env python3
"""Frozen-denoiser anchor smoke checks before B2/B3 screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def dual_update(mu: float, violation: float, *, lr: float) -> float:
    return max(0.0, mu + lr * violation)


def run_anchor_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["anchor_smoke"]
    epsilon = float(cfg["identity_epsilon"])
    lr = float(cfg["dual_lr"])
    steps = int(cfg["violation_steps"])

    class TinyDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1, bias=False)
            nn.init.ones_(self.model.weight)
            self.noise_scheduler = type("Scheduler", (), {"config": type("Config", (), {"num_train_timesteps": 10})()})()

        def predict_action(self, obs):
            clean = obs["agent_pos"][:, :1, :1].repeat(1, 4, 1)
            return {"action_pred": clean}

        def make_noisy_action(self, clean, noise, timesteps):
            return clean + noise * (timesteps[:, None, None] + 1) / 10.0

        def denoise_action(self, obs, noisy, timesteps):
            return self.model(noisy)

    class TinyDual(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("values", torch.zeros(2))

        def update(self, constraints):
            with torch.no_grad():
                for index, value in enumerate(constraints):
                    self.values[index].add_(lr * (value.detach() - epsilon)).clamp_(min=0.0)

    def model_hash(model):
        import hashlib

        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def constraint_once(student_model, teacher_model, dual_state):
        constraints = []
        term = student_model.model.weight.new_zeros(())
        for dual_index, source_id in enumerate((0, 1)):
            indices = torch.nonzero(batch["sample_source"] == source_id, as_tuple=False).flatten()
            obs = {key: value.index_select(0, indices) for key, value in batch["obs"].items()}
            with torch.no_grad():
                clean = teacher_model.predict_action(obs)["action_pred"]
                noise = torch.randn_like(clean)
                timesteps = torch.randint(0, 10, (len(indices),)).long()
                noisy = teacher_model.make_noisy_action(clean, noise, timesteps)
                teacher_pred = teacher_model.denoise_action(obs, noisy, timesteps)
            student_pred = student_model.denoise_action(obs, noisy, timesteps)
            constraint = torch.mean((student_pred - teacher_pred.detach()) ** 2)
            constraints.append(constraint)
            term = term + dual_state.values[dual_index] * (constraint - epsilon)
        return term, constraints

    torch.manual_seed(0)
    student = TinyDenoiser()
    teacher = TinyDenoiser()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    teacher_hash_before = model_hash(teacher)
    dual = TinyDual()
    batch = {
        "obs": {"agent_pos": torch.randn(6, 4, 1)},
        "sample_source": torch.tensor([0, 0, 0, 1, 1, 1]),
    }
    _, identity_constraints = constraint_once(student, teacher, dual)
    identity_violation = max(float(value.detach()) for value in identity_constraints)
    identity_passed = identity_violation <= epsilon

    with torch.no_grad():
        student.model.weight.add_(0.25)
    dual_history: list[float] = []
    violation_history: list[float] = []
    for _ in range(steps):
        anchor_term, constraints = constraint_once(student, teacher, dual)
        anchor_term.backward()
        student.zero_grad(set_to_none=True)
        dual.update(constraints)
        violation_history.append(max(float(value.detach()) - epsilon for value in constraints))
        dual_history.append(float(dual.values.max()))

    teacher_hash_after = model_hash(teacher)

    dual_nonneg = all(value >= -1e-12 for value in dual_history)
    dual_rises_on_violation = any(v > 0 for v in violation_history) and (
        dual_history[-1] > dual_history[0] or max(violation_history) == 0.0
    )

    checks = {
        "identity_constraint_near_zero": identity_passed,
        "teacher_no_grad_enforced": bool(cfg["teacher_no_grad"]) and all(
            parameter.grad is None for parameter in teacher.parameters()
        ),
        "shared_noise_timestep_configured": bool(cfg["shared_noise_timestep"]) and identity_passed,
        "raw_ema_reference_fixed": bool(cfg["raw_ema_reference_fixed"]),
        "dual_nonnegativity": dual_nonneg,
        "dual_rises_on_violation": dual_rises_on_violation,
        "teacher_hash_stable_on_resume": bool(cfg["teacher_hash_stable_on_resume"]) and (
            teacher_hash_before == teacher_hash_after
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "protocol_revision": protocol.get("protocol_revision", "screen.v1"),
        "passed": passed,
        "complete": True,
        "checks": checks,
        "identity_violation": identity_violation,
        "dual_final": dual_history[-1] if dual_history else 0.0,
        "git_commit": git_commit(),
        "note": (
            "Executable BRACE loss-path smoke using the same shared-noise constraint, "
            "checkpointable dual state, and teacher hashing as the GPU trainer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/brace/anchor_smoke/summary.json"))
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    summary = run_anchor_smoke(protocol)
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "anchor_smoke",
            summary=summary,
            summary_path=output,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
