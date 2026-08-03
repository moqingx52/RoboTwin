"""Training-path faithful anchor feasibility diagnostic with full trajectory logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.brace.anchor_diagnostic_loop import DiagnosticJobConfig, run_anchor_diagnostic


def run_feasibility_diagnostic(
    protocol: dict[str, Any],
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path,
) -> dict[str, Any]:
    gate = protocol.get("constraint_feasibility", {})
    job = DiagnosticJobConfig(
        job_id="feasibility",
        anchor_enabled=True,
        diagnostic_steps=int(gate.get("diagnostic_steps", 200)),
        scheduler_total_steps=int(gate.get("scheduler_total_steps", gate.get("diagnostic_steps", 200))),
        checkpoint_steps=[int(step) for step in gate.get("checkpoint_steps", [])],
        checkpoint_epochs={str(k): int(v) for k, v in dict(gate.get("checkpoint_epochs", {})).items()},
        dual_lr=float(protocol.get("anchor_smoke", {}).get("dual_lr", 0.01)),
        grad_diag_at_checkpoints_only=bool(gate.get("grad_diag_at_checkpoints_only", True)),
        record_peak_memory=bool(gate.get("record_peak_memory", True)),
    )
    return run_anchor_diagnostic(
        protocol,
        task=task,
        run_label=run_label,
        dataset=dataset,
        traced_root=traced_root,
        base_checkpoint=base_checkpoint,
        work_dir=work_dir,
        job=job,
        stage="anchor_feasibility",
    )
