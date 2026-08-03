"""OOM-safe gradient diagnostics for BRACE anchor training loops."""

from __future__ import annotations

import math
from typing import Any

import torch


def flatten_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().data.norm(2).item() ** 2)
    return math.sqrt(total)


def measure_basic_grad_norms(
    params,
    *,
    scaled_raw: torch.Tensor,
    scaled_anchor: torch.Tensor,
) -> tuple[float, float]:
    """Backward-based SFT/anchor norms without autograd.grad or full-vector concat."""
    for param in params:
        if param.grad is not None:
            param.grad = None

    scaled_raw.backward(retain_graph=True)
    grad_norm_sft = flatten_grad_norm(params)

    for param in params:
        if param.grad is not None:
            param.grad = None

    scaled_anchor.backward(retain_graph=True)
    grad_norm_anchor = flatten_grad_norm(params)

    for param in params:
        if param.grad is not None:
            param.grad = None

    return grad_norm_sft, grad_norm_anchor


def measure_checkpoint_grad_metrics(
    params,
    *,
    scaled_raw: torch.Tensor,
    scaled_anchor: torch.Tensor,
    anchor_constraints: dict[str, torch.Tensor],
    grad_accum: int,
    dual_values: dict[str, float],
    grad_norm_sft: float,
) -> dict[str, float | None]:
    """Streaming dot/norm cosine and per-group constraint gradient norms at checkpoints."""
    sft_grads: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _register_sft_hooks():
        for param in params:
            if not param.requires_grad:
                continue

            def _hook(grad, pid=id(param)):
                if grad is not None:
                    sft_grads[pid] = grad.detach()

            handles.append(param.register_hook(_hook))

    for param in params:
        if param.grad is not None:
            param.grad = None

    _register_sft_hooks()
    scaled_raw.backward(retain_graph=True)
    for handle in handles:
        handle.remove()
    handles.clear()

    anchor_dot = 0.0
    anchor_norm_sq = 0.0
    anchor_handles = []

    def _make_anchor_hook(param):
        def _hook(grad):
            nonlocal anchor_dot, anchor_norm_sq
            if grad is None:
                return
            anchor_norm_sq += float(grad.detach().pow(2).sum().item())
            sft_grad = sft_grads.get(id(param))
            if sft_grad is not None:
                anchor_dot += float(torch.dot(sft_grad.reshape(-1), grad.detach().reshape(-1)).item())

        return _hook

    for param in params:
        if param.grad is not None:
            param.grad = None

    for param in params:
        if param.requires_grad:
            anchor_handles.append(param.register_hook(_make_anchor_hook(param)))

    scaled_anchor.backward(retain_graph=True)
    for handle in anchor_handles:
        handle.remove()

    grad_norm_anchor = math.sqrt(max(anchor_norm_sq, 0.0))
    grad_cosine = None
    if grad_norm_sft > 0 and grad_norm_anchor > 0:
        grad_cosine = anchor_dot / (grad_norm_sft * grad_norm_anchor)

    metrics: dict[str, float | None] = {
        "grad_cosine_sft_anchor": grad_cosine,
    }

    sft_norm_sq = sum(float(tensor.pow(2).sum().item()) for tensor in sft_grads.values())
    sft_norm = math.sqrt(max(sft_norm_sq, 0.0))

    for group, constraint in anchor_constraints.items():
        for param in params:
            if param.grad is not None:
                param.grad = None

        group_dot = 0.0
        group_norm_sq = 0.0
        group_handles = []

        def _make_group_hook(param):
            def _hook(grad):
                nonlocal group_dot, group_norm_sq
                if grad is None:
                    return
                group_norm_sq += float(grad.detach().pow(2).sum().item())
                sft_grad = sft_grads.get(id(param))
                if sft_grad is not None:
                    group_dot += float(torch.dot(sft_grad.reshape(-1), grad.detach().reshape(-1)).item())

            return _hook

        for param in params:
            if param.requires_grad:
                group_handles.append(param.register_hook(_make_group_hook(param)))

        (constraint / grad_accum).backward(retain_graph=True)
        for handle in group_handles:
            handle.remove()

        group_norm = math.sqrt(max(group_norm_sq, 0.0))
        metrics[f"grad_norm_constraint/{group}"] = group_norm
        if sft_norm > 0 and group_norm > 0:
            metrics[f"grad_cosine_sft_{group}"] = group_dot / (sft_norm * group_norm)
        else:
            metrics[f"grad_cosine_sft_{group}"] = None
        dual_value = float(dual_values.get(group, 0.0))
        metrics[f"lambda_grad_scale/{group}"] = dual_value * group_norm / max(sft_norm, 1e-12)

    for param in params:
        if param.grad is not None:
            param.grad = None

    return metrics


def estimate_warmstart_lambdas(
    params,
    *,
    scaled_raw: torch.Tensor,
    anchor_constraints: dict[str, torch.Tensor],
    grad_accum: int,
    target_ratio: float = 0.3,
) -> dict[str, float]:
    """Estimate per-group lambda so ||lambda * grad_c|| / ||grad_sft|| ~= target_ratio."""
    for param in params:
        if param.grad is not None:
            param.grad = None
    scaled_raw.backward(retain_graph=True)
    sft_norm = flatten_grad_norm(params)
    if sft_norm <= 0:
        return {group: 0.0 for group in anchor_constraints}

    lambdas: dict[str, float] = {}
    for group, constraint in anchor_constraints.items():
        for param in params:
            if param.grad is not None:
                param.grad = None
        (constraint / grad_accum).backward(retain_graph=True)
        group_norm = flatten_grad_norm(params)
        if group_norm <= 0:
            lambdas[group] = 0.0
        else:
            lambdas[group] = target_ratio * sft_norm / group_norm

    for param in params:
        if param.grad is not None:
            param.grad = None
    return lambdas
