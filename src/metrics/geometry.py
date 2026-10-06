"""
Worker-diversity geometry diagnostics (Experiment E3).

Metrics computed at each outer synchronization:
  - Pairwise cosine similarity of pseudo-gradients across workers
  - Norm dispersion (std of pseudo-gradient L2 norms)
  - Pairwise model-parameter L2 distance between workers
  - Angle between averaged pseudo-gradient and a synchronous reference gradient
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor


def _flatten(param_dict: Dict[str, Tensor]) -> Tensor:
    """Flatten a {name: tensor} dict into a single 1-D vector."""
    return torch.cat([p.detach().float().view(-1) for p in param_dict.values()])


def pseudo_gradient_cosine_similarity(
    pseudo_grads: List[Dict[str, Tensor]],
) -> float:
    """
    Mean pairwise cosine similarity of pseudo-gradients across M workers.
    Returns a float in [-1, 1]; higher = more agreement.
    """
    M = len(pseudo_grads)
    if M < 2:
        return 1.0

    vecs = [_flatten(pg) for pg in pseudo_grads]
    total, count = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            cos = torch.nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
            ).item()
            total += cos
            count += 1
    return total / count if count else 0.0


def pseudo_gradient_norm_dispersion(
    pseudo_grads: List[Dict[str, Tensor]],
) -> dict:
    """
    Compute mean and std of pseudo-gradient L2 norms across workers.
    """
    norms = [_flatten(pg).norm(2).item() for pg in pseudo_grads]
    mean = sum(norms) / len(norms)
    variance = sum((n - mean) ** 2 for n in norms) / len(norms)
    return {
        "pg_norm_mean": mean,
        "pg_norm_std":  math.sqrt(variance),
        "pg_norms":     norms,
    }


def worker_parameter_distance(
    worker_params: List[Dict[str, Tensor]],
) -> dict:
    """
    Mean pairwise L2 distance between worker model parameters.
    Measures how much workers have drifted from each other after H steps.
    """
    M = len(worker_params)
    if M < 2:
        return {"worker_param_dist_mean": 0.0, "worker_param_dist_max": 0.0}

    vecs = [_flatten(wp) for wp in worker_params]
    dists = []
    for i in range(M):
        for j in range(i + 1, M):
            d = (vecs[i] - vecs[j]).norm(2).item()
            dists.append(d)
    return {
        "worker_param_dist_mean": sum(dists) / len(dists),
        "worker_param_dist_max":  max(dists),
    }


def pseudo_gradient_vs_reference_angle(
    avg_pseudo_grad: Dict[str, Tensor],
    reference_grad: Optional[Dict[str, Tensor]],
) -> float:
    """
    Cosine angle between the averaged pseudo-gradient and a synchronous
    reference gradient sampled at the sync point.
    Returns 1.0 if no reference is provided (undefined angle).
    """
    if reference_grad is None:
        return float("nan")

    pg_vec  = _flatten(avg_pseudo_grad)
    ref_vec = _flatten(reference_grad)
    cos = torch.nn.functional.cosine_similarity(
        pg_vec.unsqueeze(0), ref_vec.unsqueeze(0)
    ).item()
    return cos


def outer_momentum_buffer_norm(velocity: Dict[str, Tensor]) -> float:
    """
    L2 norm of the outer Nesterov velocity buffer.
    Tests whether outer memory drives acceleration.
    """
    return _flatten(velocity).norm(2).item()


def compute_all_geometry_metrics(
    pseudo_grads:    List[Dict[str, Tensor]],
    worker_params:   List[Dict[str, Tensor]],
    avg_pseudo_grad: Dict[str, Tensor],
    reference_grad:  Optional[Dict[str, Tensor]] = None,
    velocity:        Optional[Dict[str, Tensor]]  = None,
) -> dict:
    """
    Compute and return all geometry metrics as a flat dict for logging.
    """
    metrics: dict = {}
    metrics["pg_cosine_sim"]  = pseudo_gradient_cosine_similarity(pseudo_grads)
    metrics.update(pseudo_gradient_norm_dispersion(pseudo_grads))
    metrics.update(worker_parameter_distance(worker_params))
    metrics["pg_ref_angle"]   = pseudo_gradient_vs_reference_angle(avg_pseudo_grad, reference_grad)
    if velocity is not None:
        metrics["outer_v_norm"] = outer_momentum_buffer_norm(velocity)
    return {k: v for k, v in metrics.items() if not isinstance(v, list)}
