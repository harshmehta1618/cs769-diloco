"""
Outer optimizers for DiLoCo-family methods.

Three outer-optimizer types as required by the experimental design:

  "none"      → No outer optimizer (used by DP-AdamW and SW-AdamW).
  "nesterov"  → Nesterov outer optimizer (DiLoCo / SNOO).
                Applies Nesterov momentum to averaged pseudo-gradients.
  "avg"       → Plain averaging (PeriodicAvg / FedAvg).
                Sets θ = mean of worker parameters; equivalent to
                pseudo-gradient = θ_ref − avg(θ_w), outer step = identity.

Pseudo-gradient definition (per the DiLoCo paper):
    Δ_w = θ_ref − θ_w     (for worker w after H inner steps)
    avg_Δ = mean_w(Δ_w)   (averaged across workers)

Nesterov outer update:
    v_{t+1}  = momentum * v_t + avg_Δ_t
    θ_{t+1}  = θ_ref − outer_lr * (avg_Δ_t + momentum * v_{t+1})

Plain averaging outer update:
    θ_{t+1}  = θ_ref − avg_Δ_t   ⟺   θ_{t+1} = mean(θ_w)
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Nesterov outer optimizer (DiLoCo / SNOO)
# ---------------------------------------------------------------------------

class NesterovOuterOptimizer:
    """
    Nesterov momentum applied to averaged pseudo-gradients.

    Parameters
    ----------
    params      : named parameter iterator of the reference model
    outer_lr    : outer learning rate
    momentum    : Nesterov momentum coefficient (β)
    nesterov    : if False, uses plain heavy-ball (SGD with momentum)
    """

    def __init__(
        self,
        params: Iterator[tuple[str, Tensor]],
        outer_lr: float   = 0.7,
        momentum: float   = 0.9,
        nesterov: bool    = True,
    ) -> None:
        self.outer_lr  = outer_lr
        self.momentum  = momentum
        self.nesterov  = nesterov
        # Velocity buffer — same shape as parameters, initialised to 0
        self._velocity: Dict[str, Tensor] = {
            name: torch.zeros_like(p) for name, p in params
        }

    def step(
        self,
        ref_params: Dict[str, Tensor],
        avg_pseudo_grad: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Apply one outer update.

        Parameters
        ----------
        ref_params       : {name: tensor}   — θ_ref (model params BEFORE inner loop)
        avg_pseudo_grad  : {name: tensor}   — averaged Δ = mean_w(θ_ref − θ_w)

        Returns
        -------
        new_params : {name: tensor}  — updated parameter values to broadcast
        """
        new_params: Dict[str, Tensor] = {}
        for name, g in avg_pseudo_grad.items():
            v = self._velocity[name]
            v.mul_(self.momentum).add_(g)

            if self.nesterov:
                # θ_{t+1} = θ_ref − outer_lr * (g + β * v_{t+1})
                update = g + self.momentum * v
            else:
                # Plain heavy-ball: θ_{t+1} = θ_ref − outer_lr * v_{t+1}
                update = v.clone()

            new_params[name] = ref_params[name] - self.outer_lr * update
        return new_params

    def state_dict(self) -> dict:
        return {
            "velocity":   {k: v.cpu() for k, v in self._velocity.items()},
            "outer_lr":   self.outer_lr,
            "momentum":   self.momentum,
            "nesterov":   self.nesterov,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.outer_lr = sd["outer_lr"]
        self.momentum = sd["momentum"]
        self.nesterov = sd["nesterov"]
        for k, v in sd["velocity"].items():
            if k in self._velocity:
                self._velocity[k].copy_(v)


# ---------------------------------------------------------------------------
# Plain-averaging outer optimizer (PeriodicAvg / FedAvg)
# ---------------------------------------------------------------------------

class AvgOuterOptimizer:
    """
    No outer momentum — just average the worker parameters.

    Equivalent to outer_lr = 1.0, momentum = 0 in the Nesterov variant.
    """

    def step(
        self,
        ref_params: Dict[str, Tensor],          # unused but kept for API parity
        avg_pseudo_grad: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Returns θ_ref − avg_Δ = mean(θ_w).
        """
        return {name: ref_params[name] - g for name, g in avg_pseudo_grad.items()}

    def state_dict(self) -> dict:
        return {"type": "avg"}

    def load_state_dict(self, sd: dict) -> None:
        pass  # stateless


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_outer_optimizer(
    outer_type: str,
    ref_params_iter: Optional[Iterator[tuple[str, Tensor]]],
    outer_lr: float = 0.7,
    momentum: float = 0.9,
    nesterov: bool  = True,
):
    """
    Parameters
    ----------
    outer_type       : "none" | "nesterov" | "avg"
    ref_params_iter  : iterator of (name, param) — only needed for "nesterov"
    """
    if outer_type == "nesterov":
        assert ref_params_iter is not None
        return NesterovOuterOptimizer(
            ref_params_iter,
            outer_lr=outer_lr,
            momentum=momentum,
            nesterov=nesterov,
        )
    elif outer_type == "avg":
        return AvgOuterOptimizer()
    elif outer_type == "none":
        return None
    else:
        raise ValueError(f"Unknown outer optimizer type: {outer_type!r}")
