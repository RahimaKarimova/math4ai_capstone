"""Optimizer implementations for parameter-array based training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


ArrayLike = np.ndarray


def _zeros_like_params(params: Sequence[ArrayLike]) -> List[ArrayLike]:
    """Allocate one zero array per parameter array."""
    return [np.zeros_like(p) for p in params]


@dataclass
class SGD:
    """Vanilla SGD: theta <- theta - lr * grad."""

    lr: float = 0.05

    def step(self, params: Sequence[ArrayLike], grads: Sequence[ArrayLike]) -> None:
        for p, g in zip(params, grads):
            p -= self.lr * g


@dataclass
class Momentum:
    """Momentum optimizer with persistent velocity state."""

    lr: float = 0.05
    beta: float = 0.9
    velocity: List[ArrayLike] = field(default_factory=list)

    def _ensure_state(self, params: Sequence[ArrayLike]) -> None:
        if not self.velocity:
            self.velocity = _zeros_like_params(params)

    def step(self, params: Sequence[ArrayLike], grads: Sequence[ArrayLike]) -> None:
        self._ensure_state(params)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocity[i] = self.beta * self.velocity[i] + g
            p -= self.lr * self.velocity[i]


@dataclass
class Adam:
    """Adam optimizer with bias correction and persistent moments."""

    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: List[ArrayLike] = field(default_factory=list)
    v: List[ArrayLike] = field(default_factory=list)
    t: int = 0

    def _ensure_state(self, params: Sequence[ArrayLike]) -> None:
        if not self.m:
            self.m = _zeros_like_params(params)
        if not self.v:
            self.v = _zeros_like_params(params)

    def step(self, params: Sequence[ArrayLike], grads: Sequence[ArrayLike]) -> None:
        self._ensure_state(params)
        self.t += 1

        beta1_pow_t = self.beta1 ** self.t
        beta2_pow_t = self.beta2 ** self.t

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - beta1_pow_t)
            v_hat = self.v[i] / (1.0 - beta2_pow_t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def build_optimizer(name: str):
    """Create one of the assignment-required optimizers by name.

    Supported names: 'sgd', 'momentum', 'adam' (case-insensitive).
    """
    key = name.strip().lower()
    if key == "sgd":
        return SGD(lr=0.05)
    if key == "momentum":
        return Momentum(lr=0.05, beta=0.9)
    if key == "adam":
        return Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    raise ValueError(f"Unknown optimizer '{name}'. Use one of: sgd, momentum, adam.")


OPTIMIZER_BUILDERS: Dict[str, type] = {
    "sgd": SGD,
    "momentum": Momentum,
    "adam": Adam,
}

