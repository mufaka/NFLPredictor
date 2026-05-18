"""Combination enumeration + (later) pipeline orchestration (§5, TR-RUNG-07)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TrainingConfig


TRIVIAL_RUNGS: frozenset[str] = frozenset({"mean", "team_mean"})
LEARNED_RUNGS: frozenset[str] = frozenset({"linear", "mlp"})


@dataclass(frozen=True)
class Combination:
    """One (rung, shape, strategy) the training build will train and predict for."""

    rung: str
    shape: str       # "none" for trivial rungs; "flat" | "pos" for learned rungs
    strategy: str    # "S1" or "S3"

    @property
    def is_trivial(self) -> bool:
        return self.rung in TRIVIAL_RUNGS

    @property
    def is_learned(self) -> bool:
        return self.rung in LEARNED_RUNGS


def enumerate_combinations(config: TrainingConfig) -> list[Combination]:
    """Cartesian product of ``rungs × shapes × strategies``, filtered per TR-RUNG-07.

    - Trivial rungs (``mean``, ``team_mean``) pair only with ``shape = "none"``
      (config.shapes is irrelevant for them).
    - Learned rungs (``linear``, ``mlp``) pair with each entry in
      ``config.shapes``.

    Order is stable: outer iteration over config.rungs, then config.shapes
    (learned rungs only), then config.strategies. This keeps manifest /
    log layout predictable across runs.
    """
    combos: list[Combination] = []
    for rung in config.rungs:
        if rung in TRIVIAL_RUNGS:
            for strategy in config.strategies:
                combos.append(Combination(rung=rung, shape="none", strategy=strategy))
        elif rung in LEARNED_RUNGS:
            for shape in config.shapes:
                for strategy in config.strategies:
                    combos.append(
                        Combination(rung=rung, shape=shape, strategy=strategy)
                    )
        else:
            # config.py rejects unknown rungs; defense-in-depth.
            raise ValueError(f"unknown rung in config: {rung!r}")
    return combos
