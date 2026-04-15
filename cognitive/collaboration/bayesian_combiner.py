"""Bayesian module combiner (Tenenbaum layer).

Maintains a posterior distribution over how much to weight each
Spelke core knowledge module for the current game. Updated after
each observation via approximate Bayesian inference.

P(module_weights | observations) ∝ P(observations | module_weights) · P(module_weights)

The likelihood is the product of module firing rates. The prior is
a Dirichlet that starts uniform and sharpens as evidence accumulates.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from .knowledge_module import KnowledgeModule, ModuleScore, ALL_MODULES

logger = logging.getLogger(__name__)


class BayesianCombiner:
    """Learns which core knowledge modules explain the current game.

    Each module gets a weight (pseudo-count in a Dirichlet). After
    each observation, modules that fire strongly get their weight
    boosted. The combined score is a weighted sum of firing rates.

    Attributes:
        modules: Instantiated core knowledge modules.
        weights: Per-module Dirichlet pseudo-counts.
        _total_updates: Number of Bayesian updates performed.
    """

    def __init__(self) -> None:
        self.modules: List[KnowledgeModule] = [cls() for cls in ALL_MODULES]
        # Start with uniform Dirichlet prior (α=1 for each module).
        self.weights: Dict[str, float] = {m.name: 1.0 for m in self.modules}
        self._total_updates: int = 0
        self._last_scores: Dict[str, ModuleScore] = {}

    def update(self, scene: Any, delta: Any, action: Any) -> Dict[str, float]:
        """Score all modules against the current observation and update posterior.

        Args:
            scene: Current SceneGraph.
            delta: StateDelta from the last action.
            action: The action that was taken.

        Returns:
            Dict of module_name → normalized posterior weight.
        """
        self._last_scores = {}
        for module in self.modules:
            score = module.score(scene, delta, action)
            self._last_scores[module.name] = score

            # Bayesian update: boost pseudo-count by firing rate.
            # High-firing modules accumulate more evidence.
            self.weights[module.name] += score.firing_rate

        self._total_updates += 1
        return self.get_normalized_weights()

    def get_normalized_weights(self) -> Dict[str, float]:
        """Return current posterior weights normalized to sum to 1.

        Returns:
            Dict of module_name → weight in [0, 1], summing to 1.
        """
        total = sum(self.weights.values())
        if total < 1e-9:
            n = len(self.weights)
            return {k: 1.0 / n for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}

    def combined_score(self) -> float:
        """Weighted combination of module firing rates.

        This is the agent's overall confidence in its world model.
        Used as the success_rate input to the PhaseController.

        Returns:
            Float in [0, 1].
        """
        if not self._last_scores:
            return 0.5

        normalized = self.get_normalized_weights()
        total = 0.0
        for module in self.modules:
            ms = self._last_scores.get(module.name)
            if ms is not None:
                total += normalized[module.name] * ms.firing_rate

        return total

    def get_dominant_module(self) -> str:
        """Return the name of the highest-weighted module.

        Returns:
            Module name string.
        """
        normalized = self.get_normalized_weights()
        return max(normalized, key=normalized.get)

    def get_module_report(self) -> Dict[str, Any]:
        """Summary of current module states for logging.

        Returns:
            Dict with module names, weights, and last firing rates.
        """
        normalized = self.get_normalized_weights()
        report = {}
        for module in self.modules:
            ms = self._last_scores.get(module.name)
            report[module.name] = {
                "weight": round(normalized[module.name], 3),
                "firing_rate": round(ms.firing_rate, 3) if ms else 0.0,
                "evidence": ms.evidence[:2] if ms else [],
            }
        return report

    def reset(self) -> None:
        """Reset to uniform prior (call between games, not levels).

        Keeps module instances but resets weights.
        """
        self.weights = {m.name: 1.0 for m in self.modules}
        self._total_updates = 0
        self._last_scores = {}
        # Reset stateful modules.
        for m in self.modules:
            if hasattr(m, 'player_detected'):
                m.player_detected = False
