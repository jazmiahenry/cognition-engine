"""Hebbian plasticity engine for synaptic weight updates.

Implements Oja-style bounded Hebbian learning:
  LTP: w += α·(1-w)  — bounded potentiation, strong synapses resist further strengthening
  LTD: w -= β·w      — proportional depression, strong synapses resist depression
  Decay: w -= δ       — use-it-or-lose-it for inactive connections
  Homeostasis: row sum capped at max_sum to prevent runaway potentiation

The step() sequence matters:
  1. Decay (degrade before LTP can rescue)
  2. Phase assignment (choose partners from current weights)
  3. Scoring (evaluate module firing rates)
  4. LTP/LTD (update based on is_helpful)
  5. Homeostasis (cap after all updates)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Tuple

from .synaptic_network import SynapticNetwork

logger = logging.getLogger(__name__)


class CouplingPhase(Enum):
    """Staged coupling protocol based on agent success rate."""

    AMODULAR = "amodular"       # sr ≥ 0.58 — solo, no borrowing
    BIMODAL = "bimodal"         # 0.38 ≤ sr < 0.58 — single best partner
    MULTIMODAL = "multimodal"   # sr < 0.38 — top 3 partners


class HebbianEngine:
    """Plasticity rule engine managing LTP/LTD/decay/homeostasis.

    Attributes:
        network: The SynapticNetwork holding W[i][j].
        alpha: LTP learning rate.
        beta: LTD learning rate.
        delta: Passive decay rate per cycle.
        max_row_sum: Homeostatic cap on incoming weight sum.
    """

    def __init__(
        self,
        network: SynapticNetwork,
        alpha: float = 0.1,
        beta: float = 0.05,
        delta: float = 0.01,
        max_row_sum: float = 2.5,
    ) -> None:
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.max_row_sum = max_row_sum

    def get_phase(self, success_rate: float) -> CouplingPhase:
        """Determine coupling phase from agent's success rate.

        Args:
            success_rate: Fraction of actions that produced non-noop changes.

        Returns:
            CouplingPhase determining borrowing behavior.
        """
        if success_rate >= 0.58:
            return CouplingPhase.AMODULAR
        elif success_rate >= 0.38:
            return CouplingPhase.BIMODAL
        else:
            return CouplingPhase.MULTIMODAL

    def get_partners(
        self, agent_id: str, phase: CouplingPhase
    ) -> List[Tuple[str, float]]:
        """Get partner list based on coupling phase.

        Args:
            agent_id: The requesting agent's game_id.
            phase: Current coupling phase.

        Returns:
            List of (partner_id, weight) tuples. Empty for AMODULAR.
        """
        if phase == CouplingPhase.AMODULAR:
            return []
        elif phase == CouplingPhase.BIMODAL:
            best = self.network.get_best_partner(agent_id)
            return [best] if best and best[1] > 0.01 else []
        else:  # MULTIMODAL
            partners = self.network.get_partners(agent_id, top_k=3)
            return [(pid, w) for pid, w in partners if w > 0.01]

    def apply_ltp(self, i: str, j: str) -> None:
        """Long-term potentiation: strengthen synapse i←j.

        Oja-style bounded: w += α·(1-w). Strong synapses saturate
        naturally — a weight at 0.9 only gets +0.01 per potentiation.

        Args:
            i: Receiving agent's game_id.
            j: Donor agent's game_id.
        """
        w = self.network.get_weight(i, j)
        w_new = w + self.alpha * (1.0 - w)
        self.network.set_weight(i, j, w_new)
        logger.debug("LTP: W[%s][%s] %.4f → %.4f", i, j, w, w_new)

    def apply_ltd(self, i: str, j: str) -> None:
        """Long-term depression: weaken synapse i←j.

        Proportional: w -= β·w. Strong synapses resist depression —
        a connection at 0.8 loses 0.04, one at 0.1 loses 0.005.

        Args:
            i: Receiving agent's game_id.
            j: Donor agent's game_id.
        """
        w = self.network.get_weight(i, j)
        w_new = w - self.beta * w
        self.network.set_weight(i, j, w_new)
        logger.debug("LTD: W[%s][%s] %.4f → %.4f", i, j, w, w_new)

    def apply_decay(self, agent_id: str) -> None:
        """Passive decay on all incoming synapses for this agent.

        Use-it-or-lose-it: connections not actively reinforced degrade.

        Args:
            agent_id: The agent whose incoming synapses decay.
        """
        self.network.decay_all(agent_id, self.delta)

    def apply_homeostasis(self, agent_id: str) -> None:
        """Homeostatic normalization: cap row sum.

        Prevents runaway potentiation from multiple successful borrows
        in one cycle.

        Args:
            agent_id: The agent whose row to normalize.
        """
        self.network.normalize_row(agent_id, self.max_row_sum)

    def step(
        self,
        agent_id: str,
        success_rate: float,
        q_before: float,
        q_after: float,
        borrowed_from: Optional[List[str]] = None,
    ) -> CouplingPhase:
        """Execute one full Hebbian cycle.

        Sequence: decay → phase → LTP/LTD → homeostasis.

        Args:
            agent_id: This agent's game_id.
            success_rate: Current success rate for phase assignment.
            q_before: MCTS Q-value before borrowing (or 0 if no MCTS).
            q_after: MCTS Q-value after borrowing.
            borrowed_from: List of donor game_ids that were used this cycle.

        Returns:
            The CouplingPhase assigned this cycle.
        """
        # 1. Decay
        self.apply_decay(agent_id)

        # 2. Phase assignment
        phase = self.get_phase(success_rate)

        # 3-4. LTP/LTD based on whether borrowing helped
        if borrowed_from:
            helpful = self._is_helpful(q_before, q_after)
            for donor_id in borrowed_from:
                if helpful:
                    self.apply_ltp(agent_id, donor_id)
                    logger.info(
                        "Hebbian LTP: %s ← %s (Q %.4f → %.4f)",
                        agent_id, donor_id, q_before, q_after,
                    )
                else:
                    self.apply_ltd(agent_id, donor_id)
                    logger.info(
                        "Hebbian LTD: %s ← %s (Q %.4f → %.4f)",
                        agent_id, donor_id, q_before, q_after,
                    )

        # 5. Homeostasis
        self.apply_homeostasis(agent_id)

        return phase

    def seed_from_mentor(
        self, mentor_id: str, all_agent_ids: list, base_weight: float = 0.15
    ) -> None:
        """Seed initial synaptic weights from a mentor to all other agents.

        Called when an agent completes a level. Bootstraps the network
        so BIMODAL agents have a partner to reach for.

        Args:
            mentor_id: Game ID of the agent that completed a level.
            all_agent_ids: All agent game IDs in the swarm.
            base_weight: Initial weight to seed.
        """
        for agent_id in all_agent_ids:
            if agent_id == mentor_id:
                continue
            current = self.network.get_weight(agent_id, mentor_id)
            if current < base_weight:
                self.network.set_weight(agent_id, mentor_id, base_weight)
        logger.info("Mentor %s seeded weights to %d agents", mentor_id, len(all_agent_ids) - 1)

    def _is_helpful(self, q_before: float, q_after: float) -> bool:
        """The key epistemological judgement.

        Did borrowing rules bring the agent closer to solving the game
        than it was before? Measured by MCTS Q-value improvement.

        Args:
            q_before: Q-value before incorporating borrowed rules.
            q_after: Q-value after incorporating borrowed rules.

        Returns:
            True if borrowing improved performance.
        """
        return q_after > q_before + 0.01
