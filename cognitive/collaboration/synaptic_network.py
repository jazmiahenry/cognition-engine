"""Synaptic weight network between agents.

W[i][j] represents how much agent i should trust agent j's rules.
Weights live here, not inside agents — the synapse is between cells,
not inside one. Thread-safe for concurrent agent access.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SynapticNetwork:
    """Thread-safe weight matrix between agents.

    Weights start near zero and evolve through HebbianEngine updates.
    The network knows nothing about rules — it only tracks trust
    relationships between agent game_ids.

    Attributes:
        _W: Nested dict W[i][j] = float weight.
        _lock: Threading lock for concurrent access.
    """

    def __init__(self) -> None:
        self._W: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def get_weight(self, i: str, j: str) -> float:
        """Get synapse weight from j to i.

        Args:
            i: Receiving agent's game_id.
            j: Donor agent's game_id.

        Returns:
            Current weight, 0.0 if no connection exists.
        """
        with self._lock:
            return self._W.get(i, {}).get(j, 0.0)

    def set_weight(self, i: str, j: str, w: float) -> None:
        """Set synapse weight from j to i.

        Args:
            i: Receiving agent's game_id.
            j: Donor agent's game_id.
            w: New weight value.
        """
        with self._lock:
            if i not in self._W:
                self._W[i] = {}
            self._W[i][j] = max(0.0, w)

    def get_partners(self, i: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k partners for agent i, sorted by weight descending.

        Args:
            i: Agent's game_id.
            top_k: Number of partners to return.

        Returns:
            List of (game_id, weight) tuples.
        """
        with self._lock:
            row = self._W.get(i, {})
            sorted_partners = sorted(row.items(), key=lambda x: -x[1])
            return sorted_partners[:top_k]

    def get_best_partner(self, i: str) -> Optional[Tuple[str, float]]:
        """Get the single highest-weight partner for agent i.

        Args:
            i: Agent's game_id.

        Returns:
            (game_id, weight) or None if no connections.
        """
        partners = self.get_partners(i, top_k=1)
        return partners[0] if partners else None

    def decay_all(self, i: str, delta: float) -> None:
        """Apply passive decay to all of agent i's incoming synapses.

        Use-it-or-lose-it: inactive connections fade.

        Args:
            i: Agent's game_id.
            delta: Decay rate (subtracted from each weight).
        """
        with self._lock:
            row = self._W.get(i, {})
            for j in list(row.keys()):
                row[j] = max(0.0, row[j] - delta)
                if row[j] < 1e-6:
                    del row[j]

    def normalize_row(self, i: str, max_sum: float = 2.5) -> None:
        """Homeostatic normalization: cap row sum to prevent runaway.

        Args:
            i: Agent's game_id.
            max_sum: Maximum allowed sum of incoming weights.
        """
        with self._lock:
            row = self._W.get(i, {})
            if not row:
                return
            total = sum(row.values())
            if total > max_sum:
                factor = max_sum / total
                for j in row:
                    row[j] *= factor

    def all_agents(self) -> List[str]:
        """Return all agent game_ids that have any connections.

        Returns:
            List of game_id strings.
        """
        with self._lock:
            agents = set(self._W.keys())
            for row in self._W.values():
                agents.update(row.keys())
            return list(agents)

    def ensure_agent(self, game_id: str) -> None:
        """Ensure an agent has an entry in the weight matrix.

        Args:
            game_id: Agent's game identifier.
        """
        with self._lock:
            if game_id not in self._W:
                self._W[game_id] = {}
