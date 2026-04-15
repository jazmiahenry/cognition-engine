"""Spelke-inspired core knowledge modules.

Each module is a fixed feature detector for one domain of core
knowledge. The module's score() method returns a firing rate —
how confidently this module explains the current observations.
Rules within a module are the dendritic tree; score() is the soma's
output firing rate that propagates across synapses.

Modules never learn internally. They evaluate observations against
their fixed domain and report confidence. The Bayesian combiner
learns which modules to trust for which game.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ModuleScore:
    """Output of a KnowledgeModule evaluation.

    Attributes:
        firing_rate: Confidence that this module explains the current state [0, 1].
        evidence: Specific observations supporting this score.
        rules_contributed: Rules this module would contribute to a borrower.
    """

    firing_rate: float
    evidence: List[str] = field(default_factory=list)
    rules_contributed: List[Any] = field(default_factory=list)


class KnowledgeModule(ABC):
    """Base class for core knowledge modules.

    Each subclass implements a Spelke-style innate knowledge domain.
    The module evaluates observations and returns a firing rate.
    """

    name: str = "base"

    @abstractmethod
    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        """Evaluate how well this module explains the current observation.

        Args:
            scene: Current SceneGraph.
            delta: StateDelta from the last action (None on first tick).
            action: The action that was taken (None on first tick).

        Returns:
            ModuleScore with firing_rate in [0, 1].
        """
        raise NotImplementedError


class ObjectPersistence(KnowledgeModule):
    """Objects continue to exist across frames and maintain identity.

    High firing rate when tracked objects maintain consistent IDs.
    Low firing rate when objects appear/disappear unexpectedly.
    """

    name = "object_persistence"

    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        if delta is None:
            return ModuleScore(firing_rate=0.5, evidence=["no delta yet"])

        appeared = len(getattr(delta, 'objects_appeared', []))
        disappeared = len(getattr(delta, 'objects_disappeared', []))
        total = len(getattr(scene, 'objects', {}))

        if total == 0:
            return ModuleScore(firing_rate=0.5)

        # Persistence = fraction of objects that remained stable.
        churned = appeared + disappeared
        stability = max(0.0, 1.0 - churned / max(1, total))

        evidence = []
        if churned == 0:
            evidence.append("all objects persisted")
        else:
            evidence.append(f"{churned} objects changed existence")

        return ModuleScore(firing_rate=stability, evidence=evidence)


class Solidity(KnowledgeModule):
    """Objects occupy contiguous space and don't pass through each other.

    High firing rate when movement actions cause no overlap between objects.
    Detects wall-like behavior: actions that produce noops near dense objects.
    """

    name = "solidity"

    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        if delta is None:
            return ModuleScore(firing_rate=0.5)

        is_noop = getattr(delta, 'is_noop', True)
        action_name = getattr(action, 'name', str(action)) if action else ""
        is_movement = action_name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

        if is_movement and is_noop:
            # Movement blocked = evidence of solidity (walls).
            return ModuleScore(
                firing_rate=0.9,
                evidence=["movement blocked — solid boundary detected"],
            )
        elif is_movement and not is_noop:
            # Movement succeeded = space is open, solidity consistent.
            return ModuleScore(
                firing_rate=0.7,
                evidence=["movement succeeded — open space"],
            )

        return ModuleScore(firing_rate=0.5)


class Agency(KnowledgeModule):
    """Self-propelled entities respond to directional actions.

    High firing rate when exactly one object moves in response to
    a directional action — confirming a controllable player exists.
    """

    name = "agency"

    def __init__(self) -> None:
        self.player_detected: bool = False

    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        if delta is None:
            return ModuleScore(firing_rate=0.3)

        action_name = getattr(action, 'name', '') if action else ''
        is_movement = action_name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

        if not is_movement:
            return ModuleScore(firing_rate=0.5 if self.player_detected else 0.3)

        movers = []
        for obj_delta in getattr(delta, 'object_deltas', []):
            if 'centroid' in obj_delta.property_changes:
                movers.append(obj_delta.obj_id)

        if len(movers) == 1:
            self.player_detected = True
            return ModuleScore(
                firing_rate=0.95,
                evidence=[f"single mover obj_{movers[0]} — agent detected"],
            )
        elif len(movers) == 0 and self.player_detected:
            return ModuleScore(
                firing_rate=0.7,
                evidence=["no movement — agent blocked"],
            )
        elif len(movers) > 1:
            return ModuleScore(
                firing_rate=0.4,
                evidence=[f"{len(movers)} movers — ambiguous agency"],
            )

        return ModuleScore(firing_rate=0.3)


class ContactCausality(KnowledgeModule):
    """Effects require spatial proximity between action target and affected object.

    High firing rate when click actions affect only the clicked object
    or spatially adjacent objects, not distant ones.
    """

    name = "contact_causality"

    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        if delta is None:
            return ModuleScore(firing_rate=0.5)

        is_noop = getattr(delta, 'is_noop', True)
        action_name = getattr(action, 'name', '') if action else ''
        is_click = action_name in ("ACTION5", "ACTION6", "ACTION7")

        if is_click and not is_noop:
            # Click caused a change — evidence of contact causality.
            changed_count = len(getattr(delta, 'object_deltas', []))
            if changed_count <= 2:
                return ModuleScore(
                    firing_rate=0.9,
                    evidence=[f"click affected {changed_count} nearby object(s)"],
                )
            else:
                # Many objects changed — action-at-a-distance?
                return ModuleScore(
                    firing_rate=0.4,
                    evidence=[f"click affected {changed_count} objects — non-local"],
                )
        elif is_click and is_noop:
            return ModuleScore(
                firing_rate=0.6,
                evidence=["click had no effect — no contact target"],
            )

        return ModuleScore(firing_rate=0.5)


class SpatialGeometry(KnowledgeModule):
    """Distances, directions, containment, and adjacency structure.

    High firing rate when the scene has clear spatial structure
    (objects at regular intervals, clear boundaries, containment).
    """

    name = "spatial_geometry"

    def score(self, scene: Any, delta: Any, action: Any) -> ModuleScore:
        objects = getattr(scene, 'objects', {})
        relations = getattr(scene, 'relations', [])

        if len(objects) < 2:
            return ModuleScore(firing_rate=0.3)

        # Measure spatial structure: ratio of adjacent pairs to total pairs.
        adjacent_count = sum(1 for r in relations if r.relation == 'adjacent')
        total_pairs = max(1, len(objects) * (len(objects) - 1) // 2)
        adjacency_density = adjacent_count / total_pairs

        # High adjacency density = structured layout (grid, maze).
        # Low density = scattered objects.
        firing = 0.3 + 0.6 * min(1.0, adjacency_density * 5)

        evidence = [f"adjacency density={adjacency_density:.2f}, {len(objects)} objects"]

        return ModuleScore(firing_rate=min(1.0, firing), evidence=evidence)


# Registry of all core modules.
ALL_MODULES = [
    ObjectPersistence,
    Solidity,
    Agency,
    ContactCausality,
    SpatialGeometry,
]
