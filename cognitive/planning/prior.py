"""Hierarchical action prior for MCTS rollouts and expansion ordering.

Three stacked levels of prior knowledge:

  Level 0 — Core Knowledge (Spelke 2007)
    Hardcoded weights encoding the expectation that the player agent
    mostly moves (ACTION1-4 map to cardinal directions) rather than
    using special actions or clicking arbitrary grid positions.
    Always available — solves the cold-start problem.

  Level 1 — Empirical / RuleEngine
    High-confidence rule confidences are folded in as multipliers on
    the Level 0 weights, skewing the prior toward actions that have
    been empirically observed to cause useful state transitions.

  Level 2 — Library Transfer (stub)
    Planned: transfer weights from the offline environment library
    when novelty detection says the current environment matches a
    known archetype. Returns 1.0 (no effect) until implemented.

The three levels are composed multiplicatively so that Level 0 always
exerts a nonzero influence and prevents pure-random rollouts even in
fully novel environments.

Spatial bias is applied on top of all three levels: during rollout the
agent is nudged toward the spatial direction that points from its
current position to the estimated goal position.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from .mcts import ClickAction, MCTSState


# ---------------------------------------------------------------------------
# Level 0 — Core Knowledge constants
# ---------------------------------------------------------------------------

# Movement actions get a boost (directional agency prior).
# Special actions (ACTION5-7) and RESET are down-weighted.
_CORE_WEIGHTS: Dict[str, float] = {
    "ACTION1": 1.6,  # MOVE_UP
    "ACTION2": 1.6,  # MOVE_DOWN
    "ACTION3": 1.6,  # MOVE_LEFT
    "ACTION4": 1.6,  # MOVE_RIGHT
    "ACTION5": 1.0,  # context-dependent special action
    "ACTION6": 0.8,
    "ACTION7": 0.5,  # rare / high-cost action
    "RESET": 0.1,  # only useful at explicit junctions
}

# Direction unit vectors for each movement action (dy, dx in grid space).
_ACTION_DIRECTION: Dict[str, Tuple[float, float]] = {
    "ACTION1": (-1.0, 0.0),  # up   → decreasing y
    "ACTION2": (1.0, 0.0),   # down → increasing y
    "ACTION3": (0.0, -1.0),  # left → decreasing x
    "ACTION4": (0.0, 1.0),   # right→ increasing x
}

# Grid diagonal — used to normalise distances.
_GRID_DIAGONAL: float = math.sqrt(64.0 ** 2 + 64.0 ** 2)  # ≈ 90.5


# ---------------------------------------------------------------------------
# HierarchicalPrior
# ---------------------------------------------------------------------------


class HierarchicalPrior:
    """Three-level hierarchical action prior for MCTS.

    Combines Core Knowledge, RuleEngine empirical confidences, and
    (optionally) a library transfer signal into a single action-weight
    function used for expansion ordering and rollout sampling.

    Attributes:
        rules: Rule objects from RuleEngine, updated after each real action.
        player_id: obj_id of the identified player agent.
        player_pos: Latest (cy, cx) of the player.
        goal_pos: Estimated goal position (cy, cx).
        _lib_weight: Level-2 library transfer weight (default 1.0 = off).
    """

    def __init__(self, rules: Optional[List] = None) -> None:
        self.rules: List = rules if rules is not None else []
        self.player_id: Optional[int] = None
        self.player_pos: Optional[Tuple[float, float]] = None
        self.goal_pos: Optional[Tuple[float, float]] = None
        self._lib_weight: float = 1.0  # Level 2 stub

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def action_weight(self, action: object, state: MCTSState) -> float:
        """Composite weight for one action from the current state.

        Used by MCTSTree._expand() to order untried actions so the most
        promising direction is explored first. ClickActions get a
        proximity bonus when their target is near the goal.

        Args:
            action: A GameAction, ClickAction, or any named object.
            state: Current MCTS state.

        Returns:
            Non-negative float weight. Higher = more desirable.
        """
        if isinstance(action, ClickAction):
            return self._click_weight(action, state)

        name = action.name if hasattr(action, "name") else str(action)
        w0 = self._core_weight(name, state)    # Level 0 + spatial
        w1 = self._empirical_weight(name)       # Level 1 RuleEngine
        w2 = self._lib_weight                   # Level 2 stub
        # Multiplicative composition; Level 0 never drops to zero.
        return w0 * w1 * w2

    def sample_action(
        self, state: MCTSState, available_actions: List
    ) -> object:
        """Sample an action proportional to composite weights.

        Used in MCTS rollouts for fast prior-guided play-outs.
        Falls back to uniform random if all weights collapse to zero.

        Args:
            state: Current rollout state.
            available_actions: Non-RESET actions to choose from.

        Returns:
            A single sampled action.
        """
        weights = [
            max(self.action_weight(a, state), 1e-6)
            for a in available_actions
        ]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for action, w in zip(available_actions, weights):
            cumulative += w
            if r <= cumulative:
                return action
        return available_actions[-1]  # numerical fallback

    def proximity_reward(self, state: MCTSState) -> float:
        """Shaping reward based on player-to-goal distance.

        Returns a nonzero signal even when no goal has been reached,
        guiding MCTS to prefer paths that approach the goal region.

        Args:
            state: Terminal rollout state.

        Returns:
            Float in [0.0, 0.4]. Zero if player or goal is unknown.
        """
        if self.player_id is None or self.goal_pos is None:
            return 0.0
        entry = state.objs.get(self.player_id)
        if entry is None:
            return 0.0
        _, py, px = entry
        gy, gx = self.goal_pos
        dist = math.sqrt((py - gy) ** 2 + (px - gx) ** 2)
        # Reward is 0.4 at zero distance, decays linearly to 0 at GRID_DIAGONAL.
        return 0.4 * max(0.0, 1.0 - dist / _GRID_DIAGONAL)

    def update_player(self, delta: object, scene: object) -> None:
        """Identify the player from the first observed movement.

        Core Knowledge prior: the object that moves in response to a
        directional action is the controllable agent. Once identified,
        we never reassign unless the object disappears.

        Args:
            delta: StateDelta from world model.
            scene: SceneGraph after the action.
        """
        if self.player_id is not None:
            # Update position if player still exists.
            obj = scene.objects.get(self.player_id)
            if obj is not None:
                self.player_pos = obj.centroid
            return

        # Detect player by movement (largest displacement).
        best_id: Optional[int] = None
        best_dist: float = 0.0
        for obj_delta in getattr(delta, "object_deltas", []):
            if "centroid" in obj_delta.property_changes:
                old_pos, new_pos = obj_delta.property_changes["centroid"]
                if old_pos and new_pos:
                    d = math.sqrt(
                        (old_pos[0] - new_pos[0]) ** 2
                        + (old_pos[1] - new_pos[1]) ** 2
                    )
                    if d > best_dist:
                        best_dist = d
                        best_id = obj_delta.obj_id

        if best_id is not None and best_dist > 0.5:
            self.player_id = best_id
            obj = scene.objects.get(best_id)
            if obj is not None:
                self.player_pos = obj.centroid

        self.goal_pos = self._find_goal_pos(scene)

    def update_rules(self, rules: List) -> None:
        """Replace the rule list with a fresh copy from RuleEngine.

        Args:
            rules: Updated list of Rule objects.
        """
        self.rules = list(rules)

    def set_library_weight(self, w2: float) -> None:
        """Set Level-2 library transfer weight.

        Args:
            w2: Multiplicative weight from environment library match score.
        """
        self._lib_weight = max(0.5, w2)  # never zero-out entirely

    # ------------------------------------------------------------------
    # Level 0 helpers
    # ------------------------------------------------------------------

    def _core_weight(self, action_name: str, state: MCTSState) -> float:
        """Level 0 weight: Core Knowledge + spatial bias.

        Args:
            action_name: String name of the action.
            state: Current MCTS state (used for spatial bias direction).

        Returns:
            Float ≥ 0.1.
        """
        base = _CORE_WEIGHTS.get(action_name, 1.0)
        bias = self._spatial_bias(action_name, state)
        return base * bias

    def _spatial_bias(self, action_name: str, state: MCTSState) -> float:
        """Dot-product bias toward goal direction.

        Scales the Level 0 weight up to 1.8 when the action points
        directly at the goal and down to 0.3 when pointing away.

        Args:
            action_name: Name of the candidate action.
            state: Current MCTS state.

        Returns:
            Float in [0.3, 1.8]. Returns 1.0 when direction unknown.
        """
        direction = _ACTION_DIRECTION.get(action_name)
        if direction is None:
            return 1.0  # non-movement actions get no spatial adjustment

        if self.player_id is None or self.goal_pos is None:
            return 1.0  # spatial bias unavailable — use neutral weight

        entry = state.objs.get(self.player_id)
        if entry is None:
            return 1.0

        _, py, px = entry
        gy, gx = self.goal_pos
        goal_dy = gy - py
        goal_dx = gx - px
        goal_dist = math.sqrt(goal_dy ** 2 + goal_dx ** 2)
        if goal_dist < 1e-6:
            return 1.0  # already at goal position

        # Normalise goal vector and compute dot product with action direction.
        norm_dy = goal_dy / goal_dist
        norm_dx = goal_dx / goal_dist
        dot = direction[0] * norm_dy + direction[1] * norm_dx

        # Map [-1, 1] → [0.3, 1.8] with midpoint at 1.0.
        return 0.3 + 0.75 * (dot + 1.0)

    # ------------------------------------------------------------------
    # Click action weighting
    # ------------------------------------------------------------------

    def _click_weight(self, action: ClickAction, state: MCTSState) -> float:
        """Weight for a click action targeting a specific object.

        Click weight is a combination of:
          - Base click weight from Core Knowledge (ACTION5/6/7)
          - Goal proximity: objects near the estimated goal get a boost
          - Uniqueness: objects with rare colors are more likely targets

        Args:
            action: ClickAction with target coordinates and obj_id.
            state: Current MCTS state.

        Returns:
            Non-negative float weight.
        """
        base_name = (
            action.base_action.name
            if hasattr(action.base_action, "name")
            else str(action.base_action)
        )
        base_w = _CORE_WEIGHTS.get(base_name, 0.8)

        # Goal proximity bonus for the clicked position.
        goal_bonus = 1.0
        if self.goal_pos is not None:
            gy, gx = self.goal_pos
            dist = math.sqrt((action.y - gy) ** 2 + (action.x - gx) ** 2)
            # Closer to goal → higher weight, range [1.0, 2.0].
            goal_bonus = 1.0 + max(0.0, 1.0 - dist / _GRID_DIAGONAL)

        # Uniqueness bonus: rare colors in the state are more interesting.
        color_bonus = 1.0
        if action.target_obj_id is not None:
            entry = state.objs.get(action.target_obj_id)
            if entry is not None:
                target_color = entry[0]
                color_counts = {}
                for c, _, _ in state.objs.values():
                    color_counts[c] = color_counts.get(c, 0) + 1
                # Unique colors get up to 1.5x boost.
                color_bonus = 1.0 + 0.5 / color_counts.get(target_color, 1)

        return base_w * goal_bonus * color_bonus * self._lib_weight

    # ------------------------------------------------------------------
    # Level 1 helper
    # ------------------------------------------------------------------

    def _empirical_weight(self, action_name: str) -> float:
        """Level 1 weight derived from RuleEngine confidences.

        Aggregates over all high-confidence rules whose action_type
        matches this action. A rule with confidence 0.9 for a movement
        action that causes useful state transitions will nudge MCTS to
        try that direction earlier.

        Args:
            action_name: String name of the action.

        Returns:
            Float in [0.5, 2.0]. Returns 1.0 when no matching rules exist.
        """
        if not self.rules:
            return 1.0

        matching_confidences: List[float] = []
        for rule in self.rules:
            rule_type = rule.action_type
            # action_type can be an int ID or a string name.
            rule_name = rule_type if isinstance(rule_type, str) else None
            if rule_name == action_name and rule.confidence >= 0.35:
                matching_confidences.append(rule.confidence)

        if not matching_confidences:
            return 1.0

        mean_conf = sum(matching_confidences) / len(matching_confidences)
        # Map confidence [0.35, 1.0] → weight multiplier [0.8, 1.6].
        return 0.8 + 0.8 * (mean_conf - 0.35) / 0.65

    # ------------------------------------------------------------------
    # Goal estimation
    # ------------------------------------------------------------------

    def _find_goal_pos(self, scene: object) -> Optional[Tuple[float, float]]:
        """Estimate goal location using saliency heuristics.

        A salient object is one that is:
          - A unique color not shared by many other objects,
          - Smaller than average (goals are often small targets), and
          - Far from the player.

        Args:
            scene: SceneGraph to search.

        Returns:
            (cy, cx) of the most salient candidate, or None.
        """
        objects = getattr(scene, "objects", {})
        if not objects:
            return None

        color_count: Dict[int, int] = {}
        for obj in objects.values():
            color_count[obj.color] = color_count.get(obj.color, 0) + 1

        avg_area = sum(o.area for o in objects.values()) / len(objects)
        best_id: Optional[int] = None
        best_score: float = -float("inf")

        for oid, obj in objects.items():
            if oid == self.player_id:
                continue

            uniqueness = 1.0 / color_count[obj.color]
            size_score = max(0.0, 1.0 - obj.area / (avg_area + 1))
            dist_score = 0.0
            if self.player_pos is not None:
                py, px = self.player_pos
                cy, cx = obj.centroid
                dist_score = math.sqrt((cy - py) ** 2 + (cx - px) ** 2) / _GRID_DIAGONAL

            score = uniqueness * 0.5 + size_score * 0.3 + dist_score * 0.2
            if score > best_score:
                best_score = score
                best_id = oid

        if best_id is not None:
            return objects[best_id].centroid

        return None
