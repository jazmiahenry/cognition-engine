"""Monte Carlo Tree Search planner for ARC-AGI-3.

UCT-based MCTS over simulated world-model states. Uses a lightweight
MCTSState representation (no pixel sets) so hundreds of simulations
can run within the API round-trip budget without counting as real actions.

Four phases per simulation:
  Selection    — UCT descent to a promising leaf
  Expansion    — try one untried action, prior-ordered
  Rollout      — fast rule-guided playout to depth limit
  Backprop     — propagate reward back to root

ClickAction wraps a GameAction + (x, y) target so the tree can reason
about clicking specific objects, not just cardinal movement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..perception.scene_graph import SceneGraph


# ---------------------------------------------------------------------------
# Lightweight state
# ---------------------------------------------------------------------------


@dataclass
class MCTSState:
    """Hashable, lightweight state for MCTS simulation.

    Stores only (color, cy, cx) per object — no pixel sets, no raw frame.
    Constructed from a SceneGraph in O(n_objects); suitable for thousands
    of copies per real action.

    Attributes:
        objs: Mapping from obj_id to (color, cy, cx).
        bg_color: Background colour index.
    """

    objs: dict  # {obj_id: (color, cy, cx)}
    bg_color: int
    _hash: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hash = hash(tuple(sorted(
            (oid, c, round(cy, 1), round(cx, 1))
            for oid, (c, cy, cx) in self.objs.items()
        )))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSState):
            return False
        return self._hash == other._hash

    @classmethod
    def from_scene(cls, scene: SceneGraph) -> MCTSState:
        """Construct from a full SceneGraph.

        Args:
            scene: Source SceneGraph.

        Returns:
            MCTSState with object positions and colours.
        """
        return cls(
            objs={
                oid: (o.color, o.centroid[0], o.centroid[1])
                for oid, o in scene.objects.items()
            },
            bg_color=scene.background_color,
        )

    def copy(self) -> MCTSState:
        """Shallow copy — safe because tuple values are immutable.

        Returns:
            New MCTSState with an independent objs dict.
        """
        return MCTSState(objs=dict(self.objs), bg_color=self.bg_color)

    def object_count(self) -> int:
        """Return number of objects in this state."""
        return len(self.objs)

    def colors(self) -> set:
        """Return set of all object colour indices."""
        return {c for c, _, _ in self.objs.values()}


# ---------------------------------------------------------------------------
# Click action wrapper
# ---------------------------------------------------------------------------


@dataclass
class ClickAction:
    """A click action targeting a specific object or coordinate.

    Wraps a base GameAction (e.g. ACTION5/6/7) with (x, y) coordinates
    and optionally the obj_id being targeted. Used inside the MCTS tree
    so that "click object 3" and "click object 7" are distinct branches.

    Attributes:
        base_action: The underlying GameAction (ACTION5, ACTION6, etc.).
        x: Grid x-coordinate to click.
        y: Grid y-coordinate to click.
        target_obj_id: Object being targeted (for rule matching).
        name: Unique string key for tree child dict.
    """

    base_action: object
    x: int
    y: int
    target_obj_id: Optional[int] = None

    @property
    def name(self) -> str:
        """Unique key combining action type and target object."""
        base_name = (
            self.base_action.name
            if hasattr(self.base_action, "name")
            else str(self.base_action)
        )
        if self.target_obj_id is not None:
            return f"{base_name}_obj{self.target_obj_id}"
        return f"{base_name}_{self.x}_{self.y}"

    @property
    def is_click(self) -> bool:
        """Always True for ClickAction."""
        return True


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def apply_rules(
    state: MCTSState, action: object, rules: list
) -> MCTSState:
    """Apply rule-engine predictions to produce the next lightweight state.

    Operates directly on MCTSState — no SceneGraph overhead. Used for
    both tree expansion and rollout steps.

    For ClickAction instances, rules are preferentially applied to the
    targeted object (if target_obj_id is set). For regular actions, all
    matching objects are affected.

    Args:
        state: Current lightweight state.
        action: Action being simulated (GameAction or ClickAction).
        rules: High-confidence Rule objects from RuleEngine.

    Returns:
        New MCTSState with all predicted changes applied.
    """
    from ..world_model.rule_engine import RuleEngine

    # Resolve the base action type for rule matching.
    is_click = isinstance(action, ClickAction)
    base = action.base_action if is_click else action
    action_type = RuleEngine._action_type(base)
    click_target = action.target_obj_id if is_click else None

    new_objs = dict(state.objs)

    for rule in rules:
        if rule.action_type != action_type or rule.confidence < 0.35:
            continue

        # Iterate over a snapshot so deletions inside don't skip entries.
        for oid, (color, cy, cx) in list(new_objs.items()):
            if oid not in new_objs:
                continue  # already removed by a previous rule

            # For click actions, only affect the targeted object.
            if click_target is not None and oid != click_target:
                continue

            if rule.target_color is not None and color != rule.target_color:
                continue

            prop = rule.target_property
            eff = rule.effect

            if prop == 'color':
                new_color = eff.get('new_value', color)
                new_objs[oid] = (new_color, cy, cx)

            elif prop == 'existence' and eff.get('change_type') == 'disappear':
                del new_objs[oid]

            elif prop == 'centroid':
                new_pos = eff.get('new_value')
                if new_pos:
                    new_objs[oid] = (color, float(new_pos[0]), float(new_pos[1]))

    return MCTSState(objs=new_objs, bg_color=state.bg_color)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        state: Lightweight state at this node.
        parent: Parent node (None for root).
        action_from_parent: GameAction that led here.
        untried_actions: Actions not yet expanded from this node.
        children: Expanded children keyed by action name string.
        visits: Simulation visits through this node.
        total_reward: Cumulative reward from all rollouts.
    """

    state: MCTSState
    parent: Optional[MCTSNode]
    action_from_parent: Optional[object]
    untried_actions: list
    children: dict = field(default_factory=dict)  # action_name → MCTSNode
    visits: int = 0
    total_reward: float = 0.0

    @property
    def q_value(self) -> float:
        """Mean reward per visit (0.0 for unvisited nodes)."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def uct(self, c: float) -> float:
        """UCT selection score.

        Args:
            c: Exploration constant (sqrt(2) ≈ 1.41 is standard).

        Returns:
            Score; +inf for unvisited nodes to guarantee exploration.
        """
        if self.visits == 0 or self.parent is None or self.parent.visits == 0:
            return float('inf')
        return self.q_value + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, c: float) -> MCTSNode:
        """Child with highest UCT score.

        Args:
            c: Exploration constant.

        Returns:
            Best child node.
        """
        return max(self.children.values(), key=lambda n: n.uct(c))

    def is_fully_expanded(self) -> bool:
        """True when all available actions have been tried from this node."""
        return len(self.untried_actions) == 0


# ---------------------------------------------------------------------------
# Search tree
# ---------------------------------------------------------------------------


class MCTSTree:
    """MCTS search tree for one goal hypothesis.

    Runs Selection → Expansion → Rollout → Backpropagation for
    n_simulations iterations, then recommends the best action via
    visit counts (more robust than raw Q-value under noisy rollouts).

    Attributes:
        root: Root node representing the current real state.
        goal_fn: Goal predicate on MCTSState.
        rules: High-confidence rules for fast state simulation.
        prior: HierarchicalPrior for expansion ordering and rollout sampling.
        available_actions: Non-click GameActions to consider.
        exploration_c: UCT exploration constant.
        max_rollout_depth: Maximum simulated steps per rollout.
    """

    def __init__(
        self,
        root_state: MCTSState,
        goal_fn: Callable[[MCTSState], bool],
        rules: list,
        prior: object,
        available_actions: list,
        exploration_c: float = 1.41,
        max_rollout_depth: int = 12,
    ) -> None:
        self.root = MCTSNode(
            state=root_state,
            parent=None,
            action_from_parent=None,
            untried_actions=list(available_actions),
        )
        self.goal_fn = goal_fn
        self.rules = rules
        self.prior = prior
        self.available_actions = list(available_actions)
        self.exploration_c = exploration_c
        self.max_rollout_depth = max_rollout_depth

    def run(self, n_simulations: int) -> None:
        """Execute n_simulations MCTS iterations.

        Args:
            n_simulations: Number of Selection→Expansion→Rollout→Backprop cycles.
        """
        for _ in range(n_simulations):
            node = self._select(self.root)
            child = self._expand(node)
            reward = self._rollout(child.state)
            self._backprop(child, reward)

    def best_action(self) -> Optional[object]:
        """Recommend action with the highest visit count from root.

        Visit count is used rather than Q-value to average out
        rollout variance.

        Returns:
            GameAction or None if the tree has no children yet.
        """
        if not self.root.children:
            return None
        best_name = max(
            self.root.children,
            key=lambda k: self.root.children[k].visits,
        )
        for action in self.available_actions:
            if hasattr(action, 'name') and action.name == best_name:
                return action
        return None

    def root_q_value(self) -> float:
        """Best child Q-value from root, used for ensemble comparison.

        Returns:
            Highest Q-value among root children; 0.0 if none.
        """
        if not self.root.children:
            return 0.0
        return max(c.q_value for c in self.root.children.values())

    # ------------------------------------------------------------------
    # Four MCTS phases (private)
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend via UCT until a non-fully-expanded or terminal node.

        Args:
            node: Starting node.

        Returns:
            Leaf or not-fully-expanded node.
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Create one new child by trying the highest-weighted untried action.

        Ordering by prior weight means the tree explores promising directions
        before exhausting its untried-action budget.

        Args:
            node: Node to expand.

        Returns:
            Newly created child, or node itself if already terminal.
        """
        if not node.untried_actions:
            return node

        # Pick the untried action with the highest prior weight.
        action = max(
            node.untried_actions,
            key=lambda a: self.prior.action_weight(a, node.state),
        )
        node.untried_actions.remove(action)

        next_state = apply_rules(node.state, action, self.rules)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action_from_parent=action,
            untried_actions=list(self.available_actions),
        )
        key = action.name if hasattr(action, 'name') else str(action)
        node.children[key] = child
        return child

    def _rollout(self, state: MCTSState) -> float:
        """Fast prior-guided playout.

        No node creation during rollout — states are evolved directly
        via apply_rules() for speed. The prior's spatial bias guides
        action sampling toward the goal even with zero learned rules.

        Args:
            state: Starting rollout state.

        Returns:
            Reward in [0, 1.5].
            Goal reached   → 1.0 + 0.5/(depth+1)   (bonus for short paths)
            No goal        → proximity_reward        (0–0.4 from prior)
        """
        current = state.copy()
        for depth in range(self.max_rollout_depth):
            if self.goal_fn(current):
                return 1.0 + 0.5 / (depth + 1)
            action = self.prior.sample_action(current, self.available_actions)
            current = apply_rules(current, action, self.rules)

        # Rollout exhausted — partial credit for proximity to goal.
        return self.prior.proximity_reward(current)

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        """Update visit counts and cumulative rewards up the tree.

        Args:
            node: Leaf node where rollout started.
            reward: Reward to propagate.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
