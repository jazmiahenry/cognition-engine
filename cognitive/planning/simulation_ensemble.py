"""Simulation ensemble: N MCTS trees × K sims, one tree per goal hypothesis.

The winning tree is chosen by root Q-value — the hypothesis whose best
child has the highest mean reward after K simulations. This gives the
ensemble a joint objective: find the hypothesis-action pair that is
most likely to make real progress toward *some* plausible goal, without
committing to one interpretation of the environment upfront.

Design rationale
----------------
- One MCTSTree per hypothesis means each tree's rollout reward function
  is internally consistent (goal_fn matches the spatial/object bias in
  the prior).
- Winner-by-Q-value is used rather than winner-by-visits so that a tree
  that converged fast (high Q, few visits) beats one that spread
  simulations evenly without finding a path (many visits, low Q).
- The ensemble runs trees sequentially within each real-action budget
  to avoid GIL contention; wall-clock time is dominated by LLM calls
  further up the stack so sequential simulation is fine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .mcts import ClickAction, MCTSState, MCTSTree
from .prior import HierarchicalPrior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal function factory
# ---------------------------------------------------------------------------

def _build_goal_fn(hypothesis: Dict, initial_state: MCTSState):
    """Create a goal predicate for an MCTS tree from a hypothesis dict.

    Args:
        hypothesis: Hypothesis dict from GoalInference.
        initial_state: Root state at ensemble start time.

    Returns:
        Callable[[MCTSState], bool] that returns True when the goal is met.
    """
    h_type = hypothesis.get("type", "maximize_change")

    if h_type == "clear_all":
        def _goal(state: MCTSState) -> bool:
            return state.object_count() == 0
        return _goal

    if h_type == "uniform_color":
        target_color = hypothesis.get("target_color")
        def _goal(state: MCTSState) -> bool:
            if not state.objs:
                return False
            return all(c == target_color for c, _, _ in state.objs.values())
        return _goal

    if h_type == "click_all":
        n0 = max(1, initial_state.object_count())
        threshold = n0 // 2
        def _goal(state: MCTSState) -> bool:
            return state.object_count() <= threshold
        return _goal

    if h_type == "spatial_target":
        initial_hash = hash(initial_state)
        def _goal(state: MCTSState) -> bool:
            return hash(state) != initial_hash and state.object_count() > 0
        return _goal

    # "maximize_change" and fallback.
    initial_hash = hash(initial_state)
    def _default_goal(state: MCTSState) -> bool:
        return hash(state) != initial_hash
    return _default_goal


# ---------------------------------------------------------------------------
# SimulationEnsemble
# ---------------------------------------------------------------------------


class SimulationEnsemble:
    """Runs an ensemble of MCTS trees and returns the best action.

    Each call to run() constructs fresh MCTSTrees, runs simulations, and
    selects the winner. The prior is shared across all trees so spatial
    and empirical knowledge accumulated by the agent carries over.

    Attributes:
        prior: Shared HierarchicalPrior instance.
    """

    def __init__(self, prior: HierarchicalPrior) -> None:
        self.prior = prior

    def run(
        self,
        scene: Any,
        goal_hypotheses: List[Dict],
        available_actions: List,
        n_trees: int = 4,
        n_sims: int = 300,
    ) -> Tuple[Optional[Any], Optional[Dict], float]:
        """Choose the best action by running N trees with K simulations each.

        Args:
            scene: Current SceneGraph from Stage 1 perception.
            goal_hypotheses: Ranked list from GoalInference. We use
                the top ``n_trees`` hypotheses.
            available_actions: Non-RESET actions from the game environment.
            n_trees: Number of MCTS trees to run (one per hypothesis).
            n_sims: Simulations per tree.

        Returns:
            Tuple of (best_action, winning_hypothesis, best_q_value).
            action/hypothesis are None if the scene is empty.
            best_q_value is used by the caller for adaptive exploration.
        """
        if not available_actions:
            logger.warning("SimulationEnsemble: no available actions — skipping.")
            return None, None, 0.0

        # Snapshot the current state once; all trees share the same root.
        root_state = MCTSState.from_scene(scene)

        # Build click actions from scene objects (up to 8 most salient).
        click_actions = self._build_click_actions(scene, available_actions)
        all_actions = list(available_actions) + click_actions

        # Limit to top n_trees hypotheses, fall back to a default if fewer.
        top_hypotheses = goal_hypotheses[:n_trees]
        if not top_hypotheses:
            top_hypotheses = [{"type": "maximize_change", "priority": 0.1,
                                "description": "Maximise state change (fallback)"}]

        # Retrieve high-confidence rules for simulation.
        rules = [r for r in self.prior.rules if r.confidence >= 0.35]

        best_action: Optional[Any] = None
        best_hypothesis: Optional[Dict] = None
        best_q: float = -float("inf")
        # Early-exit: if first tree gets Q=0, no rules can produce state
        # changes so remaining trees will also get Q=0. Skip them.
        zero_q_count: int = 0

        # Probe phase: run a small batch first to check if the world model
        # has any traction. If probe Q > 0, commit the remaining sims.
        _PROBE_SIMS: int = 30

        for hypothesis in top_hypotheses:
            goal_fn = _build_goal_fn(hypothesis, root_state)
            tree = MCTSTree(
                root_state=root_state,
                goal_fn=goal_fn,
                rules=rules,
                prior=self.prior,
                available_actions=all_actions,
                exploration_c=1.41,
                max_rollout_depth=10,
            )
            # Probe with a small batch first.
            tree.run(_PROBE_SIMS)
            probe_q = tree.root_q_value()
            # If the probe found signal, commit the remaining simulations.
            if probe_q > 1e-6:
                tree.run(n_sims - _PROBE_SIMS)
            q = tree.root_q_value()
            action = tree.best_action()

            logger.debug(
                "Hypothesis '%s' → action=%s  root_Q=%.4f",
                hypothesis.get("type"),
                getattr(action, "name", action),
                q,
            )

            if q > best_q and action is not None:
                best_q = q
                best_action = action
                best_hypothesis = hypothesis

            # Early exit: if first two trees both return Q≈0, the world
            # model has no traction — skip remaining trees to save time.
            if q < 1e-6:
                zero_q_count += 1
                if zero_q_count >= 2:
                    logger.debug("Early exit: %d trees with Q≈0, skipping rest", zero_q_count)
                    break

        if best_action is None:
            # All trees failed to expand — fall back to uniform random.
            import random as _rnd
            best_action = _rnd.choice(available_actions)
            best_hypothesis = top_hypotheses[0]
            best_q = 0.0
            logger.warning(
                "SimulationEnsemble: all trees produced no action; "
                "falling back to random choice: %s",
                getattr(best_action, "name", best_action),
            )

        return best_action, best_hypothesis, best_q

    def _build_click_actions(
        self, scene: Any, available_actions: List
    ) -> List[ClickAction]:
        """Build ClickAction instances targeting scene objects.

        For each complex action type (ACTION5-7) and each non-player
        object in the scene, create a ClickAction at the object's
        centroid. Limits to the 8 most salient objects to keep the
        action space tractable for MCTS.

        Args:
            scene: Current SceneGraph.
            available_actions: Base GameAction list (used to find complex actions).

        Returns:
            List of ClickAction instances.
        """
        objects = getattr(scene, "objects", {})
        if not objects:
            return []

        # Find complex (click-capable) actions from the available set.
        complex_actions = [
            a for a in available_actions
            if hasattr(a, "is_complex") and a.is_complex()
        ]
        if not complex_actions:
            return []

        # Score objects by saliency for prioritisation.
        player_id = self.prior.player_id
        scored: List[Tuple[float, int, Any]] = []
        color_counts: Dict[int, int] = {}
        for obj in objects.values():
            color_counts[obj.color] = color_counts.get(obj.color, 0) + 1
        avg_area = sum(o.area for o in objects.values()) / max(1, len(objects))

        for oid, obj in objects.items():
            if oid == player_id:
                continue
            uniqueness = 1.0 / color_counts.get(obj.color, 1)
            size_score = max(0.0, 1.0 - obj.area / (avg_area + 1))
            score = uniqueness * 0.6 + size_score * 0.4
            scored.append((score, oid, obj))

        # Take top 8 most salient objects.
        scored.sort(key=lambda t: -t[0])
        top_objects = scored[:8]

        click_actions: List[ClickAction] = []
        for _, oid, obj in top_objects:
            cy, cx = obj.centroid
            x, y = int(round(cx)), int(round(cy))
            # Clamp to grid bounds.
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            for base in complex_actions:
                click_actions.append(ClickAction(
                    base_action=base,
                    x=x,
                    y=y,
                    target_obj_id=oid,
                ))

        return click_actions
