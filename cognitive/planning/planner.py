"""BFS planner over simulated world-model states.

Click actions (ACTION6 / is_complex()) are excluded from BFS search —
the coordinate space is too large to enumerate. The Explorer handles
click-based goals through direct object targeting.
"""

from collections import deque
from typing import Callable, Optional

from ..world_model.rule_engine import RuleEngine
from ..world_model.predictor import Predictor
from ..perception.scene_graph import SceneGraph
from .explorer import _is_click


class Planner:
    """BFS planner using the induced world model for look-ahead.

    Attributes:
        rule_engine: Source of causal rules used by the Predictor.
    """

    _DEFAULT_MAX_DEPTH = 15
    _MIN_RULES_TO_PLAN = 1

    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine

    def plan(
        self,
        current_scene: SceneGraph,
        goal_hypothesis: dict,
        available_actions: list,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ) -> list:
        """Find a short action sequence satisfying goal_hypothesis via BFS.

        Args:
            current_scene: Start state.
            goal_hypothesis: Hypothesis dict from GoalInference.
            available_actions: All actions the environment accepts.
            max_depth: Maximum plan length to search.

        Returns:
            List of actions, or [] if no plan found.
        """
        if len(self.rule_engine.get_high_confidence_rules()) < self._MIN_RULES_TO_PLAN:
            return []

        goal_fn = self._build_goal_fn(goal_hypothesis)
        if goal_fn(current_scene):
            return []

        predictor = Predictor(self.rule_engine)
        search_actions = [a for a in available_actions if not _is_click(a)]

        queue: deque = deque([(current_scene, [])])
        visited: set = {current_scene.frame_hash}

        while queue:
            scene, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for action in search_actions:
                next_scene = predictor.simulate(scene, action)

                if next_scene.frame_hash in visited:
                    continue
                visited.add(next_scene.frame_hash)

                new_path = path + [action]
                if goal_fn(next_scene):
                    return new_path

                queue.append((next_scene, new_path))

        return []

    def _build_goal_fn(self, goal_hypothesis: dict) -> Callable[[SceneGraph], bool]:
        """Construct a boolean goal-test function from a hypothesis dict.

        Args:
            goal_hypothesis: Hypothesis dict with at minimum a 'type' key.

        Returns:
            A callable (SceneGraph) -> bool.
        """
        goal_type = goal_hypothesis.get('type', '')

        if goal_type == 'clear_all':
            return lambda scene: len(scene.objects) == 0

        if goal_type == 'uniform_color':
            target = goal_hypothesis.get('target_color')
            return lambda scene: (
                bool(scene.objects)
                and all(o.color == target for o in scene.objects.values())
            )

        # Default fallback.
        return lambda scene: len(scene.objects) == 0
