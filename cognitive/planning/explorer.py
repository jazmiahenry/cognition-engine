"""Information-seeking exploration strategy.

Selects actions that maximise information gain during the exploration
phase. Works directly with GameAction enum objects — uses the
is_complex() method (where available) to distinguish click actions.
"""

import random
from typing import Optional

from ..perception.scene_graph import SceneGraph


class Explorer:
    """Selects information-seeking actions during the exploration phase.

    Priority order:
      1. Try each simple action (directional/undo) exactly once.
      2. Click each distinct object once (centroid click).
      3. Click the object the rule engine knows least about.
      4. Systematic grid scan: click a 8×8 sample of grid coordinates
         that haven't been tried yet.
      5. Last resort: random action.

    Attributes:
        tested_actions: Set of (type_key, target_id) pairs already tried.
        _grid_scan_coords: Iterator over unvisited scan-grid coordinates.
    """

    # Sample the 64×64 grid at every GRID_STEP pixels in each dimension.
    # 8 → 8×8 = 64 candidate click coordinates.
    _GRID_STEP: int = 8

    def __init__(self) -> None:
        self.tested_actions: set = set()
        self._grid_scan_coords: list = self._build_grid_coords()

    def select_exploration_action(
        self,
        scene: SceneGraph,
        available_actions: list,
        world_model=None,
    ) -> tuple:
        """Select the most informative available action.

        Args:
            scene: Current SceneGraph.
            available_actions: Actions the environment currently accepts.
            world_model: Optional RuleEngine for rule-count-based priority.

        Returns:
            (action, data, reason) — action is None for click actions
            (caller sets up ACTION6 with data={'x':int,'y':int}),
            or (action, None, reason) for simple actions.
        """
        simple_actions = [a for a in available_actions if not _is_click(a)]
        click_actions = [a for a in available_actions if _is_click(a)]

        # Phase 1: try each simple action exactly once.
        for action in simple_actions:
            key = _action_key(action)
            if ('dir', key) not in self.tested_actions:
                self.tested_actions.add(('dir', key))
                return action, None, f"explore:dir_{key}"

        # Phase 2: click each detected object once at its centroid.
        if click_actions:
            for obj in scene.objects.values():
                if ('click', obj.obj_id) not in self.tested_actions:
                    self.tested_actions.add(('click', obj.obj_id))
                    cy, cx = int(obj.centroid[0]), int(obj.centroid[1])
                    return click_actions[0], {'x': cx, 'y': cy}, f"explore:click_obj_{obj.obj_id}"

            # Phase 3: click the object the rule engine knows least about.
            if world_model and scene.objects:
                rule_counts = {
                    obj.obj_id: sum(
                        1 for r in world_model.rules
                        if r.target_obj_id == obj.obj_id
                    )
                    for obj in scene.objects.values()
                }
                least = min(rule_counts, key=rule_counts.get)
                obj = scene.objects[least]
                cy, cx = int(obj.centroid[0]), int(obj.centroid[1])
                return click_actions[0], {'x': cx, 'y': cy}, f"explore:least_known_{least}"

            # Phase 4: systematic grid scan — click unvisited grid coordinates.
            # Replenish when exhausted so we keep probing on reset.
            if not self._grid_scan_coords:
                self._grid_scan_coords = self._build_grid_coords()
            gx, gy = self._grid_scan_coords.pop(0)
            coord_key = ('grid', gx, gy)
            self.tested_actions.add(coord_key)
            return click_actions[0], {'x': gx, 'y': gy}, f"explore:grid_{gx}_{gy}"

        # Phase 5: last resort — random simple action.
        if simple_actions:
            action = random.choice(simple_actions)
            return action, None, f"explore:random_{_action_key(action)}"

        action = random.choice(available_actions)
        return action, None, "explore:random"

    def reset(self) -> None:
        """Reset per-level exploration state.

        Preserves grid scan coordinates so a strategy-change reset
        advances through new grid positions rather than replaying old ones.
        """
        self.tested_actions = set()
        # Don't reset _grid_scan_coords — keep scanning new positions.

    def _build_grid_coords(self) -> list:
        """Build a shuffled list of (x, y) grid sample coordinates.

        Returns:
            List of (x, y) tuples sampling the 64×64 grid every GRID_STEP pixels,
            randomly shuffled so consecutive calls probe different areas.
        """
        coords = [
            (x, y)
            for y in range(self._GRID_STEP // 2, 64, self._GRID_STEP)
            for x in range(self._GRID_STEP // 2, 64, self._GRID_STEP)
        ]
        random.shuffle(coords)
        return coords


# ---------------------------------------------------------------------------
# Module-level helpers for action classification (not tied to GameAction import)
# ---------------------------------------------------------------------------

def _is_click(action: object) -> bool:
    """Return True if this action requires coordinate data (ACTION6).

    Args:
        action: Any action object.

    Returns:
        True for click/complex actions.
    """
    if hasattr(action, 'is_complex'):
        return bool(action.is_complex())
    # Fallback: numeric or string ID check.
    key = _action_key(action)
    return key in (6, '6', 'ACTION6')


def _action_key(action: object) -> object:
    """Return a stable hashable key for an action (strips coordinate data).

    Args:
        action: Any action object.

    Returns:
        Stable key (action .name, .value, or the object itself).
    """
    if isinstance(action, tuple):
        action = action[0]
    # GameAction enum: prefer .name ("ACTION1" etc.) for readability.
    if hasattr(action, 'name') and hasattr(action, 'is_simple'):
        return action.name
    if hasattr(action, 'value'):
        return action.value
    return action
