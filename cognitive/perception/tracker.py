"""Object identity tracking across frames."""

from __future__ import annotations

from typing import Optional

from .scene_graph import SceneGraph


class ObjectTracker:
    """Tracks object identity across frames using color + centroid proximity.

    Attributes:
        prev_scene: The last scene passed to update().
        next_persistent_id: Counter for fresh persistent IDs.
    """

    _MAX_MATCH_DIST = 40
    _MIN_SIZE_RATIO = 0.3

    def __init__(self) -> None:
        self.prev_scene: Optional[SceneGraph] = None
        self.next_persistent_id: int = 0
        self._prev_ids: set = set()

    def update(self, scene: SceneGraph) -> SceneGraph:
        """Match objects in scene to previous scene; reassign persistent IDs.

        Args:
            scene: Freshly segmented SceneGraph with local obj_ids.

        Returns:
            The same SceneGraph with obj_ids replaced by persistent IDs.
        """
        curr_objs = list(scene.objects.values())

        if self.prev_scene is None:
            for obj in curr_objs:
                obj.obj_id = self.next_persistent_id
                self.next_persistent_id += 1
            scene.objects = {obj.obj_id: obj for obj in curr_objs}
            self.prev_scene = scene
            self._prev_ids = set(scene.objects.keys())
            return scene

        prev_objs = list(self.prev_scene.objects.values())
        matched_prev: set = set()

        for curr in curr_objs:
            best_match = None
            best_dist = float('inf')

            for prev in prev_objs:
                if prev.obj_id in matched_prev or prev.color != curr.color:
                    continue
                dist = (
                    abs(curr.centroid[0] - prev.centroid[0])
                    + abs(curr.centroid[1] - prev.centroid[1])
                )
                size_ratio = (
                    min(curr.area, prev.area)
                    / max(curr.area, prev.area, 1)
                )
                if dist < best_dist and size_ratio >= self._MIN_SIZE_RATIO:
                    best_dist = dist
                    best_match = prev

            if best_match is not None and best_dist <= self._MAX_MATCH_DIST:
                curr.obj_id = best_match.obj_id
                matched_prev.add(best_match.obj_id)
            else:
                curr.obj_id = self.next_persistent_id
                self.next_persistent_id += 1

        scene.objects = {obj.obj_id: obj for obj in curr_objs}
        self._prev_ids = set(self.prev_scene.objects.keys())
        self.prev_scene = scene
        return scene

    def disappeared_ids(self) -> list:
        """Return persistent IDs that vanished in the latest frame.

        Returns:
            List of obj_ids present in the previous frame but not current.
        """
        if self.prev_scene is None:
            return []
        current_ids = set(self.prev_scene.objects.keys())
        return list(self._prev_ids - current_ids)

    def reset(self) -> None:
        """Reset tracker for a new level; preserves ID counter."""
        self.prev_scene = None
        self._prev_ids = set()
