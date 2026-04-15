"""Structured diffing between consecutive SceneGraphs."""

from dataclasses import dataclass, field
from typing import Optional

from ..perception.scene_graph import SceneGraph, Object


@dataclass
class ObjectDelta:
    """Property-level changes for a single persistent object.

    Attributes:
        obj_id: Persistent ID of the changed object.
        property_changes: Maps property name to (old_value, new_value).
        appeared: True if this object did not exist in the prior frame.
        disappeared: True if this object no longer exists after the action.
    """

    obj_id: int
    property_changes: dict
    appeared: bool = False
    disappeared: bool = False


@dataclass
class StateDelta:
    """Structured diff between two scene graphs caused by one action.

    Attributes:
        action_taken: The action that produced this transition.
        object_deltas: Per-object change records.
        objects_appeared: New Object instances not present before.
        objects_disappeared: IDs of objects that vanished.
        global_changes: Frame-level changes (e.g. object count).
        is_noop: True if the action produced no observable change.
    """

    action_taken: object
    object_deltas: list
    objects_appeared: list
    objects_disappeared: list
    global_changes: dict
    is_noop: bool

    def summary(self) -> str:
        """Return a human-readable one-line description of what changed.

        Returns:
            String summarising property changes, appearances, and disappearances.
        """
        if self.is_noop:
            return "no change"

        parts = []
        for d in self.object_deltas:
            for prop, (old, new) in d.property_changes.items():
                parts.append(f"obj_{d.obj_id}.{prop}: {old!r} -> {new!r}")
        if self.objects_appeared:
            parts.append(f"appeared: {[o.obj_id for o in self.objects_appeared]}")
        if self.objects_disappeared:
            parts.append(f"disappeared: {self.objects_disappeared}")

        return "; ".join(parts) if parts else "pixel-level changes only"


def compute_delta(
    before: SceneGraph,
    after: SceneGraph,
    action: object,
) -> StateDelta:
    """Compute a structured diff between two scene graphs.

    Args:
        before: SceneGraph from before the action.
        after: SceneGraph from after the action.
        action: The action taken between the two frames.

    Returns:
        StateDelta describing all observable changes.
    """
    before_ids = set(before.objects.keys())
    after_ids = set(after.objects.keys())

    disappeared_ids = list(before_ids - after_ids)
    appeared_objs = [after.objects[oid] for oid in after_ids - before_ids]
    object_deltas = []

    for oid in before_ids & after_ids:
        obj_b = before.objects[oid]
        obj_a = after.objects[oid]
        changes = {}

        if obj_b.color != obj_a.color:
            changes['color'] = (obj_b.color, obj_a.color)

        cy_diff = abs(obj_b.centroid[0] - obj_a.centroid[0])
        cx_diff = abs(obj_b.centroid[1] - obj_a.centroid[1])
        if cy_diff > 1 or cx_diff > 1:
            changes['centroid'] = (obj_b.centroid, obj_a.centroid)

        if abs(obj_b.area - obj_a.area) > 2:
            changes['area'] = (obj_b.area, obj_a.area)

        if changes:
            object_deltas.append(ObjectDelta(obj_id=oid, property_changes=changes))

    is_noop = (
        not object_deltas
        and not appeared_objs
        and not disappeared_ids
        and before.frame_hash == after.frame_hash
    )

    return StateDelta(
        action_taken=action,
        object_deltas=object_deltas,
        objects_appeared=appeared_objs,
        objects_disappeared=disappeared_ids,
        global_changes={
            'object_count': (len(before.objects), len(after.objects))
        },
        is_noop=is_noop,
    )
