"""State predictor: applies rule-engine predictions to simulate next state."""

import copy
from typing import Optional

from .rule_engine import RuleEngine
from .state_delta import StateDelta
from ..perception.scene_graph import SceneGraph


class Predictor:
    """Simulates environment transitions using induced rules.

    Attributes:
        rule_engine: Source of causal rules for predictions.
        last_predictions: Cached predictions from the last call to simulate/predict.
    """

    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine
        self.last_predictions: list = []

    def predict(self, action: object, current_scene: SceneGraph) -> list:
        """Return predicted changes without modifying any scene.

        Args:
            action: Proposed action.
            current_scene: Current SceneGraph.

        Returns:
            List of prediction dicts.
        """
        self.last_predictions = self.rule_engine.predict(action, current_scene)
        return self.last_predictions

    def simulate(self, scene: SceneGraph, action: object) -> SceneGraph:
        """Return a synthetic SceneGraph after applying predicted effects.

        Deep-copies the input scene to avoid mutating the original.

        Args:
            scene: Current SceneGraph to simulate from.
            action: Action to simulate.

        Returns:
            New SceneGraph with predicted changes applied.
        """
        predictions = self.predict(action, scene)
        new_scene = copy.deepcopy(scene)

        for pred in sorted(predictions, key=lambda p: -p['confidence']):
            obj_id = pred['obj_id']
            prop = pred['property']
            effect = pred['predicted_effect']

            if obj_id not in new_scene.objects:
                continue

            obj = new_scene.objects[obj_id]

            if prop == 'color':
                new_val = effect.get('new_value')
                if new_val is not None:
                    obj.color = new_val
            elif prop == 'existence' and effect.get('change_type') == 'disappear':
                del new_scene.objects[obj_id]
            elif prop == 'centroid':
                new_val = effect.get('new_value')
                if new_val is not None:
                    obj.centroid = new_val

        state_repr = str(sorted(
            (oid, o.color, o.centroid)
            for oid, o in new_scene.objects.items()
        ))
        new_scene.frame_hash = hash(state_repr)

        return new_scene

    def evaluate_predictions(self, actual_delta: StateDelta) -> bool:
        """Check whether the last prediction was directionally correct.

        Args:
            actual_delta: Observed StateDelta to compare against.

        Returns:
            True if prediction was at least partially correct.
        """
        if not self.last_predictions:
            return actual_delta.is_noop

        if actual_delta.is_noop:
            return False

        actual_changed_ids = {d.obj_id for d in actual_delta.object_deltas}
        actual_changed_ids |= set(actual_delta.objects_disappeared)
        actual_changed_ids |= {o.obj_id for o in actual_delta.objects_appeared}

        predicted_ids = {p['obj_id'] for p in self.last_predictions}
        return bool(predicted_ids & actual_changed_ids)
