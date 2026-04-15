"""Rule induction and storage for the world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Rule:
    """A hypothesized causal rule about the environment.

    Attributes:
        action_type: Normalized action identifier that triggers the rule.
        target_property: Which object property changes.
        target_color: If set, rule applies to objects of this color only.
        target_obj_id: Object this rule was first observed on.
        effect: Description of what happens.
        observation_count: Total times this rule has been applicable.
        confirmation_count: Times the predicted effect actually occurred.
        confidence: confirmation_count / observation_count.
    """

    action_type: object
    target_property: str
    target_color: Optional[int] = None
    target_obj_id: Optional[int] = None
    effect: dict = field(default_factory=dict)
    observation_count: int = 0
    confirmation_count: int = 0
    confidence: float = 0.0

    def update_confidence(self) -> None:
        """Recompute confidence from observation counts."""
        if self.observation_count > 0:
            self.confidence = self.confirmation_count / self.observation_count


class RuleEngine:
    """Induces and stores symbolic rules from observed state transitions.

    Attributes:
        rules: All induced Rule instances.
        observations: Full history of (action, before, after, delta) tuples.
    """

    def __init__(self) -> None:
        self.rules: List[Rule] = []
        self.observations: list = []

    def observe(
        self,
        action: object,
        before_scene: object,
        after_scene: object,
        delta: object,
    ) -> None:
        """Record an observation and update/create rules accordingly.

        Args:
            action: The action taken.
            before_scene: SceneGraph before the action.
            after_scene: SceneGraph after the action.
            delta: StateDelta from compute_delta().
        """
        self.observations.append((action, before_scene, after_scene, delta))

        if delta.is_noop:
            return

        action_type = self._action_type(action)

        for obj_delta in delta.object_deltas:
            obj = before_scene.objects.get(obj_delta.obj_id)
            obj_color = obj.color if obj else None

            for prop, (old_val, new_val) in obj_delta.property_changes.items():
                matched = False
                for rule in self.rules:
                    if (rule.action_type == action_type
                            and rule.target_property == prop
                            and rule.target_color == obj_color):
                        rule.observation_count += 1
                        predicted = rule.effect.get('new_value')
                        if predicted == new_val or rule.effect.get('change_type') == 'toggle':
                            rule.confirmation_count += 1
                        rule.update_confidence()
                        matched = True
                        break

                if not matched:
                    self.rules.append(Rule(
                        action_type=action_type,
                        target_property=prop,
                        target_color=obj_color,
                        target_obj_id=obj_delta.obj_id,
                        effect={'change_type': 'set', 'old_value': old_val,
                                'new_value': new_val},
                        observation_count=1,
                        confirmation_count=1,
                        confidence=1.0,
                    ))

        for obj in delta.objects_appeared:
            self.rules.append(Rule(
                action_type=action_type,
                target_property='existence',
                target_color=obj.color,
                effect={'change_type': 'appear'},
                observation_count=1,
                confirmation_count=1,
                confidence=1.0,
            ))

        for obj_id in delta.objects_disappeared:
            obj = before_scene.objects.get(obj_id)
            self.rules.append(Rule(
                action_type=action_type,
                target_property='existence',
                target_color=obj.color if obj else None,
                effect={'change_type': 'disappear'},
                observation_count=1,
                confirmation_count=1,
                confidence=1.0,
            ))

    def predict(self, action: object, current_scene: object) -> list:
        """Predict which objects will change and how, given an action.

        Args:
            action: Proposed next action.
            current_scene: Current SceneGraph.

        Returns:
            List of prediction dicts with obj_id, property, predicted_effect, confidence.
        """
        action_type = self._action_type(action)
        predictions = []

        for rule in self.rules:
            if rule.action_type != action_type or rule.confidence < 0.3:
                continue
            for obj in current_scene.objects.values():
                if rule.target_color is not None and obj.color != rule.target_color:
                    continue
                predictions.append({
                    'obj_id': obj.obj_id,
                    'property': rule.target_property,
                    'predicted_effect': rule.effect,
                    'confidence': rule.confidence,
                })

        return predictions

    def get_high_confidence_rules(
        self,
        min_confidence: float = 0.5,
        min_observations: int = 2,
    ) -> list:
        """Return rules confirmed enough to act on.

        Args:
            min_confidence: Minimum confidence threshold.
            min_observations: Minimum observation count.

        Returns:
            Filtered list of Rule instances.
        """
        return [
            r for r in self.rules
            if r.confidence >= min_confidence
            and r.observation_count >= min_observations
        ]

    def reset(self) -> None:
        """Clear observations while retaining rules (call between levels)."""
        self.observations = []

    def full_reset(self) -> None:
        """Clear everything — called when predictions are systematically wrong."""
        self.rules = []
        self.observations = []

    @staticmethod
    def _action_type(action: object) -> object:
        """Normalise an action to a hashable identifier.

        Handles GameAction enums (with .name), tuples, ints, and strings.

        Args:
            action: Raw action value.

        Returns:
            Hashable identifier for the action type (strips coordinate data).
        """
        if isinstance(action, tuple):
            return action[0]
        # GameAction enum: use .name ("ACTION1", "ACTION6", etc.) for stability.
        if hasattr(action, 'name') and hasattr(action, 'is_simple'):
            return action.name
        if hasattr(action, 'value'):
            return action.value
        return action
