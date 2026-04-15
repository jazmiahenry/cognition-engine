"""Goal hypothesis generation and prioritisation."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..perception.scene_graph import SceneGraph


class GoalInference:
    """Maintains and updates a ranked set of goal hypotheses.

    Each hypothesis is a dict with keys: type, description, priority,
    and optional type-specific fields (e.g. target_color).

    Attributes:
        hypotheses: Current ranked list of hypothesis dicts.
    """

    def __init__(self) -> None:
        self.hypotheses: List[Dict] = []

    def generate_hypotheses(
        self,
        scene: SceneGraph,
        observations: list,
    ) -> List[Dict]:
        """Generate and rank goal hypotheses from the current scene.

        Args:
            scene: Current SceneGraph.
            observations: Full observation history from RuleEngine.

        Returns:
            Sorted list of hypothesis dicts, highest priority first.
        """
        hypotheses = []

        hypotheses.append({
            'type': 'click_all',
            'priority': 0.4,
            'description': 'Interact with every distinct object',
        })

        colors = {o.color for o in scene.objects.values()}
        if len(colors) > 1:
            for target_color in colors:
                hypotheses.append({
                    'type': 'uniform_color',
                    'target_color': target_color,
                    'priority': 0.3,
                    'description': f'Make all objects colour {target_color}',
                })

        if scene.objects:
            hypotheses.append({
                'type': 'clear_all',
                'priority': 0.25,
                'description': 'Remove all objects from the board',
            })

        hypotheses.append({
            'type': 'spatial_target',
            'priority': 0.2,
            'description': 'Move objects to match a target arrangement',
        })

        hypotheses.append({
            'type': 'maximize_change',
            'priority': 0.15,
            'description': 'Maximise cumulative state changes',
        })

        self.hypotheses = sorted(hypotheses, key=lambda h: -h['priority'])
        return self.hypotheses

    def update_priorities(
        self,
        last_action_result: object,
        world_model_rules: list,
    ) -> None:
        """Adjust hypothesis priorities based on observed rules.

        Args:
            last_action_result: Most recent action (reserved for future use).
            world_model_rules: Rules from RuleEngine.get_high_confidence_rules().
        """
        for h in self.hypotheses:
            for rule in world_model_rules:
                if (rule.target_property == 'existence'
                        and rule.effect.get('change_type') == 'disappear'
                        and h['type'] == 'clear_all'):
                    h['priority'] = min(h['priority'] + 0.15, 1.0)

                if (rule.target_property == 'color'
                        and h['type'] == 'uniform_color'):
                    h['priority'] = min(h['priority'] + 0.15, 1.0)

        self.hypotheses.sort(key=lambda h: -h['priority'])

    def top_hypothesis(self) -> Optional[Dict]:
        """Return the highest-priority hypothesis, or None.

        Returns:
            Top hypothesis dict or None if empty.
        """
        return self.hypotheses[0] if self.hypotheses else None
