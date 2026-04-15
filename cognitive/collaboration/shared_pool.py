"""Thread-safe shared rule pool for greedy collaborative adaptation.

Agents publish their learned rules with game metadata. When an agent
gets stuck, it queries the pool for the most compatible peer and
borrows high-confidence rules weighted by compatibility score.

Graduated agents (those that completed a level) are tagged as mentors
and prioritized as rule donors — their rules are proven to work.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Colors with known semantic meaning in ARC-AGI-3.
WALL_COLOR: int = 5       # black
PLAYER_COLORS: Set[int] = {9, 12}  # blue body, orange head
DOOR_COLOR: int = 6       # pink


@dataclass
class AgentRuleEntry:
    """Published rule set from one agent.

    Attributes:
        game_id: Which game this agent is playing.
        rules: List of Rule objects from the agent's RuleEngine.
        colors_seen: Set of object color indices in the agent's game.
        action_types_used: Set of action type strings that triggered rules.
        available_action_ids: Action IDs this game accepts (from env).
        object_count: Number of objects in the agent's current scene.
        levels_completed: How many levels the agent has passed.
        is_mentor: True if agent completed at least one level.
        confidence_sum: Sum of all rule confidences (quality signal).
    """

    game_id: str
    rules: List = field(default_factory=list)
    colors_seen: Set[int] = field(default_factory=set)
    action_types_used: Set[str] = field(default_factory=set)
    available_action_ids: Optional[List[int]] = None
    object_count: int = 0
    levels_completed: int = 0
    is_mentor: bool = False
    confidence_sum: float = 0.0


class SharedRulePool:
    """Thread-safe pool for cross-agent rule sharing.

    Each CognitiveAgent holds a reference to the same SharedRulePool
    instance (class-level singleton). Agents publish after each world
    model update and query when stuck.

    Attributes:
        _entries: game_id → AgentRuleEntry mapping.
        _lock: Threading lock for safe concurrent access.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, AgentRuleEntry] = {}
        self._lock = threading.Lock()

    def publish(
        self,
        game_id: str,
        rules: List,
        colors_seen: Set[int],
        action_types_used: Set[str],
        object_count: int,
        levels_completed: int,
        available_action_ids: Optional[List[int]] = None,
    ) -> None:
        """Publish or update an agent's rule set in the pool.

        Args:
            game_id: Agent's game identifier.
            rules: Current high-confidence rules.
            colors_seen: Object colors present in the agent's game.
            action_types_used: Action types that triggered rule changes.
            object_count: Current object count in the scene.
            levels_completed: Levels the agent has completed.
            available_action_ids: Action IDs this game accepts.
        """
        conf_sum = sum(r.confidence for r in rules)
        entry = AgentRuleEntry(
            game_id=game_id,
            rules=list(rules),
            colors_seen=set(colors_seen),
            action_types_used=set(action_types_used),
            available_action_ids=available_action_ids,
            object_count=object_count,
            levels_completed=levels_completed,
            is_mentor=(levels_completed > 0),
            confidence_sum=conf_sum,
        )
        with self._lock:
            self._entries[game_id] = entry

    # Minimum compatibility score to allow rule transfer.
    _COMPAT_THRESHOLD: float = 0.70

    def query_best_match(
        self,
        game_id: str,
        my_colors: Set[int],
        my_action_types: Set[str],
        my_object_count: int,
        my_available_action_ids: Optional[List[int]] = None,
        my_existing_rules: Optional[List] = None,
    ) -> Tuple[Optional[List], float, Optional[str]]:
        """Find the most compatible peer and return novel, applicable rules.

        Only transfers rules when compatibility ≥ 70%. Filters out:
          - Rules the receiver already has (by action_type + target_property + target_color).
          - Rules referencing action types the receiver's game doesn't support.
          - Rules referencing colors not present in the receiver's game.

        Args:
            game_id: Requesting agent's game (excluded from candidates).
            my_colors: Colors in the requesting agent's game.
            my_action_types: Action types the requesting agent has tried.
            my_object_count: Object count in the requesting agent's scene.
            my_available_action_ids: Action IDs this game accepts (from env).
            my_existing_rules: Agent's current rules for dedup checking.

        Returns:
            (rules, compatibility_score, donor_game_id) or (None, 0, None).
        """
        with self._lock:
            candidates = {
                gid: entry for gid, entry in self._entries.items()
                if gid != game_id and entry.rules
            }

        if not candidates:
            return None, 0.0, None

        best_rules: Optional[List] = None
        best_score: float = -1.0
        best_donor: Optional[str] = None

        for gid, entry in candidates.items():
            score = self._compatibility_score(
                my_colors, my_action_types, my_object_count, entry,
                my_available_action_ids,
            )
            if score > best_score:
                best_score = score
                best_rules = entry.rules
                best_donor = gid

        if best_rules is None or best_score < self._COMPAT_THRESHOLD:
            return None, 0.0, None

        # Build set of existing rule keys for dedup.
        existing_keys: Set[tuple] = set()
        if my_existing_rules:
            for r in my_existing_rules:
                existing_keys.add((r.action_type, r.target_property, r.target_color))

        # Filter: high confidence + novel + applicable to this game.
        transferable = []
        for r in best_rules:
            if r.confidence < 0.5:
                continue

            # Novelty: skip if we already have this rule.
            key = (r.action_type, r.target_property, r.target_color)
            if key in existing_keys:
                continue

            # Action applicability: skip rules for actions this game doesn't support.
            if my_available_action_ids is not None:
                rule_action = r.action_type
                if isinstance(rule_action, str):
                    # Convert "ACTION6" → 6 for matching.
                    try:
                        action_id = int(rule_action.replace("ACTION", ""))
                        if action_id not in my_available_action_ids:
                            continue
                    except ValueError:
                        pass

            # Color applicability: skip rules targeting colors not in this game.
            if r.target_color is not None and r.target_color not in my_colors:
                continue

            transferable.append(r)

        if not transferable:
            return None, 0.0, None

        logger.info(
            "Rule transfer: %s ← %s (compat=%.3f, %d novel rules, mentor=%s)",
            game_id, best_donor, best_score, len(transferable),
            candidates[best_donor].is_mentor if best_donor else False,
        )

        return transferable, best_score, best_donor

    def get_mentors(self) -> List[str]:
        """Return game_ids of agents that have completed at least one level.

        Returns:
            List of mentor game IDs.
        """
        with self._lock:
            return [gid for gid, e in self._entries.items() if e.is_mentor]

    def _compatibility_score(
        self,
        my_colors: Set[int],
        my_action_types: Set[str],
        my_object_count: int,
        entry: AgentRuleEntry,
        my_available_action_ids: Optional[List[int]] = None,
    ) -> float:
        """Score how compatible a donor's rules are with the requester.

        Games that share the same available_actions are the strongest
        candidates — they have the same mechanics. Action-type overlap
        is weighted highest to enforce this.

        Args:
            my_colors: Requester's object colors.
            my_action_types: Requester's tried action types.
            my_object_count: Requester's object count.
            entry: Donor's published entry.
            my_available_action_ids: Requester's game action IDs.

        Returns:
            Float in [0, 1]. Higher = more compatible.
        """
        # Game-type match: available_action_ids overlap — 40%.
        game_type_sim = 0.0
        if my_available_action_ids and entry.available_action_ids:
            my_set = set(my_available_action_ids)
            donor_set = set(entry.available_action_ids)
            union = my_set | donor_set
            inter = my_set & donor_set
            game_type_sim = len(inter) / max(1, len(union))

        # Color Jaccard — 20%.
        color_union = my_colors | entry.colors_seen
        color_inter = my_colors & entry.colors_seen
        color_sim = len(color_inter) / max(1, len(color_union))

        # Action type Jaccard (tried actions) — 15%.
        action_union = my_action_types | entry.action_types_used
        action_inter = my_action_types & entry.action_types_used
        action_sim = len(action_inter) / max(1, len(action_union))

        # Structural similarity — 5%.
        if my_object_count > 0 and entry.object_count > 0:
            ratio = min(my_object_count, entry.object_count) / max(
                my_object_count, entry.object_count
            )
        else:
            ratio = 0.0

        # Mentor bonus — 20%.
        mentor = 1.0 if entry.is_mentor else 0.0

        return (0.4 * game_type_sim + 0.2 * color_sim
                + 0.15 * action_sim + 0.05 * ratio + 0.2 * mentor)
