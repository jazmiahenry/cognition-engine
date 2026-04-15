"""Cognitive agent for the ARC-AGI-3 competition.

Implements a four-stage cognitive cascade:
  Stage 1 — Perception:      segment_frame + ObjectTracker
  Stage 2 — World Model:     RuleEngine + Predictor
  Stage 3 — Planning:        GoalInference + Explorer + SimulationEnsemble (MCTS)
  Stage 4 — Metacognition:   MetacognitiveMonitor

Stage 3 operates in two phases per level:
  Diagnostic  (actions 0…EXPLORATION_BUDGET-1): Explorer gathers observations
              to seed the RuleEngine and HierarchicalPrior.
  Exploitation (actions EXPLORATION_BUDGET+): SimulationEnsemble runs N MCTS
              trees × K sims and picks the best action by root Q-value.

Extends the competition's Agent base class. The harness calls
choose_action() once per turn and is_done() to decide when to stop.

No external API calls are made — everything runs locally.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from cognitive.perception import segment_frame, ObjectTracker
from cognitive.world_model import compute_delta, RuleEngine, Predictor
from cognitive.planning import (
    GoalInference,
    Explorer,
    HierarchicalPrior,
    SimulationEnsemble,
    ClickAction,
)
from cognitive.metacognition import MetacognitiveMonitor

logger = logging.getLogger(__name__)

# Actions per level spent in diagnostic exploration before switching to MCTS.
_EXPLORATION_BUDGET: int = 10
# Number of MCTS trees in the ensemble (one per top hypothesis).
_ENSEMBLE_N_TREES: int = 4
# Simulations per MCTS tree per action.
_ENSEMBLE_N_SIMS: int = 200
# If MCTS root Q-value is below this, fall back to exploration.
_ADAPTIVE_Q_THRESHOLD: float = 0.05


class CognitiveAgent(Agent):
    """Four-stage cognitive agent for ARC-AGI-3.

    Registered as "cognitiveagent" in AVAILABLE_AGENTS.
    Run with: uv run main.py --agent=cognitiveagent --game=<game_id>
    """

    MAX_ACTIONS: int = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Stage 1 — Perception
        self.tracker = ObjectTracker()

        # Stage 2 — World Model
        self.rule_engine = RuleEngine()
        self.predictor = Predictor(self.rule_engine)

        # Stage 3 — Planning
        self.goal_inference = GoalInference()
        self.explorer = Explorer()
        self.prior = HierarchicalPrior()
        self.ensemble = SimulationEnsemble(self.prior)

        # Stage 4 — Metacognition
        self.monitor = MetacognitiveMonitor()

        # Per-episode state
        self._prev_scene = None
        self._last_action_taken: Optional[GameAction] = None
        self._levels_completed_internal: int = 0
        self._level_action_count: int = 0
        self._reperceive_min_size: int = 2

    # ------------------------------------------------------------------
    # Competition API
    # ------------------------------------------------------------------

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Return True when the agent has won the game.

        Args:
            frames: Full frame history.
            latest_frame: Most recent frame.

        Returns:
            True on WIN state.
        """
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Select the next action using the four-stage cognitive cascade.

        The harness calls this once per turn. Returns a GameAction with
        reasoning attached for tracing.

        Args:
            frames: Full frame history (oldest first).
            latest_frame: Most recent FrameData observation.

        Returns:
            A GameAction ready to execute.
        """
        # States that require a RESET before anything else.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "resetting to start/restart game"
            return action

        # Full-environment reset signal (e.g. level skip by the server).
        # Ignore full_reset on the very first observation after our own RESET —
        # it just means the game started fresh, not that a mid-game restart happened.
        if (getattr(latest_frame, 'full_reset', False)
                and self._level_action_count > 0):
            self._on_full_reset()
            action = GameAction.RESET
            action.reasoning = "full reset received from server"
            return action

        # Detect level transitions (levels_completed incremented).
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self._levels_completed_internal:
            self._on_level_complete(current_levels)

        # Extract the current 2D grid (last frame in the observation list).
        raw_grid = latest_frame.frame[-1] if latest_frame.frame else []
        frame = np.array(raw_grid, dtype=np.int32)

        # ------------------------------------------------------------------
        # Stage 1: Perception
        # ------------------------------------------------------------------
        scene = segment_frame(frame, min_size=self._reperceive_min_size)
        scene = self.tracker.update(scene)

        # ------------------------------------------------------------------
        # Stage 2: World Model update
        # ------------------------------------------------------------------
        if self._prev_scene is not None and self._last_action_taken is not None:
            delta = compute_delta(self._prev_scene, scene, self._last_action_taken)
            self.rule_engine.observe(
                self._last_action_taken, self._prev_scene, scene, delta
            )

            # Sync prior with latest rules and update player position.
            self.prior.update_rules(self.rule_engine.get_high_confidence_rules())
            self.prior.update_player(delta, scene)

            novelty = (
                'noop' if delta.is_noop
                else 'novel' if scene.frame_hash != self._prev_scene.frame_hash
                else 'repeat'
            )
            self.monitor.record_action_result(novelty)

            if self.predictor.last_predictions:
                correct = self.predictor.evaluate_predictions(delta)
                self.monitor.record_prediction(correct)

            logger.debug(
                "t=%d  delta=%s  rules=%d  player_id=%s  status=%s",
                self._level_action_count,
                delta.summary(),
                len(self.rule_engine.rules),
                self.prior.player_id,
                self.monitor.get_status(),
            )

        # ------------------------------------------------------------------
        # Stage 4: Metacognitive overrides
        # ------------------------------------------------------------------
        if self.monitor.should_reset_world_model():
            logger.warning(
                "Override: resetting world model (accuracy=%.2f)",
                self.monitor.prediction_accuracy,
            )
            self.rule_engine.full_reset()
            self.prior.rules = []

        if self.monitor.should_reperceive():
            logger.info("Override: re-perceiving with min_size=1")
            self._reperceive_min_size = 1
            scene = segment_frame(frame, min_size=1)
            scene = self.tracker.update(scene)

        if self.monitor.should_change_strategy():
            logger.info("Override: resetting explorer to advance to next scan region")
            # Reset tested_actions so new object positions get probed;
            # _grid_scan_coords is preserved to advance to new grid coords.
            self.explorer.reset()

        # ------------------------------------------------------------------
        # Stage 3: Action selection
        # ------------------------------------------------------------------
        # available_actions in FrameData are integer IDs, not GameAction objects.
        # Use the full GameAction enum directly (same as the random agent pattern).
        available = self._all_actions()
        action = self._select_action(scene, available)

        # Persist state for the next turn.
        self._prev_scene = scene
        self._last_action_taken = action
        self._level_action_count += 1

        return action

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_action(
        self, scene: object, available: list
    ) -> GameAction:
        """Choose the next action using Explorer or MCTS ensemble.

        Phase 1 — Diagnostic (actions 0…EXPLORATION_BUDGET-1):
            Explorer gathers observations to seed RuleEngine + Prior.

        Phase 2 — MCTS exploitation (actions EXPLORATION_BUDGET+):
            SimulationEnsemble runs N trees × K sims. If the best root
            Q-value is below _ADAPTIVE_Q_THRESHOLD, adaptively falls
            back to Explorer to gather more data instead of wasting
            actions on blind MCTS.

        Args:
            scene: Current SceneGraph from Stage 1.
            available: Non-RESET GameAction objects for this turn.

        Returns:
            A configured GameAction with reasoning attached.
        """
        # Refresh goal hypotheses regardless of phase so priorities update.
        high_conf_rules = self.rule_engine.get_high_confidence_rules()
        goals = self.goal_inference.generate_hypotheses(
            scene, self.rule_engine.observations
        )
        self.goal_inference.update_priorities(None, high_conf_rules)

        # ------------------------------------------------------------------
        # Phase 1: Diagnostic exploration
        # ------------------------------------------------------------------
        if self._level_action_count < _EXPLORATION_BUDGET:
            return self._do_explore(scene, available)

        # ------------------------------------------------------------------
        # Phase 2: MCTS ensemble with adaptive fallback
        # ------------------------------------------------------------------
        best_action, winning_hypothesis, best_q = self.ensemble.run(
            scene=scene,
            goal_hypotheses=goals,
            available_actions=available,
            n_trees=_ENSEMBLE_N_TREES,
            n_sims=_ENSEMBLE_N_SIMS,
        )

        # Adaptive fallback: if MCTS has no traction, explore instead.
        if best_action is None or best_q < _ADAPTIVE_Q_THRESHOLD:
            logger.info(
                "Adaptive fallback: Q=%.4f < %.2f, exploring instead",
                best_q, _ADAPTIVE_Q_THRESHOLD,
            )
            return self._do_explore(scene, available)

        # Translate ClickAction → real GameAction with coordinates.
        action = self._resolve_action(best_action)

        logger.info(
            "MCTS[%d]  action=%s  Q=%.3f  hypothesis=%s  rules=%d",
            self._level_action_count,
            getattr(action, 'name', action),
            best_q,
            winning_hypothesis.get('type') if winning_hypothesis else 'none',
            len(high_conf_rules),
        )
        action.reasoning = {
            "stage": "mcts",
            "q_value": round(best_q, 4),
            "hypothesis": winning_hypothesis.get('type') if winning_hypothesis else None,
            "hypothesis_description": (
                winning_hypothesis.get('description') if winning_hypothesis else None
            ),
            "rules_available": len(high_conf_rules),
            "player_id": self.prior.player_id,
            "goal_pos": self.prior.goal_pos,
        }
        return action

    def _do_explore(self, scene: object, available: list) -> GameAction:
        """Run one Explorer step and return a configured GameAction.

        Args:
            scene: Current SceneGraph.
            available: Available GameAction list.

        Returns:
            A GameAction with reasoning attached.
        """
        raw_action, data, reason = self.explorer.select_exploration_action(
            scene, available, self.rule_engine
        )
        logger.debug(
            "Explore[%d]: %s", self._level_action_count, reason,
        )
        if data is not None:
            action = GameAction.ACTION6
            action.set_data({'x': data['x'], 'y': data['y']})
            action.reasoning = {
                "stage": "explore",
                "reason": reason,
                "coords": data,
            }
        else:
            action = raw_action
            action.reasoning = {"stage": "explore", "reason": reason}
        return action

    def _resolve_action(self, action: object) -> GameAction:
        """Convert a ClickAction (or pass through a GameAction) to a real action.

        Args:
            action: Either a ClickAction or a regular GameAction.

        Returns:
            A GameAction ready to submit to the environment.
        """
        if isinstance(action, ClickAction):
            game_action = action.base_action
            game_action.set_data({'x': action.x, 'y': action.y})
            game_action.reasoning = {
                "click_target_obj": action.target_obj_id,
                "click_coords": (action.x, action.y),
            }
            return game_action
        return action

    def _all_actions(self) -> list:
        """Return all GameAction values except RESET as a fallback list.

        Returns:
            List of GameAction constants.
        """
        return [a for a in GameAction if a is not GameAction.RESET]

    def _on_level_complete(self, new_level_count: int) -> None:
        """Handle a level transition — reset per-level state.

        World model rules are preserved; they often transfer to harder
        levels within the same game.

        Args:
            new_level_count: Updated levels_completed value.
        """
        logger.info(
            "Level complete: %d → %d  (rules=%d)",
            self._levels_completed_internal,
            new_level_count,
            len(self.rule_engine.rules),
        )
        self._levels_completed_internal = new_level_count
        self._level_action_count = 0
        self._reperceive_min_size = 2

        # Reset prior's per-episode state but keep rules — they often transfer.
        self.prior.player_id = None
        self.prior.player_pos = None
        self.prior.goal_pos = None

        self.explorer.reset()
        self.tracker.reset()
        self.monitor.reset()
        self.rule_engine.reset()  # clears observations, keeps rules

    def _on_full_reset(self) -> None:
        """Handle a full server-side reset — clear everything.

        Called when the server signals a complete game restart.
        """
        logger.info("Full reset: clearing all cognitive state")
        self._levels_completed_internal = 0
        self._level_action_count = 0
        self._prev_scene = None
        self._last_action_taken = None
        self._reperceive_min_size = 2

        # Clear prior completely on full reset.
        self.prior.player_id = None
        self.prior.player_pos = None
        self.prior.goal_pos = None
        self.prior.rules = []

        self.tracker.reset()
        self.explorer.reset()
        self.monitor.reset()
        self.rule_engine.full_reset()
