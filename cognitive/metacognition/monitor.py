"""Metacognitive monitor: Stage 4 of the cognitive cascade.

Observes the performance of Stages 1–3 over a rolling window and
emits override signals when systematic failures are detected:

  should_reset_world_model()  → Stage 4 → Stage 2
  should_change_strategy()    → Stage 4 → Stage 3
  should_reperceive()         → Stage 4 → Stage 1

This lets the agent self-correct without external supervision.
"""

from collections import deque


class MetacognitiveMonitor:
    """Monitors prediction accuracy and exploration efficiency.

    Thresholds are set conservatively: only trigger overrides when
    the rolling window is full *and* performance is clearly poor,
    to avoid false-positive resets during the early exploration phase.

    Attributes:
        prediction_results: Rolling window of True/False prediction outcomes.
        action_results: Rolling window of 'novel'/'repeat'/'noop' labels.
        total_actions: Cumulative action count across all levels.
        total_novel_states: Cumulative novel-state count.
        override_count: Number of overrides issued (diagnostic).
    """

    # Override thresholds
    _WORLD_MODEL_RESET_ACCURACY = 0.20   # reset if accuracy below this
    _STRATEGY_CHANGE_EFFICIENCY = 0.05   # change strategy if novelty below this
    _REPERCEIVE_ACCURACY = 0.15          # reperceive if accuracy below this
    _MIN_WINDOW = 10                     # minimum samples before any override

    # Minimum actions between consecutive strategy-change overrides.
    _STRATEGY_CHANGE_COOLDOWN = 30

    def __init__(self, window_size: int = 20) -> None:
        self.prediction_results: deque = deque(maxlen=window_size)
        self.action_results: deque = deque(maxlen=window_size)
        self.total_actions: int = 0
        self.total_novel_states: int = 0
        self.override_count: int = 0
        self._last_strategy_change_at: int = -self._STRATEGY_CHANGE_COOLDOWN

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(self, predicted_correctly: bool) -> None:
        """Record whether the latest prediction was correct.

        Args:
            predicted_correctly: True if prediction matched actual delta.
        """
        self.prediction_results.append(predicted_correctly)

    def record_action_result(self, result_type: str) -> None:
        """Record the information content of the latest action.

        Args:
            result_type: One of 'novel' (new state reached), 'repeat'
                (state already seen), or 'noop' (environment unchanged).
        """
        self.action_results.append(result_type)
        self.total_actions += 1
        if result_type == 'novel':
            self.total_novel_states += 1

    # ------------------------------------------------------------------
    # Override signals
    # ------------------------------------------------------------------

    def should_reset_world_model(self) -> bool:
        """True if world model predictions are consistently wrong.

        Returning True causes agent.py to call rule_engine.full_reset(),
        discarding all induced rules and starting over.

        Returns:
            True if the rolling prediction accuracy is critically low.
        """
        if len(self.prediction_results) < self._MIN_WINDOW:
            return False
        if self.prediction_accuracy < self._WORLD_MODEL_RESET_ACCURACY:
            self.override_count += 1
            return True
        return False

    def should_change_strategy(self) -> bool:
        """True if exploration is producing no new information.

        Rate-limited by _STRATEGY_CHANGE_COOLDOWN to prevent the agent
        cycling through the same actions on every window evaluation.

        Returns:
            True if efficiency is low AND the cooldown has elapsed.
        """
        if len(self.action_results) < self._MIN_WINDOW:
            return False
        if self.exploration_efficiency >= self._STRATEGY_CHANGE_EFFICIENCY:
            return False
        cooldown_elapsed = (
            self.total_actions - self._last_strategy_change_at
            >= self._STRATEGY_CHANGE_COOLDOWN
        )
        if cooldown_elapsed:
            self._last_strategy_change_at = self.total_actions
            return True
        return False

    def should_reperceive(self) -> bool:
        """True if the perception layer may be mis-segmenting objects.

        Returning True causes agent.py to re-segment the current frame
        with a lower min_size threshold, potentially revealing small objects
        that were previously filtered out.

        Returns:
            True if prediction accuracy is critically low (implies bad perception).
        """
        if len(self.prediction_results) < self._MIN_WINDOW:
            return False
        if self.prediction_accuracy < self._REPERCEIVE_ACCURACY:
            self.override_count += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def prediction_accuracy(self) -> float:
        """Rolling mean prediction accuracy (0–1).

        Returns:
            Fraction of recent predictions that were correct.
            Returns 0.5 (neutral prior) when the window is empty.
        """
        if not self.prediction_results:
            return 0.5
        return sum(self.prediction_results) / len(self.prediction_results)

    @property
    def exploration_efficiency(self) -> float:
        """Fraction of recent actions that reached a novel state.

        Returns:
            Float in [0, 1]; 0.0 when window is empty.
        """
        if not self.action_results:
            return 0.0
        return sum(1 for r in self.action_results if r == 'novel') / len(self.action_results)

    def get_status(self) -> dict:
        """Return a snapshot of monitor state for logging.

        Returns:
            Dict with prediction_accuracy, exploration_efficiency,
            total_actions, total_novel_states, override_count.
        """
        return {
            'prediction_accuracy': round(self.prediction_accuracy, 3),
            'exploration_efficiency': round(self.exploration_efficiency, 3),
            'total_actions': self.total_actions,
            'total_novel_states': self.total_novel_states,
            'override_count': self.override_count,
        }

    def reset(self) -> None:
        """Clear rolling windows for a new level.

        Total cumulative counters are preserved for cross-level diagnostics.
        """
        self.prediction_results.clear()
        self.action_results.clear()
