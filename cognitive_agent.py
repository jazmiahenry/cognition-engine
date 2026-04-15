"""Cognitive agent for the ARC-AGI-3 competition.

Implements a four-stage cognitive cascade with Hebbian collaborative adaptation:
  Stage 1 — Perception:      segment_frame + ObjectTracker
  Stage 2 — World Model:     RuleEngine + Predictor + BayesianCombiner
  Stage 3 — Planning:        GoalInference + Explorer + SimulationEnsemble (MCTS)
  Stage 4 — Metacognition:   MetacognitiveMonitor + HebbianEngine

Collaboration architecture:
  KnowledgeModules (Spelke)  — fixed feature detectors per domain
  BayesianCombiner (Tenenbaum) — learns which modules explain this game
  SynapticNetwork            — W[i][j] trust weights between agents
  HebbianEngine              — LTP/LTD/decay/homeostasis plasticity
  PhaseController            — amodular/bimodal/multimodal gating

No external API calls — everything runs locally.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set

import random

import numpy as np
from arc_agi import Arcade
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
from cognitive.collaboration import (
    SharedRulePool,
    SynapticNetwork,
    HebbianEngine,
    BayesianCombiner,
    CouplingPhase,
)

logger = logging.getLogger(__name__)

_EXPLORATION_BUDGET: int = 10
_ENSEMBLE_N_TREES: int = 4
_ENSEMBLE_N_SIMS: int = 200
# Hebbian cycle runs every N actions after exploration phase.
_HEBBIAN_CYCLE_INTERVAL: int = 10
# Simulation pre-play: free actions on a copy before real game.
_SIM_ACTIONS: int = 150
# Brute force kicks in after this many real actions with no level progress.
_BRUTE_FORCE_THRESHOLD: int = 60


class CognitiveAgent(Agent):
    """Four-stage cognitive agent with Hebbian collaborative adaptation.

    Registered as "cognitiveagent" in AVAILABLE_AGENTS.
    Run with: uv run main.py --agent=cognitiveagent --game=<game_id>
    """

    MAX_ACTIONS: int = 80

    # Class-level shared state — all instances share these. Thread-safe.
    _shared_pool: SharedRulePool = SharedRulePool()
    _synaptic_network: SynapticNetwork = SynapticNetwork()
    _hebbian_engine: HebbianEngine = HebbianEngine(
        network=_synaptic_network,
        alpha=0.1,
        beta=0.05,
        delta=0.01,
        max_row_sum=2.5,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Stage 1 — Perception
        self.tracker = ObjectTracker()

        # Stage 2 — World Model
        self.rule_engine = RuleEngine()
        self.predictor = Predictor(self.rule_engine)
        self.bayesian = BayesianCombiner()

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
        self._available_action_ids: Optional[List[int]] = None
        self._colors_seen: Set[int] = set()
        self._action_types_used: Set[str] = set()
        self._last_q: float = 0.0
        self._last_hebbian_at: int = -_HEBBIAN_CYCLE_INTERVAL
        self._borrowed_from: List[str] = []
        self._sim_done: bool = False
        self._brute_force_idx: int = 0

        # Register in synaptic network.
        self._synaptic_network.ensure_agent(self.game_id)

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

        # Capture available_actions from the environment on first observation.
        if (self._available_action_ids is None
                and getattr(latest_frame, 'available_actions', None)):
            self._available_action_ids = list(latest_frame.available_actions)
            logger.info(
                "Game %s available_actions: %s",
                self.game_id, self._available_action_ids,
            )

        # Simulation pre-play: run free actions on a copy to build world model.
        if not self._sim_done:
            self._run_simulation_preplay()
            self._sim_done = True

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
            high_rules = self.rule_engine.get_high_confidence_rules()
            self.prior.update_rules(high_rules)
            self.prior.update_player(delta, scene)

            # Bayesian update: score all core knowledge modules.
            self.bayesian.update(scene, delta, self._last_action_taken)

            # Track colors and action types for collaboration scoring.
            self._colors_seen.update(o.color for o in scene.objects.values())
            action_name = getattr(self._last_action_taken, 'name', '')
            if action_name:
                self._action_types_used.add(action_name)

            # Publish rules to the shared pool for other agents.
            self._shared_pool.publish(
                game_id=self.game_id,
                rules=high_rules,
                colors_seen=self._colors_seen,
                action_types_used=self._action_types_used,
                object_count=len(scene.objects),
                levels_completed=self._levels_completed_internal,
                available_action_ids=self._available_action_ids,
            )

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
        # Stage 2.5: Hebbian collaborative adaptation
        # ------------------------------------------------------------------
        self._run_hebbian_cycle(scene)

        # ------------------------------------------------------------------
        # Stage 3: Action selection
        # ------------------------------------------------------------------
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
        """Choose the next action across three phases.

        Phase 1 — Exploration (actions 0…EXPLORATION_BUDGET-1):
            Explorer gathers observations to supplement simulation knowledge.

        Phase 2 — MCTS (actions EXPLORATION_BUDGET…BRUTE_FORCE_THRESHOLD-1):
            MCTS always runs. Uses whatever rules exist (from simulation
            pre-play + real observations). Never bypassed.

        Phase 3 — Brute force (actions BRUTE_FORCE_THRESHOLD+):
            If no level completed, systematically cycle through all actions
            and object clicks deterministically.

        Args:
            scene: Current SceneGraph from Stage 1.
            available: GameAction objects for this turn.

        Returns:
            A configured GameAction with reasoning attached.
        """
        high_conf_rules = self.rule_engine.get_high_confidence_rules()
        goals = self.goal_inference.generate_hypotheses(
            scene, self.rule_engine.observations
        )
        self.goal_inference.update_priorities(None, high_conf_rules)

        # ------------------------------------------------------------------
        # Phase 3: Brute force fallback
        # ------------------------------------------------------------------
        if self._level_action_count >= _BRUTE_FORCE_THRESHOLD:
            return self._do_brute_force(scene, available)

        # ------------------------------------------------------------------
        # Phase 1: Short exploration
        # ------------------------------------------------------------------
        if self._level_action_count < _EXPLORATION_BUDGET:
            return self._do_explore(scene, available)

        # ------------------------------------------------------------------
        # Phase 2: MCTS always runs
        # ------------------------------------------------------------------
        best_action, winning_hypothesis, best_q = self.ensemble.run(
            scene=scene,
            goal_hypotheses=goals,
            available_actions=available,
            n_trees=_ENSEMBLE_N_TREES,
            n_sims=_ENSEMBLE_N_SIMS,
        )

        self._last_q = best_q

        # Fall back to Explorer if MCTS has zero signal.
        if best_action is None or best_q < 0.01:
            return self._do_explore(scene, available)

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
            "rules_available": len(high_conf_rules),
            "player_id": self.prior.player_id,
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

    def _run_simulation_preplay(self) -> None:
        """Multi-phase simulation proving ground.

        Phase 1 — Explore: build rules until saturation.
        Phase 2 — Attempt: use MCTS to try solving the level in sim.
        Phase 3 — Brute force: if MCTS fails, try systematic actions in sim.
        Phase 4 — Collaborate: borrow rules from peers and retry.

        Only strategies proven in simulation transfer to the real game.
        """
        import os
        op_mode = os.environ.get("OPERATION_MODE", "online")
        if op_mode == "online":
            logger.info("Simulation preplay skipped — online mode would hit rate limits")
            return

        try:
            sim_arc = Arcade()
            sim_env = sim_arc.make(self.game_id)
            sim_obs = sim_env.reset()
        except Exception as e:
            logger.warning("Simulation preplay failed to create env: %s", e)
            return

        if (self._available_action_ids is None
                and getattr(sim_obs, 'available_actions', None)):
            self._available_action_ids = list(sim_obs.available_actions)

        available = self._all_actions()
        if not available:
            return

        sim_tracker = ObjectTracker()
        raw = sim_obs.frame[0] if sim_obs.frame else None
        if raw is None:
            return

        prev_frame = np.array(raw.tolist() if hasattr(raw, 'tolist') else raw, dtype=np.int32)
        prev_scene = segment_frame(prev_frame, min_size=2)
        prev_scene = sim_tracker.update(prev_scene)

        # ------- Phase 1: Explore — build rules until saturation -------
        _SAT_WINDOW = 20
        sim_actions = 0
        actions_since_new_rule = 0
        prev_rule_count = 0

        for i in range(200):
            if i < len(available):
                action = available[i % len(available)]
            else:
                action = random.choice(available)

            data = {}
            if hasattr(action, 'is_complex') and action.is_complex():
                objs = list(prev_scene.objects.values())
                if objs:
                    obj = objs[i % len(objs)]
                    cy, cx = obj.centroid
                    data = {'x': int(round(cx)), 'y': int(round(cy))}
                else:
                    data = {'x': (i * 7) % 64, 'y': (i * 11) % 64}

            prev_scene, sim_obs = self._sim_step(
                sim_env, action, data, prev_scene, sim_tracker
            )
            if prev_scene is None:
                continue
            sim_actions += 1

            current_rule_count = len(self.rule_engine.rules)
            if current_rule_count > prev_rule_count:
                actions_since_new_rule = 0
                prev_rule_count = current_rule_count
            else:
                actions_since_new_rule += 1

            if actions_since_new_rule >= _SAT_WINDOW and sim_actions >= 30:
                break
            if sim_obs.state in (GameState.WIN, GameState.GAME_OVER):
                break

        rules_p1 = len(self.rule_engine.rules)
        self.prior.update_rules(self.rule_engine.get_high_confidence_rules())

        self._shared_pool.publish(
            game_id=self.game_id,
            rules=self.rule_engine.get_high_confidence_rules(),
            colors_seen=self._colors_seen,
            action_types_used=self._action_types_used,
            object_count=len(prev_scene.objects) if prev_scene else 0,
            levels_completed=self._levels_completed_internal,
            available_action_ids=self._available_action_ids,
        )

        # ------- Phase 2: Attempt — MCTS-guided solve in sim -------
        phase2_solved = False
        phase2_actions = 0
        try:
            sim_obs = sim_env.reset()
            raw = sim_obs.frame[0] if sim_obs.frame else None
            if raw is not None:
                pf = np.array(raw.tolist() if hasattr(raw, 'tolist') else raw, dtype=np.int32)
                prev_scene = segment_frame(pf, min_size=2)
                prev_scene = sim_tracker.update(prev_scene)

            for j in range(80):
                goals = self.goal_inference.generate_hypotheses(
                    prev_scene, self.rule_engine.observations
                )
                best_act, _, best_q = self.ensemble.run(
                    scene=prev_scene, goal_hypotheses=goals,
                    available_actions=available, n_trees=2, n_sims=100,
                )
                act = best_act if best_act and best_q >= 0.01 else random.choice(available)

                d = {}
                if isinstance(act, ClickAction):
                    d = {'x': act.x, 'y': act.y}
                    act = act.base_action
                elif hasattr(act, 'is_complex') and act.is_complex():
                    objs = list(prev_scene.objects.values())
                    if objs:
                        o = random.choice(objs)
                        cy, cx = o.centroid
                        d = {'x': int(round(cx)), 'y': int(round(cy))}

                prev_scene, sim_obs = self._sim_step(
                    sim_env, act, d, prev_scene, sim_tracker
                )
                phase2_actions += 1
                if prev_scene is None:
                    continue
                if sim_obs.state == GameState.WIN:
                    phase2_solved = True
                    break
                if sim_obs.state == GameState.GAME_OVER:
                    break
        except Exception:
            pass

        if phase2_solved:
            logger.info(
                "Sim preplay: %s SOLVED Phase2 (MCTS) — %d+%d actions, %d rules",
                self.game_id, sim_actions, phase2_actions, len(self.rule_engine.rules),
            )
            return

        # ------- Phase 3: Brute force in sim -------
        phase3_solved = False
        phase3_actions = 0
        try:
            sim_obs = sim_env.reset()
            raw = sim_obs.frame[0] if sim_obs.frame else None
            if raw is not None:
                pf = np.array(raw.tolist() if hasattr(raw, 'tolist') else raw, dtype=np.int32)
                prev_scene = segment_frame(pf, min_size=2)
                prev_scene = sim_tracker.update(prev_scene)

            objects = list(prev_scene.objects.values()) if prev_scene else []
            simple = [a for a in available if not (hasattr(a, 'is_complex') and a.is_complex())]
            clicks = [a for a in available if hasattr(a, 'is_complex') and a.is_complex()]
            brute_list = list(simple)
            for ob in objects:
                for ca in clicks:
                    brute_list.append((ca, ob))

            for idx in range(min(len(brute_list), 100)):
                entry = brute_list[idx]
                d = {}
                if isinstance(entry, tuple):
                    act, ob = entry
                    cy, cx = ob.centroid
                    d = {'x': int(round(cx)), 'y': int(round(cy))}
                else:
                    act = entry

                prev_scene, sim_obs = self._sim_step(
                    sim_env, act, d, prev_scene, sim_tracker
                )
                phase3_actions += 1
                if prev_scene is None:
                    continue
                if sim_obs.state == GameState.WIN:
                    phase3_solved = True
                    break
                if sim_obs.state == GameState.GAME_OVER:
                    break
        except Exception:
            pass

        if phase3_solved:
            logger.info(
                "Sim preplay: %s SOLVED Phase3 (brute) — %d total, %d rules",
                self.game_id, sim_actions + phase2_actions + phase3_actions,
                len(self.rule_engine.rules),
            )
            return

        # ------- Phase 4: Borrow from peers and retry -------
        obj_count = len(prev_scene.objects) if prev_scene else 0
        borrowed, score, donor = self._shared_pool.query_best_match(
            game_id=self.game_id,
            my_colors=self._colors_seen,
            my_action_types=self._action_types_used,
            my_object_count=obj_count,
            my_available_action_ids=self._available_action_ids,
            my_existing_rules=self.rule_engine.rules,
        )
        if borrowed:
            noop_pairs: set = set()
            for obs_a, bef, _, dl in self.rule_engine.observations:
                if dl.is_noop:
                    an = getattr(obs_a, 'name', str(obs_a))
                    for ob in bef.objects.values():
                        noop_pairs.add((an, ob.color))
            existing = {(r.action_type, r.target_property, r.target_color)
                        for r in self.rule_engine.rules}
            added = 0
            for rule in borrowed:
                rk = (rule.action_type, rule.target_property, rule.target_color)
                if rk in existing:
                    continue
                ra = rule.action_type if isinstance(rule.action_type, str) else str(rule.action_type)
                if (ra, rule.target_color) in noop_pairs:
                    continue
                self.rule_engine.rules.append(rule)
                existing.add(rk)
                added += 1
            if added > 0:
                self.prior.update_rules(self.rule_engine.get_high_confidence_rules())
                logger.info(
                    "Sim Phase4: %s borrowed %d from %s (compat=%.3f)",
                    self.game_id, added, donor, score,
                )

        logger.info(
            "Sim preplay: %s all phases — %d rules, player=%s, dominant=%s",
            self.game_id, len(self.rule_engine.rules),
            self.prior.player_id, self.bayesian.get_dominant_module(),
        )

    def _sim_step(self, sim_env, action, data, prev_scene, sim_tracker):
        """Execute one step in simulation. Returns (new_scene, sim_obs)."""
        try:
            sim_obs = sim_env.step(action, data=data) if data else sim_env.step(action)
        except Exception:
            return None, None
        raw = sim_obs.frame[0] if sim_obs.frame else None
        if raw is None:
            return None, sim_obs
        cf = np.array(raw.tolist() if hasattr(raw, 'tolist') else raw, dtype=np.int32)
        cs = segment_frame(cf, min_size=2)
        cs = sim_tracker.update(cs)
        delta = compute_delta(prev_scene, cs, action)
        self.rule_engine.observe(action, prev_scene, cs, delta)
        self.prior.update_player(delta, cs)
        self.bayesian.update(cs, delta, action)
        self._colors_seen.update(o.color for o in cs.objects.values())
        an = getattr(action, 'name', '')
        if an:
            self._action_types_used.add(an)
        return cs, sim_obs

    def _do_brute_force(self, scene: object, available: list) -> GameAction:
        """Systematic brute force: cycle through all actions deterministically.

        When exploration and MCTS have both failed, try every action
        and every object click in order. Last resort before giving up.

        Args:
            scene: Current SceneGraph.
            available: Available actions.

        Returns:
            A configured GameAction.
        """
        objects = list(getattr(scene, 'objects', {}).values())
        click_actions = [a for a in available if hasattr(a, 'is_complex') and a.is_complex()]
        simple_actions = [a for a in available if not (hasattr(a, 'is_complex') and a.is_complex())]

        # Build a deterministic action list: all simple actions + click each object.
        brute_list = list(simple_actions)
        for obj in objects:
            for ca in click_actions:
                brute_list.append((ca, obj))

        if not brute_list:
            action = random.choice(available)
            action.reasoning = {"stage": "brute_force", "reason": "no actions available"}
            return action

        entry = brute_list[self._brute_force_idx % len(brute_list)]
        self._brute_force_idx += 1

        if isinstance(entry, tuple):
            action, obj = entry
            cy, cx = obj.centroid
            action.set_data({'x': int(round(cx)), 'y': int(round(cy))})
            action.reasoning = {
                "stage": "brute_force",
                "target_obj": obj.obj_id,
                "coords": (int(round(cx)), int(round(cy))),
            }
        else:
            action = entry
            action.reasoning = {"stage": "brute_force"}

        logger.debug("BruteForce[%d]: %s", self._level_action_count, getattr(action, 'name', action))
        return action

    def _run_hebbian_cycle(self, scene: object) -> None:
        """Execute one Hebbian plasticity cycle.

        Sequence: decay → phase assignment → borrow if needed → LTP/LTD → homeostasis.

        Only runs every _HEBBIAN_CYCLE_INTERVAL actions after exploration phase.
        """
        if self._level_action_count < _EXPLORATION_BUDGET:
            return
        if self._level_action_count - self._last_hebbian_at < _HEBBIAN_CYCLE_INTERVAL:
            return

        # Get success rate from Bayesian combiner.
        success_rate = self.bayesian.combined_score()

        # Run the full Hebbian step: decay → phase → LTP/LTD → homeostasis.
        q_before = self._last_q
        phase = self._hebbian_engine.step(
            agent_id=self.game_id,
            success_rate=success_rate,
            q_before=q_before,
            q_after=self._last_q,
            borrowed_from=self._borrowed_from,
        )
        self._borrowed_from = []

        # If not AMODULAR, try borrowing from partners.
        if phase != CouplingPhase.AMODULAR:
            partners = self._hebbian_engine.get_partners(self.game_id, phase)
            self._borrow_from_partners(scene, partners)

        self._last_hebbian_at = self._level_action_count

        logger.info(
            "Hebbian[%d] phase=%s sr=%.3f Q=%.3f dominant=%s",
            self._level_action_count, phase.value, success_rate,
            self._last_q, self.bayesian.get_dominant_module(),
        )

    def _borrow_from_partners(
        self, scene: object, partners: list
    ) -> None:
        """Borrow rules from Hebbian partners with full guards.

        Filters: dedup, action applicability, color applicability, noop contradiction.
        """
        if not partners:
            return

        # Build noop evidence from our own observations.
        noop_pairs: set = set()
        for obs_action, before, _, delta in self.rule_engine.observations:
            if delta.is_noop:
                act_name = getattr(obs_action, 'name', str(obs_action))
                for obj in before.objects.values():
                    noop_pairs.add((act_name, obj.color))

        existing_keys = {
            (r.action_type, r.target_property, r.target_color)
            for r in self.rule_engine.rules
        }

        for partner_id, weight in partners:
            obj_count = len(getattr(scene, 'objects', {}))
            borrowed, score, donor = self._shared_pool.query_best_match(
                game_id=self.game_id,
                my_colors=self._colors_seen,
                my_action_types=self._action_types_used,
                my_object_count=obj_count,
                my_available_action_ids=self._available_action_ids,
                my_existing_rules=self.rule_engine.rules,
            )
            if not borrowed:
                continue

            added = 0
            for rule in borrowed:
                key = (rule.action_type, rule.target_property, rule.target_color)
                if key in existing_keys:
                    continue
                rule_act = rule.action_type if isinstance(rule.action_type, str) else str(rule.action_type)
                if (rule_act, rule.target_color) in noop_pairs:
                    continue
                self.rule_engine.rules.append(rule)
                existing_keys.add(key)
                added += 1

            if added > 0:
                self._borrowed_from.append(donor if donor else partner_id)
                self.prior.update_rules(self.rule_engine.get_high_confidence_rules())
                logger.info(
                    "Hebbian borrow: %s ← %s (%d rules, W=%.3f)",
                    self.game_id, donor, added, weight,
                )

    def _all_actions(self) -> list:
        """Return GameActions filtered by what this game reports as available.

        Reads available_actions dynamically from the frame data — no
        hardcoded per-game lists. Falls back to all non-RESET actions
        if the environment didn't provide the field.

        Returns:
            List of GameAction constants the game accepts.
        """
        if self._available_action_ids:
            return [a for a in GameAction if a.value in self._available_action_ids]
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
        self._available_action_ids = None  # re-read from next frame

        # Publish as mentor — this agent's rules are proven to work.
        self._shared_pool.publish(
            game_id=self.game_id,
            rules=self.rule_engine.get_high_confidence_rules(),
            colors_seen=self._colors_seen,
            action_types_used=self._action_types_used,
            object_count=0,
            levels_completed=new_level_count,
        )

        # Seed synaptic weights so other agents can reach this mentor.
        all_agents = self._synaptic_network.all_agents()
        self._hebbian_engine.seed_from_mentor(self.game_id, all_agents)

        # Reset prior's per-episode state but keep rules — they often transfer.
        self.prior.player_id = None
        self.prior.player_pos = None
        self.prior.goal_pos = None

        self.explorer.reset()
        self.tracker.reset()
        self.monitor.reset()
        self.rule_engine.reset()  # clears observations, keeps rules
        # Bayesian weights carry across levels — module trust transfers.

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
        self.bayesian.reset()
        self._last_q = 0.0
        self._borrowed_from = []
