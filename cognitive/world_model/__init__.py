"""World model package: Stage 2 — rule induction, state diffing, prediction."""

from .state_delta import ObjectDelta, StateDelta, compute_delta
from .rule_engine import Rule, RuleEngine
from .predictor import Predictor

__all__ = ["ObjectDelta", "StateDelta", "compute_delta", "Rule", "RuleEngine", "Predictor"]
