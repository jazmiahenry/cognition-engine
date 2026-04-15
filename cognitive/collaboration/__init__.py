"""Hebbian collaborative adaptation with Bayesian module combination."""

from .shared_pool import SharedRulePool
from .knowledge_module import KnowledgeModule, ModuleScore, ALL_MODULES
from .synaptic_network import SynapticNetwork
from .hebbian_engine import HebbianEngine, CouplingPhase
from .bayesian_combiner import BayesianCombiner

__all__ = [
    "SharedRulePool",
    "KnowledgeModule",
    "ModuleScore",
    "ALL_MODULES",
    "SynapticNetwork",
    "HebbianEngine",
    "CouplingPhase",
    "BayesianCombiner",
]
