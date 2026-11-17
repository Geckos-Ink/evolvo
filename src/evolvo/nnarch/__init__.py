"""GFSL-NN: genetic description, search and training of torch architectures.

Pure Python core (language, statistics, policy, efficiency, stopping, search); the PyTorch
pieces (``build``, ``train``) import torch only when used. See ``papers/GFSL-NN.md``.
"""

from .efficiency import EfficiencyModel, PowerLawFit, fit_power_law
from .language import (
    ACTIVATIONS,
    NONE,
    SET,
    SLOT_COUNT,
    VEC,
    ArchGenome,
    ArchSpace,
    ArgSpec,
    Chooser,
    Decision,
    Emitter,
    LoweredGraph,
    Node,
    OpRegistry,
    OpSpec,
    Row,
    Shape,
    default_registry,
    register_macro,
)
from .policy import AdaptivePolicy, size_prior
from .search import MUTATIONS, ArchitectureSearch, Candidate, Knowledge
from .stats import ContextualStats, Estimate, log_bucket
from .stopping import PlateauStopper, fit_learning_curve

__all__ = [
    "ACTIVATIONS", "AdaptivePolicy", "ArchGenome", "ArchSpace", "ArchitectureSearch", "ArgSpec",
    "Candidate", "Chooser", "ContextualStats", "Decision", "EfficiencyModel", "Emitter", "Estimate",
    "Knowledge", "LoweredGraph", "MUTATIONS", "NONE", "Node", "OpRegistry", "OpSpec",
    "PlateauStopper", "PowerLawFit", "Row", "SET", "SLOT_COUNT", "Shape", "VEC",
    "default_registry", "fit_learning_curve", "fit_power_law", "log_bucket", "register_macro",
    "size_prior",
]
