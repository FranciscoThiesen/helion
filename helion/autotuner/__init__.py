from __future__ import annotations

from .cmaes_search import CMAESSearch as CMAESSearch
from .config_fragment import BooleanFragment as BooleanFragment
from .config_fragment import EnumFragment as EnumFragment
from .config_fragment import IntegerFragment as IntegerFragment
from .config_fragment import ListOf as ListOf
from .config_fragment import PowerOfTwoFragment as PowerOfTwoFragment
from .config_spec import ConfigSpec as ConfigSpec
from .de_surrogate_hybrid import DESurrogateHybrid as DESurrogateHybrid
from .differential_evolution import (
    DifferentialEvolutionSearch as DifferentialEvolutionSearch,
)
from .effort_profile import AutotuneEffortProfile as AutotuneEffortProfile
from .effort_profile import DifferentialEvolutionConfig as DifferentialEvolutionConfig
from .effort_profile import MultiFidelityBOConfig as MultiFidelityBOConfig
from .effort_profile import PatternSearchConfig as PatternSearchConfig
from .effort_profile import RandomSearchConfig as RandomSearchConfig
from .finite_search import FiniteSearch as FiniteSearch
from .genetic_algorithm_search import (
    GeneticAlgorithmSearch as GeneticAlgorithmSearch,
)
from .local_cache import LocalAutotuneCache as LocalAutotuneCache
from .local_cache import StrictLocalAutotuneCache as StrictLocalAutotuneCache
from .multifidelity_bo_search import (
    MultiFidelityBayesianSearch as MultiFidelityBayesianSearch,
)
from .multifidelity_rf_search import (
    MultiFidelityRandomForestSearch as MultiFidelityRandomForestSearch,
)
from .pattern_search import PatternSearch as PatternSearch
from .pso_search import ParticleSwarmSearch as ParticleSwarmSearch
from .random_search import RandomSearch as RandomSearch
from .tpe_search import TreeStructuredParzenEstimator as TreeStructuredParzenEstimator

search_algorithms = {
    "CMAESSearch": CMAESSearch,
    "DESurrogateHybrid": DESurrogateHybrid,
    "DifferentialEvolutionSearch": DifferentialEvolutionSearch,
    "FiniteSearch": FiniteSearch,
    "GeneticAlgorithmSearch": GeneticAlgorithmSearch,
    "MultiFidelityBayesianSearch": MultiFidelityBayesianSearch,
    "MultiFidelityRandomForestSearch": MultiFidelityRandomForestSearch,
    "ParticleSwarmSearch": ParticleSwarmSearch,
    "PatternSearch": PatternSearch,
    "RandomSearch": RandomSearch,
    "TPESearch": TreeStructuredParzenEstimator,
}
