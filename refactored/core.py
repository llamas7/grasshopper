from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import System


@dataclass
class Neighborhood:
    positive_spins: NDArray[np.int_]
    positive_contributions: NDArray[np.float64]
    negative_spins: NDArray[np.int_]
    negative_contributions: NDArray[np.float64]
    spin_potential: float


@dataclass
class NonAntipodalNeighborhood:
    neighbor_spins: NDArray[np.int_]
    contributions: NDArray[np.float64]
    spin_potential: float


NeighborhoodLike = Neighborhood | NonAntipodalNeighborhood


@dataclass
class Parameters:
    starting_temp: float = 50
    final_temp: float = 0.05
    cooling_rate: float = 0.954993
    min_acceptance: float = 0.0005
    max_runtime: float = 1  # Hours
    max_iterations: int = 100
    starting_steps: int = 100000
    its_per_save: int = 1
    grid_file: str | None = None
    save_dir: str = "./output_data"
    load_flag: bool = False


@dataclass
class AnnealData:
    steps: int
    temperature: float
    iterations: int = 0
    energies: list[float] = field(default_factory=list)
    update_stats: list[list] = field(default_factory=list)


class SystemType(str, Enum):
    INVERSE = "inverse"
    INDEPENDENT = "independent"
    NON_ANTIPODAL = "non-antipodal"


class DeltaType(str, Enum):
    TWO = "two"
    ONE_STAR = "one*"
    TWO_STAR = "two*"
