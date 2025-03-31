import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Optional, Tuple, Union

import numpy as np
from monty.json import MSONable

from .logger import logger


@dataclass
class CasgapParticleList(MSONable):
    n_particles: int
    ac: Any  # WARNING: fix, no idea what this is
    quat: Any  # WARNING: fix, no idea what this is
    xyz: np.ndarray
    polyhedra: dict = field(default_factory=dict)
    n_prime: int = 0

    def __str__(self) -> str:
        return f"N={self.n_particles} ac.shape={self.ac.shape} quat.shape={self.quat.shape} xyz.shape={self.xyz.shape}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class CasgapParameters(MSONable):
    """Parameter object used as input for the CASGAP calculations.

    Attributes:
        seed:
            The random seed used to ensure reproducibility and pseudo-
            randomness during calculations.
        box_length:
            The length of a single side of the square box used in the
            simulation.
        mean_r:
            The mean radius to use when generating particles.
        sd_r:
            Corresponding standard deviation for the particle radii.
        mean_gamma:
            The mean value of the ratio between a and c, the two independent
            values of the elliptical radii.
        sd_gamma:
            Corresponding standard deviation for the gamma ratio.
        orientation_axis:
            ...
        omega:
            ...
        kappa:
            ...
        volfrac:
            The volume fraction control parameter. Used to determine the
            packing density of the particles.
        working_directory:
            The working directory for all calculations. If it is not provided,
            defaults to a temporary location on disk, somewhere like /tmp.
    """

    seed: int
    box_length: float
    mean_r: float
    sd_r: float
    mean_gamma: float
    sd_gamma: float
    orientation_axis: Tuple[float, float, float]
    omega: float
    kappa: float
    volfrac: float
    phaseII_loop_start: int = 1
    checkpoint_frequency: int = 1000
    working_directory: Optional[Union[os.PathLike, str]] = None

    def __post_init__(self):
        if self.working_directory is not None:
            Path(self.working_directory).mkdir(exist_ok=True, parents=True)
            logger.info(f"Working directory set to: {self.working_directory}")
        else:
            d = Path(gettempdir())
            logger.warning("Working directory is unset")
            logger.warning(f"Defaulting to system temporary dir: {d}")
            self.working_directory = d


@dataclass
class CasgapState(MSONable):
    params: CasgapParameters
    particle_list: Optional[CasgapParticleList] = None
    in_error_state: bool = False

    def checkpoint(self):
        if self.params.working_directory is None:
            raise ValueError("A checkpoint (working_directory) must be set")
        target = Path(self.params.working_directory) / "state.json"
        self.save(
            target, json_kwargs={"indent": 4, "sort_keys": True}, strict=False
        )
