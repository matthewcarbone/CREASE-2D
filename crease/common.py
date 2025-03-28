import os
from dataclasses import dataclass
from typing import Optional, Tuple

from monty.json import MSONable


@dataclass
class CasgapParameters(MSONable):
    """Parameter object used as input for the CASGAP calculations.

    Attributes:
        seed:
            The random seed used to ensure reproducibility and pseudo-
            randomness during calculations.
        boxlength:
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
    boxlength: float
    mean_r: float
    sd_r: float
    mean_gamma: float
    sd_gamma: float
    orientation_axis: Tuple[float, float, float]
    omega: float
    kappa: float
    volfrac: float
    checkpoint_frequency: int = 1000
    working_directory: Optional[os.PathLike] = None
