import numpy as np

from .casgap import casgap
from .common import CasgapParameters, CasgapState
from .logger import debug


def run_example():
    params = CasgapParameters(
        seed=11351,
        box_length=100,
        mean_r=10,
        sd_r=0.1,
        mean_gamma=2,
        sd_gamma=0.1,
        orientation_axis=[
            -np.sqrt(6 + 3 * np.sqrt(3)),
            np.sqrt(2 + np.sqrt(3)),
            0,
        ]
        / np.sqrt(8 + 4 * np.sqrt(3)),
        omega=-75,
        kappa=100,
        volfrac=0.1,
        working_directory=".",
    )
    state = CasgapState(params)

    with debug():
        casgap(state)
