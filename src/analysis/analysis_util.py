'''A module with functions to make it simpler to analyze different permutations of algorithms
Compare:
 - Solvers
 - Precision levels
 - Edge cases/epsilon variation therefrom
 - Starting distance of otherwise equivalent rays
'''

from collections.abc import Iterable, Callable
import mpmath
from mpmath import mpf, mpc
import toroid


# Common precision levels to test
SINGLE_PREC: int = 23
DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999


def intersections_by_solver(
    toroid: toroid.EllipticToroid,
    ray_src: Iterable[mpf], 
    ray_dir: Iterable[mpf],
    solvers: list[type|str] = ["Ferrari", "Alg1010"],
    prec: int = DOUBLE_PREC
) -> dict:
    '''
    Returns a dictionary of intersection results for the given ray torus combination.

    Parameters
    ----------
    toroid
    '''
    pass