'''A module with implementations of assorted algorithms to solve quartic polynomials.
Works with the cmath form of complex numbers, specifically the mpmath implementation.
'''

from collections.abc import Iterable
import numpy as np
import mpmath
from mpmath import mpf, mpc, chop
import src.quartics.ferrari_qs as ferrari
import src.quartics.tenten_qs as tenten
import src.quartics.numpy_qs as numpyq

BASE_IMAG_THRESHOLD = None # Use default mpmath tolerance for chop()

def calc_real_roots(coeffs: Iterable[mpf], solver: type|str, imag_threshold: mpf = BASE_IMAG_THRESHOLD) -> list[mpf]:
    '''
    Solves the real roots of the polynomial given by the provided coefficients,
    using the specified algorithm.
    
    Parameters
    ----------
    coeffs : Iterable[mpf]
        The coefficients of the polynomial to solve
    solver : type|str
        The (Functor pattern) solver to use, or a string corresponding to a known one
        Acceptable strings:
         - "tt", "Alg1010": Uses Alg1010 solver
         - "fr", "Ferrari": Uses Ferrari solver
         - "np", "numpy": Uses the numpy solver
        A direct functor should have the pattern solver(coeffs)() = roots
    imag_threshold : mpf, optional
        The threshold below which imaginary values can be discarded

    Returns
    -------
    The roots found by the given solver, which have an imaginary component smaller than a certain threshold
    '''
    type_func = type(solver)
    assert isinstance(solver, type|str)
    # If a string is given, find which solver they meant
    if isinstance(solver, str):
        match solver:
            case "tt" | "Alg1010":
                solver = tenten.Alg1010Solver
            case "fr" | "Ferrari":
                solver = ferrari.FerrariSolver
            case "np" | "numpy":
                solver = numpyq.NumpySolver
    all_roots: list[mpc] = solver(coeffs)()
    real_roots = [r.real for r in all_roots if is_real(r, tolerance=imag_threshold)]
    return real_roots


def is_real(val: mpc, tolerance: mpf = None) -> bool:
    '''
    Returns whether the given mpc instance is real, 
    using mpmath.chop() with the given tolerance
    '''
    return chop(val, tol=tolerance).imag == 0