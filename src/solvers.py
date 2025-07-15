'''A module with implementations of assorted algorithms to solve quartic polynomials.
Works with the cmath form of complex numbers, specifically the mpmath implementation.
'''

from collections.abc import Iterable
import numpy as np
import mpmath
from mpmath import mpf, mpc, fabs, chop
import src.quartics.ferrari_qs as ferrari
import src.quartics.tenten_qs as tenten
import src.quartics.numpy_qs as numpyq

BASE_IMAG_THRESHOLD = mpf("1e-16")

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


def calc_real_roots_numpy(coeffs: Iterable[float]) -> list[float]:
    """
    Solves for real roots of the polynomial using numpy's eigenvalue / matrix implementation

    Parameters
    ----------
    coeffs : Iterable[float]
        The coefficients of the polynomial to solve

    Returns
    -------
    A list of the up to four real roots of the given quartic polynomial, or an empty list if none.
    """
    all_roots = np.roots(coeffs)
    real_roots = all_roots[np.isreal(all_roots)]
    return [float(np.real(r)) for r in real_roots]


def calc_real_roots_ferrari_highp(coeffs: Iterable[mpf], 
                                  imag_threshold=BASE_IMAG_THRESHOLD) -> list[mpf]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 for its real roots.
    Uses mpmath for high precision

    Parameters
    ----------
    coeffs : Iterable[mpf], length 5
        Quartic coefficients in the form [a, b, c, d, e] where ax^4 + bx^3 + cx^2 + dx + e = 0.
    imag_threshold : float-like, default 1e-12
        What magnitude an imaginary component a number can have and still be considered "real".
        This is likely in and of itself an experiment or at least conscious design choice.

    Returns
    -------
    A list of the up to four real roots of the given quartic. If none found, returns empty list.

    Note
    ----
    See `src/quartics/ferrari_qs_functor.py` for further notes about implementation.
    '''
    for coeff in coeffs:
        assert type(coeff) == mpf

    cmp_roots = ferrari.FerrariSolver(coeffs)()
    real_roots = [root.real for root in cmp_roots if mpmath.fabs(root.imag) < imag_threshold]
    return real_roots


def calc_real_roots_1010(coeffs: Iterable[mpf],
                         imag_threshold=BASE_IMAG_THRESHOLD) -> list[mpf]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 for its real roots.
    Uses Algorithm 1010 and mpmath for arbitrary precision

    Parameters
    ----------
    coeffs : Iterable[mpf], length 5
        Quartic coefficients in the form [a, b, c, d, e] where ax^4 + bx^3 + cx^2 + dx + e = 0.
    imag_threshold : float-like, default 1e-12
        What magnitude an imaginary component a number can have and still be considered "real".
        This is likely in and of itself an experiment or at least conscious design choice.

    Returns
    -------
    A list of the up to four real roots of the given quartic. If none found, returns empty list.

    Note
    ----
    See `src/quartics/tenten_qs.py` for further notes about implementation.
    '''
    for coeff in coeffs:
        assert isinstance(coeff, mpf)
    
    cmp_roots = tenten.Alg1010Solver(coeffs)()
    real_roots = [root.real for root in cmp_roots if mpmath.fabs(root.imag) < imag_threshold]
    return real_roots
    