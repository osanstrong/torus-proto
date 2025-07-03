'''A module with implementations of assorted algorithms to solve quartic polynomials.
Works with the cmath form of complex numbers, specifically the mpmath implementation.
'''

from collections.abc import Iterable
import numpy as np
import mpmath
from mpmath import mpf, mpc
import src.quartics.ferrari_qs_base as fqs_base
import src.quartics.ferrari_qs_mixp as fqs_mixp


def calc_real_roots_numpy(coeffs: Iterable[float]) -> list[float]:
    """
    Solves for real roots of the polynomial using numpy's eigenvalue / matrix implementation

    Returns
    -------
    A list of the up to four real roots of the given quartic polynomial, or an empty list if none.

    Parameters
    ----------
    coeffs : Iterable[float]
        The coefficients of the polynomial to solve
    """
    all_roots = np.roots(coeffs)
    real_roots = all_roots[np.isreal(all_roots)]
    return [float(np.real(r)) for r in real_roots]


def calc_real_roots_ferrari(coeffs: Iterable[float], 
                            normalized:bool = True, 
                            imag_threshold=1e-12) -> list[float]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 for its real roots.

    Returns
    -------
    A list of the up to four real roots of the given quartic. If none found, returns empty list.

    Parameters
    ----------
    coeffs : Iterable[float], length 4
        Quartic coefficients in the form [a, b, c, d, e] where ax^4 + bx^3 + cx^2 + dx + e = 0.
    normalized : bool, default True
        Whether the polynomial has already been normalized/scaled such that a = 1.
        Defaults to true, as in its primary application, the polynomial is already made so.
    imag_threshold : float-like, default 1e-12
        What magnitude an imaginary component a number can have and still be considered "real".
        This is likely in and of itself an experiment or at least conscious design choice.


    Note
    ----
    See `src/external/fqs_limited.py` for further notes about implementation.
    '''
    a, b, c, d, e = coeffs
    cmp_roots = fqs_base.single_quartic(a, b, c, d, e)

    return [root.real for root in cmp_roots if abs(root.imag) < imag_threshold]


def calc_real_roots_ferrari_highp(coeffs: Iterable[mpf], 
                                  normalized: bool = True, 
                                  imag_threshold=mpf("1e-16")) -> list[mpf]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 for its real roots.
    Uses mpmath for high precision

    Returns
    -------
    A list of the up to four real roots of the given quartic. If none found, returns empty list.

    Parameters
    ----------
    coeffs : Iterable[mpf], length 4
        Quartic coefficients in the form [a, b, c, d, e] where ax^4 + bx^3 + cx^2 + dx + e = 0.
    normalized : bool, default True
        Whether the polynomial has already been normalized/scaled such that a = 1.
        Defaults to true, as in its primary application, the polynomial is already made so.
    imag_threshold : float-like, default 1e-12
        What magnitude an imaginary component a number can have and still be considered "real".
        This is likely in and of itself an experiment or at least conscious design choice.


    Note
    ----
    See `src/external/fqs_mixp.py` for further notes about implementation.
    '''
    for coeff in coeffs:
        assert type(coeff) == mpf

    a, b, c, d, e = coeffs
    if not a == mpf(1):
        b /= a
        c /= a
        d /= a
        e /= a
        a = mpf(1)
    cmp_roots = fqs_mixp.solve_normalized_quartic(b, c, d, e)

    return [root.real for root in cmp_roots if mpmath.fabs(root.imag) < imag_threshold]