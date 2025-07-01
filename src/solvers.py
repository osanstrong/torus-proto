'''A module with implementations of assorted algorithms to solve quartic polynomials.
Works with the cmath form of complex numbers, specifically the mpmath implementation.
'''

from collections import Iterable
import numpy as np
import mpmath
from mpmath import mpf, mpc


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


def ferrari_stackexchange(coeffs: Iterable[mpf], normalized: bool) -> list[mpc]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0.

    Returns
    -------
    A list of the four roots of the given quartic polynomial, both real and complex.

    Parameters
    ----------
    coeffs : Iterable[mpf], length 4
        Quartic coefficients in the form [a, b, c, d, e] where ax^4 + bx^3 + cx^2 + dx + e = 0.
    normalized : bool
        Whether the polynomial has already been normalized/scaled such that a = 1. 

    Note
    ----
    Precise implementation derives from a stackexchange article https://stackoverflow.com/questions/35795663/fastest-way-to-find-the-smallest-positive-real-root-of-quartic-polynomial-4-degr
    Which cites the French Wikipedia's Ferrari implementation: https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Ferrari
    But still is the same Ferrari's algorithm in broader use, such as Spiegel Mathematical Handbook, Section I.9, page 33, published 1968 McGraw-Hill, Inc.
    '''
    [a, b, c, d, e] = coeffs
    if not normalized:
        b /= a
        c /= a
        d /= a
        e /= a
        a = 1
    # From this point onwards the polynomial is assumed to be normalized

    # Coordinate shift: x = z - b/4
    z0 = b / 4
    b2, c2, d2 = sq(b), sq(c), sq(d)
    p = (-3*b2)/8 + c/a
    q = (b*b2)/8 - b*c/2 + d
    r = -3*sq(b2)/256 + c*b2/16 - b*d/4 + e
    # now find a z that solves a subcubic Az^3 + Bz^2 + Cz + D = 0
    A = 8
    B = -4*p
    C = -8*r
    D = 4*r*p - sq(q)
    y0, y1, y2 = cardan(A, B, C, D)
    if abs(y1.imag) < abs(y0.imag):
        y0 = y1
    if abs(y2.imag) < abs(y0.imag):
        y0 = y2
    a0 = mpmath.sqrt(-p + 2*y0.real)
    if a0 == 0:
        b0 = sq(y0) - r
    else:
        b0 = -q / 2 / a0
    r0, r1 = roots2(1, a0, y0 + b0)
    r2, r3 = roots2(1, -a0, y0 - b0)
    return [r0 - z0, r1 - z0, r2 - z0, r3 - z0]


def calc_real_roots_ferrari(coeffs: Iterable[mpf], normalized:bool = True, 
                            imag_threshold=mpf("1e-12")) -> list[mpf]:
    '''Solves the quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 for its real roots.

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
    See ferrari_stackexchange() for further notes about implementation.
    '''
    cmp_roots = ferrari_stackexchange(coeffs, normalized=normalized)
    return [root.real for root in cmp_roots if mpmath.abs(root.imag) < imag_threshold]
    
    

# Helper Functions which aren't specifically solving quartics


J = mpmath.exp(mpc(2j * mpmath.pi / 3))
Jc = 1 / J

def cardan(a: mpf, b: mpf, c: mpf, d: mpf) -> (mpc, mpc, mpc):
    '''Finds the roots of the given cubic polynomial ax^3 + bx^2 + cx + d = 0
    
    Returns
    -------
    The three roots of the given cubic polynomial, where a pair of two might be complex.

    Parameters
    ----------
    a, b, c, d : mpf
        The coefficients of the polynomial to solve. These should be real.
    '''
    b /= a
    c /= a
    d /= a
    a = 1

    z0 = b / 3
    b2 = sq(b)
    p = -b2/3 + c
    q = (b/27) * (2*b2 - 9*c) + d
    r = mpmath.sqrt(-D/27 + 0j)
    u3 = (-q-r) / 2
    v3 = (-q+r) / 2
    u = mpmath.cbrt(mpmath.abs(u3)) * mpmath.sign(u3)
    v = mpmath.cbrt(mpmath.abs(v3)) * mpmath.sign(v3)
    w = u * v
    w0 = mpmath.abs(w + p/3)
    w1 = mpmath.abs(w*J + p/3)
    w2 = mpmath.abs(w*Jc + p/3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    return u + v - z0, u*J + v*Jc - z0, u*Jc + v*J - z0


def quadratic(a: mpf, b: mpf, c: mpf) -> (mpc, mpc): 
    '''Finds the roots of the given quadratic polynomial ax^2 + bx + c = 0

    Returns
    -------
    The two roots of the given quadratic polynomial, which may be a complex pair.

    Parameters
    ----------
    a, b, c : mpf
    '''
    half_b = b / 2
    half_discrim = sq(half_b) - a*c
    u1 = (-half_b - mpmath.sqrt(half_discrim)) / a
    u2 = -u1 - b/a
    return u1, u2


def sq(val: float):
    '''Takes the square of the given value
    
    Returns
    -------
    The input float squared, val*val
    
    Parameters
    ----------
    val : float
        The value to square
    '''
    return val*val