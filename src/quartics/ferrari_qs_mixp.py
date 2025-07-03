"""
Ferrari Quartic Solver (Mixed Precision)

A Ferrari/Cardano solver implementing mpmath to allow for arbitrary precision

Note
----
The numba parts have been stripped for now, as this is not yet a speed-based project

Substantially modified for case-specific use from an implementation by Nino Krvavica 
at https://github.com/NKrvavica/fqs

Originally licensed under the MIT license below:

|   MIT License
|
|   Copyright (c) 2019 NKrvavica
|
|   Permission is hereby granted, free of charge, to any person obtaining a copy
|   of this software and associated documentation files (the "Software"), to deal
|   in the Software without restriction, including without limitation the rights
|   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
|   copies of the Software, and to permit persons to whom the Software is
|   furnished to do so, subject to the following conditions:
|
|   The above copyright notice and this permission notice shall be included in all
|   copies or substantial portions of the Software.
|
|   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
|   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
|   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
|   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
|   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
|   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
|   SOFTWARE.
"""

import mpmath
from mpmath import mpf, mpc
from math import isclose


def solve_normalized_quadratic(b: mpf, 
                               c: mpf):
    ''' Analytical solver for a single quadratic equation
    (2nd order polynomial), normalized so a = 1.

    Parameters
    ----------
    a0, b0, c0: mpf
        Input data are coefficients of the Quadratic polynomial::

            a0*x^2 + b0*x + c0 = 0

    Returns
    -------
    r1, r2: tuple[mpf]
        Output data is a tuple of two roots of a given polynomial.
    '''
    assert type(b) == type(c) == mpf

    # Some repating variables
    a0 = -0.5*b
    delta = a0*a0 - c
    sqrt_delta = mpmath.sqrt(delta)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2

def one_real_root_of_normalized_cubic(b: mpf, 
                                      c: mpf, 
                                      d: mpf):
    ''' Analytical closed-form solver for a single cubic equation,
    (3rd order polynomial) where a is 1, gives only one real root.

    Parameters
    ----------
    b, c, d: mpf
        Input data are coefficients of the Cubic polynomial::

            x^3 + b*x^2 + c*x + d = 0

    Returns
    -------
    root: mpf
        Output data is a real root of a given polynomial.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    assert type(b) == type(c) == type(d) == mpf

    # Some repeating constants and variables
    third = mpf(1)/mpf(3)
    a13 = b*third
    a2 = a13*a13

    # Additional intermediate variables
    f = third*c - a2
    g = a13 * (2*a2 - c) + d
    h = 0.25*g*g + f*f*f

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        if x.real >= 0:
            return mpmath.cbrt(x)
        else:
            return -mpmath.cbrt(-x)

    if f == g == h == 0:
        return -cubic_root(d)

    elif h <= 0:
        j = mpmath.sqrt(-f)
        k = mpmath.acos(-0.5*g / (j*j*j))
        m = mpmath.cos(third*k)
        return 2*j*m - a13

    else:
        sqrt_h = mpmath.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        return S_plus_U - a13


def solve_normalized_quartic(b: mpf, 
                             c: mpf, 
                             d: mpf, 
                             e: mpf):
    ''' Analytical closed-form solver for a single quartic equation
    (4th order polynomial). Calls `single_cubic_one` and
    `single quadratic`.

    Parameters
    ----------
    b0, c0, d0, e0: mpf
        Input data are coefficients of the Quartic polynomial::

        x^4 + b*x^3 + c*x^2 + d*x + e = 0

    Returns
    -------
    r1, r2, r3, r4: tuple[mpf]
        Output data is a tuple of four roots of given polynomial.
    '''

    assert type(b) == type(c) == type(d) == type(e) == mpf

    # Some repeating variables
    qb = 0.25*b
    qb2 = qb*qb

    # Coefficients of subsidiary cubic euqtion
    p = 3*qb2 - 0.5*c
    q = b*qb2 - c*qb + 0.5*d
    r = 3*qb2*qb2 - c*qb2 + d*qb - e

    # Edge case: biquadratic
    # if isclose(q, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec+2)):
    if isclose(q, 0, abs_tol=mpmath.power(2, -24)):
        ir0, ir1 = solve_normalized_quadratic(-p*2, -r)
        r0 = mpmath.sqrt(ir0)
        r1 = -r0
        r2 = mpmath.sqrt(ir1)
        r3 = -r2
        return r0 - qb, r1 - qb, r2 - qb, r3 - qb

    assert type(p) == type(q) == type(r) == mpmath.mpf

    # One root of the cubic equation
    z0 = one_real_root_of_normalized_cubic(p, r, p*r - 0.5*q*q)
    
    s = mpmath.sqrt(2*p + 2*z0.real + 0j)
    if s == 0:
        t = z0*z0 + r
    else:
        t = -q / s

    # Compute roots by quadratic equations
    r0, r1 = solve_normalized_quadratic(s.real, z0.real + t.real)
    r2, r3 = solve_normalized_quadratic(-s.real, z0.real - t.real)

    return r0 - qb, r1 - qb, r2 - qb, r3 - qb

