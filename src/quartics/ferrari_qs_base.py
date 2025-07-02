"""
Ferrari Quartic Solver (Base)

A bare-bones copy of an existing implementation of the Ferrari-Cardano method.

Note
----
The numba parts have been stripped for now, as this is not yet a speed-based project

Modified from original by Nino Krvavica at https://github.com/NKrvavica/fqs for case-specific use.

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

import math, cmath
import numpy as np


def single_quadratic(a0, b0, c0):
    ''' Analytical solver for a single quadratic equation
    (2nd order polynomial).

    Parameters
    ----------
    a0, b0, c0: array_like
        Input data are coefficients of the Quadratic polynomial::

            a0*x^2 + b0*x + c0 = 0

    Returns
    -------
    r1, r2: tuple
        Output data is a tuple of two roots of a given polynomial.
    '''
    ''' Reduce the quadratic equation to to form:
        x^2 + ax + b = 0'''
    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = cmath.sqrt(delta)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2


def single_cubic_one(a0, b0, c0, d0):
    ''' Analytical closed-form solver for a single cubic equation
    (3rd order polynomial), gives only one real root.

    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::

            a0*x^3 + b0*x^2 + c0*x + d0 = 0

    Returns
    -------
    roots: float
        Output data is a real root of a given polynomial.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        if x.real >= 0:
            return x**third
        else:
            return -(-x)**third

    if f == g == h == 0:
        return -cubic_root(c)

    elif h <= 0:
        j = math.sqrt(-f)
        k = math.acos(-0.5*g / (j*j*j))
        m = math.cos(third*k)
        return 2*j*m - a13

    else:
        sqrt_h = cmath.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        return S_plus_U - a13

def single_quartic(a0, b0, c0, d0, e0):
    ''' Analytical closed-form solver for a single quartic equation
    (4th order polynomial). Calls `single_cubic_one` and
    `single quadratic`.

    Parameters
    ----------
    a0, b0, c0, d0, e0: array_like
        Input data are coefficients of the Quartic polynomial::

        a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0

    Returns
    -------
    r1, r2, r3, r4: tuple
        Output data is a tuple of four roots of given polynomial.
    '''

    ''' Reduce the quartic equation to to form:
        x^4 + a*x^3 + b*x^2 + c*x + d = 0'''
    b, c, d, e = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    qb = 0.25*b
    qb2 = qb*qb

    # Coefficients of subsidiary cubic euqtion
    p = 3*qb2 - 0.5*c
    q = b*qb2 - c*qb + 0.5*d
    r = 3*qb2*qb2 - c*qb2 + d*qb - e

    # One root of the cubic equation
    z0 = single_cubic_one(1, p, r, p*r - 0.5*q*q)
    # z0, z1, z2 = single_cubic(1, p, r, p*r - 0.5*q*q)
    # if abs(z1.imag) < abs(z0.imag):
    #     z0 = z1
    # if abs(z2.imag) < abs(z0.imag):
    #     z0 = z2
    # Additional variables
    s = cmath.sqrt(2*p + 2*z0.real + 0j)
    if s == 0:
        t = z0*z0 + r
    else:
        t = -q / s

    # Compute roots by quadratic equations
    r0, r1 = single_quadratic(1, s, z0 + t)
    r2, r3 = single_quadratic(1, -s, z0 - t)

    return r0 - qb, r1 - qb, r2 - qb, r3 - qb
