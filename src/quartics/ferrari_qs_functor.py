"""
Ferrari Quartic Solver (Arbitrary Precision)

A Ferrari/Cardano solver implementing mpmath to allow for arbitrary precision

Note
----
Created with reference to https://github.com/NKrvavica/fqs, whose MIT License statement,
should it be necessary, is included at the end of the file.
"""

from collections.abc import Iterable
import mpmath
from mpmath import mpf, mpc
from math import isclose


MpfAble: type = float|int|str|mpf


class SolveFerrari:
    def __init__(self, coeffs: Iterable[MpfAble]):
        '''
        Solves the given quartic equation using the Ferrari-Cardano algorithm.
        
        Parameters
        ----------
        coeffs : Iterable[MpfAble], length 5
            The coefficients of the quartic polynomial to solve, c[0]x^4 + c[1]x^3 + c[2]x^3 + c[3]x + c[4]
        '''
        if not all(isinstance(c, MpfAble) for c in coeffs):
            raise ValueError("All coefficients must be either mpf instances, or float, int, str which can be converted thereinto")
        if not len(coeffs) == 5:
            raise ValueError("The quartic equation must be represented using 5 coefficients.")
        
        coeffs = [mpf(c) for c in coeffs]

        a = coeffs[0]
        if not a == 1:
            # Coefficients must be normalized
            coeffs = [coeffs[i]/a for i in range(len(coeffs))]

        self._coeffs: list[mpf] = coeffs

    def __call__(self, *args, **kwds) -> list[mpc]:
        '''
        Returns
        -------
        A list of (potentially complex) roots of the given quartic polynomial, as determined by the Ferrari-Cardano method 
        '''        
        return self._solve_normalized_quartic()

    # Helper Functions

    def _solve_normalized_quartic(self):
        '''
        The core function of the functor; solves the stored normalized quadratic equation.

        Returns
        -------
        The (potentially complex) roots of the quartic polynomial stored in the functor.
        Assumes that this polynomial has already been normalized such that self._coeffs[0] = 1
        '''
        b, c, d, e = self._coeffs[1:5]
        assert type(b) == type(c) == type(d) == type(e) == mpf
        # 1/4 of b, because it comes up a lot
        qb = 0.25*b
        qb2 = sq(qb)

        # Subsidiary cubic equation
        p = 3*qb2 - 0.5*c
        q = b*qb2 - c*qb + 0.5*d
        r = 3*qb2*qb2 - c*qb2 + d*qb - e

        # Edge case: equation is biquadratic
        if isclose(q, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec)):
            ir0, ir1 = self._solve_normalized_quadratic(-2*p, -r) 
            r0 = mpmath.sqrt(ir0)
            r1 = -r0
            r2 = mpmath.sqrt(ir1)
            r3 = -r2
            return r0 - qb, r1 - qb, r2 - qb, r3 - qb
        
        # One real zero of subsidiary cubic
        z0 = self._one_real_root_of_normalized_cubic(p, r, p*r - 0.5*sq(q))

        s = mpmath.sqrt(2*p + 2*z0.real + 0j)
        if s == 0:
            t = z0*z0 + r
        else:
            t = -q / s

        # Find shifted roots using quadratic equations    
        r0, r1 = self._solve_normalized_quadratic(s.real, z0.real + t.real)
        r2, r3 = self._solve_normalized_quadratic(-s.real, z0.real - t.real)
        # Shift roots back to x
        return r0 - qb, r1 - qb, r2 - qb, r3 - qb

    def _solve_normalized_quadratic(self, b, c) -> tuple[mpc]:
        '''
        Solves a quadratic equation ax^2 + bx + c = 0, where coefficients are scaled so a = 1.

        Parameters
        ----------
        b, c : mpf
            The coefficients of the quadratic x^2 + bx + c = 0

        Returns
        -------
        The (potentially complex) roots of the given quadratic
        '''
        assert type(b) == type(c) == mpf

        # Predivide by 2
        b0 = -0.5*b
        discrim = b0*b0 - c
        sqrt_discrim = mpmath.sqrt(discrim)

        # Roots
        r1 = b0 - sqrt_discrim
        r2 = b0 + sqrt_discrim

        return r1, r2
    
    def _one_real_root_of_normalized_cubic(self, b, c, d) -> mpf:
        '''
        Solves for a single real root of the given normalized cubic equation.

        Parameters
        ----------
        b, c, d : mpf
            The coeffients of the normalized cubic equation x^3 + bx^2 + cx + d = 0

        Returns
        -------
        A real root for the given cubic equation, since at least one of the three must be real.
        '''
        assert type(b) == type(c) == type(d) == mpf

        # Repeating values
        third = mpf(1)/mpf(3)
        third_b = b*third
        third_b2 = sq(third_b)

        # Intermediate variables
        f = third*c - third_b2
        g = third_b * (2*third_b2 - c) + d
        h = 0.25*sq(g) + f**3

        if f == g == h == mpf(0): #Should this be converted to closeness test?
            return -cbrt(d)
        elif h <= 0:
            j = mpmath.sqrt(-f)
            k = mpmath.acos(-0.5*g / (j*j*j))
            m = mpmath.cos(third*k)
            return 2*j*m - third_b

        else:
            sqrt_h = mpmath.sqrt(h)
            S = cbrt(-0.5*g + sqrt_h)
            U = cbrt(-0.5*g - sqrt_h)
            S_plus_U = S + U
            return S_plus_U - third_b


def sq(val: mpc):
    '''
    Squares the given number.

    Parameters
    ----------
    val: mpc
        The value to square. May be complex

    Returns
    -------
    The square of the given value
    '''
    return val*val


def cbrt(val: mpc):
    '''
    Returns the cube root of a number, preserving its original sign.

    Parameters
    ----------
    val: mpc
        The value, which may be complex, to take the cube root of.
    
    Returns
    -------
    The cube root of the given value, while maintaining its sign.
    '''
    if val.real >= 0:
        return mpmath.cbrt(val)
    else:
        return -mpmath.cbrt(-val)


'''
The Ferrari-Cardano algorithm dates back to 1545, but this specific implementation was created
referencing a comparable python implementation at https://github.com/NKrvavica/fqs. In case it
is required, their MIT License statement is included below:

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
'''