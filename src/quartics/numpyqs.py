'''
A wrapper of numpy.roots() to match the functor design pattern of other quartic solvers
'''

from collections.abc import Iterable
from mpmath import mpf, mpc
import numpy as np


MpfAble: type = float|int|str|mpf


class NumpySolver:
    def __init__(self, coeffs: Iterable[MpfAble]):
        '''
        Solves the given quartic equation using the numpy roots algorithm.
        Does not preserve the arbitrary precision of mpmath, merely wraps in mpf/mpc instances

        Parameters
        ----------
        coeffs : Iterable[MpfAble], length 5
            The coefficients of the quartic polynomial to solve, c[0]x^4 + c[1]x^3 + c[2]x^3 + c[3]x + c[4]
        '''
        if not all_instances(coeffs, MpfAble):
            raise ValueError("All coefficients must be either mpf instances, or float, int, str which can be converted thereinto")
        if not len(coeffs) == 5:
            raise ValueError("The quartic equation must be represented using 5 coefficients.")
        
        coeffs = [mpf(c) for c in coeffs]
        self._coeffs: list[mpf] = coeffs

    def __call__(self):
        '''
        Returns
        -------
        The (potentially complex) roots of the given polynomial

        Note
        ----
        These will not preserve arbitrary precision
        '''
        return [mpc(r) for r in np.roots(self._coeffs)]


def all_instances(vals: Iterable, of_type: type) -> bool:
    '''Returns whether all the given values are mpf instances (excluding mpc)'''
    return all(isinstance(val, of_type) for val in vals)