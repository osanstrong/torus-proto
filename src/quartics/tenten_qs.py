'''An implementation of algorithm 1010 for solving quartic equations

Algorithm 1010:
Alberto Giacomo Orellana and Cristiano De Michele. 2020. Algorithm 1010: Boosting Efficiency in Solving
Quartic Equations with No Compromise in Accuracy. ACM Trans. Math. Softw. 46, 2, Article 20 (May 2020),
28 pages.
https://doi.org/10.1145/3386241

Written referencing OpenMC implementation: https://github.com/openmc-dev/openmc/blob/develop/src/external/quartic_solver.cpp
'''

import mpmath
from mpmath import mpmathify, mpf, mpc
from mpmath import sqrt, fabs as abs, power as pow, sign, chop
from math import isclose


MpfAble: type = float|int|str|mpf


class Solve1010:
    def __init__(self, coeffs: list[MpfAble]):
        '''
        Parameters
        ----------
        coeffs : list[MpfAble]
            The coefficients of the quartic polynomial to solve, c[0]x^4 + c[1]x^3 + c[2]x^3 + c[3]x + c[4]
        '''
        if not all(isinstance(c, MpfAble) for c in coeffs):
            raise ValueError("All coefficients must be either mpf instances, or float, int, str which can be converted thereinto")
        if not len(coeffs) in [4,5]:
            raise ValueError("Coefficients must either be a full 5 for a quartic, or the last 4 of one that has already been normalized (a=1)")
        
        coeffs = [mpf(c) for c in coeffs]

        if len(coeffs) == 5:
            print("Need to normalize!")
            a = coeffs[0]
            coeffs = [coeffs[i]/a for i in range(len(coeffs))]
        else:
            coeffs.insert(0,mpf(1))

        self._coeffs: list[mpf] = coeffs

    def __call__(self, *args, **kwds):
        return self._solve_normalized_quartic()
    
    def _solve_depressed_cubic_handleinf(self, b: mpf, c: mpf) -> mpf|mpc:
        '''Returns the dominant root of the depressed cubic x^3 + bx + c, where b & c are large
        See Section 2.2 of 1010 manuscript
        '''
        assert type(b) == type(c) == mpf
        q = -b / mpf(3)
        r = 0.5 * c
        if (r == 0):
            # x^3 + bx = 0
            if b <= 0:
                return sqrt(b)
            else:
                return 0
        
        if abs(q) < abs(r):
            qr = q / r
            qr2 = sq(qr)
            kk = mpf(1) - q*qr2
        else:
            rq = r / q
            kk = sign(q) * (rq*rq/q - mpf(1))

        if kk < 0:
            sqrt_q = sqrt(q)
            theta = mpmath.acos((r/abs(q)) / sqrt_q)
            
            if mpf(2)*theta < mpmath.pi:
                return mpf(-2) * sqrt_q * mpmath.cos(theta / mpf(3))
            else:
                return mpf(-2) * sqrt_q * mpmath.cos((theta + mpf(2)*mpmath.pi) / mpf(3))
        else:
            if abs(q) < abs(r):
                a = -sign(r) * cbrt(abs(r) * (mpf(1)+sqrt(kk)))
            else:
                a = -sign(r) * cbrt(
                    abs(r) + sqrt(abs(q))*abs(q)*sqrt(kk)
                )
            if a == 0: #TODO: Replace with isclose calls?
                b = 0
            else: 
                b = q / a
            return a + b
    
    def _solve_depressed_cubic(self, b: mpf, c: mpf) -> mpf|mpc:
        '''Returns the dominant root of the depressed cubic x^3 + bx + c
        See Section 2.2 of 1010 manuscript
        '''
        assert type(b) == type(c) == mpf
        q = -b / mpf(3)
        r = 0.5 * c

        if abs(q) > 1e102 or abs(r) > 1e154:
            return self._solve_depressed_cubic_handleinf(b, c)

        q3 = q*q*q
        r2 = sq(r)

        if r2 < q3:
            theta = mpmath.acos(r / sqrt(q3))
            m_sqrt_q = mpf(-2) * sqrt(q) # m for modified because it's not literally the sqrt

            if mpf(2)*theta < mpmath.pi:
                return m_sqrt_q * mpmath.cos(theta / mpf(3))
            else:
                return m_sqrt_q * mpmath.cos((theta + mpf(2)*mpmath.pi) / mpf(3))
        else:
            a = -sign(r) * mpmath.cbrt(abs(r) + sqrt(r2 - q3))
            if a == 0: #TODO: Should this be an isclose call instead?
                b = 0
            else: 
                b = q / a
            return a + b
        
    def _newton_raphson(self, coeffs: list[mpf|mpc], roots: list[mpf|mpc]):
        '''Refines the given list of roots for their matching coefficients.
        Defined in section 2.3 of manuscript. 
        '''
        assert all(isinstance(c, mpf|mpc) for c in coeffs)
        assert all(isinstance(r, mpf|mpc) for r in roots)

        a, b, c, d = coeffs

        # It might be unnecessary to go this explicit in making copies
        x = [mpmathify(root) for root in roots]
        vr = [mpmathify(coeffs[i]) for i in [3, 2, 1, 0]]
        fvec = [
            x[1]*x[3] - d,
            x[1]*x[2] + x[0]*x[3] - c,
            x[1] + x[0]*x[2] + x[3] - b,
            x[0] + x[2] - a
        ]

        errf = mpf(0)
        for k1 in range(4):
            #TODO: Should this be an isclose operation?
            print("fvec: ",fvec[k1],", vr: ",vr[k1])
            errf += abs(fvec[k1]) if chop(vr[k1]) == 0 else abs(fvec[k1]/vr[k1])
            print("After adding: ",errf)
        print("Original errf: ",errf)
        for iter_i in range(8):
            print(f"x at start of iter {iter_i}: ",x)
            x02 = x[0] - x[2]
            det = x[1]*x[1] + x[1]*(-x[2]*x02 - mpf(2)*x[3]) + x[3]*(x[0]*x02 + x[3])
            print("Determinant: ",det)
            if det == mpf(0): break
            Jinv: list[list[mpf|mpc]] = [[None,]*4,]*4 # You don't really need to do this in python but I want it and I don't want to figure out a whole numpy mixp setup for this
            Jinv = [[0,]*4,]*4
            Jinv = mpmath.matrix(Jinv)
            Jinv[0,0] = x02
            print("Jinv[0,0] at start: ",Jinv[0,0])
            Jinv[0,1] = x[3] - x[1]
            Jinv[0,2] = x[1] * x[2] - x[0] * x[3]
            Jinv[0,3] = -x[1] * Jinv[0,1] - x[0] * Jinv[0,2]
            Jinv[1,0] = x[0] * Jinv[0,0] + Jinv[0,1]
            Jinv[1,1] = -x[1] * Jinv[0,0]
            Jinv[1,2] = -x[1] * Jinv[0,1]
            Jinv[1,3] = -x[1] * Jinv[0,2]
            Jinv[2,0] = -Jinv[0,0]
            Jinv[2,1] = -Jinv[0,1]
            Jinv[2,2] = -Jinv[0,2]
            Jinv[2,3] = Jinv[0,2] * x[2] + Jinv[0,1] * x[3]
            Jinv[3,0] = -x[2] * Jinv[0,0] - Jinv[0,1]
            Jinv[3,1] = Jinv[0,0] * x[3]
            Jinv[3,2] = x[3] * Jinv[0,1]
            Jinv[3,3] = x[3] * Jinv[0,2]
            print("Jinv[0,0] at end: ",Jinv[0,0])
            print("Jinv: ",Jinv)
            
            dx = mpmath.matrix([0,]*4)
            for k1 in range(4):
                for k2 in range(4):
                    dx[k1] += Jinv[k1,k2] * fvec[k2]

            x_old = [mpmathify(xi) for xi in x]
            
            for k1 in range(4): 
                x[k1] -= dx[k1]/det

            fvec[0] = x[1]*x[3] - d
            fvec[1] = x[1]*x[2] + x[0]*x[3] - c
            fvec[2] = x[1] + x[0]*x[2] + x[3] - b
            fvec[3] = x[0] + x[2] - a

            errf_old = errf
            errf = mpf(0)
            for k1 in range(4):
                print("fvec: ",fvec[k1],", vr: ",vr[k1])
                errf += abs(fvec[k1]) if chop(vr[k1]) == 0 else abs(fvec[k1]/vr[k1])
                print("After adding: ",errf)
            print("New errf: ",errf)
                

            if (chop(errf) == 0):
                break

            if errf >= errf_old:
                print("Converged already!")
                for k1 in range(4):
                    x[k1] = x_old[k1]
                break
            else:
                print("Yet to converge, iteration ",iter_i)

        # Save results
        roots.clear()
        for i in range(4):
            roots.append(x[i])
        return locals()
        return roots

    def _solve_normalized_quartic(self) -> list[mpf|mpc]:
        '''The central solve of the algorithm, performed on self._coeffs

        Returns 
        -------
        A list of the (potentially complex) roots of the given equation

        '''


def sq(val: mpf):
    return val*val


def cbrt(val: mpc):
    '''
    Return cube root while maintaining sign
    '''
    if val.real >= 0:
        return mpmath.cbrt(val)
    else:
        return -mpmath.cbrt(-val)               