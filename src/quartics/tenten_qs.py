'''An implementation of algorithm 1010 for solving quartic equations

Algorithm 1010:
Alberto Giacomo Orellana and Cristiano De Michele. 2020. Algorithm 1010: Boosting Efficiency in Solving
Quartic Equations with No Compromise in Accuracy. ACM Trans. Math. Softw. 46, 2, Article 20 (May 2020),
28 pages.
https://doi.org/10.1145/3386241

Written referencing OpenMC implementation: https://github.com/openmc-dev/openmc/blob/develop/src/external/quartic_solver.cpp
'''

from collections.abc import Iterable
import sys
import mpmath
from mpmath import mpmathify, mpf, mpc
from mpmath import sqrt, fabs as abs, power as pow, sign, chop
from math import isclose
import math


MpfAble: type = float|int|str|mpf
# Currently evaluated for double precision, TODO: should we include a method for arbitrary precision?
# cbrt(MAX_DOUBLE) / 1.618034
CUBIC_RESCAL_FACT = 3.488062113727083e+102
# pow(DBL_MAX,1.0/4.0)/1.618034;
QUART_RESCAL_FACT = 7.156344627944542e+76
MACHEPS = sys.float_info.epsilon


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
#            print("Need to normalize!")
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

    def _calc_phi0(self, scaled: bool) -> mpf:
        '''Calculates the given phi0 value for the polynomial.
        Phi0 is the dominant root of the depressed + shifted cubic from eq. 79.
        The discussion from section 2.2 of manuscript may also be relevant.'''
        a, b, c, d = self._coeffs[1:5]

        diskr = 9*sq(a) - 24*b
        # Eq. 87
        if (diskr > 0):
            diskr = sqrt(diskr) 
            # s = -2*b / (3*a + sign(a)*diskr)
            s = -2*b / (3*a + copysign(a, diskr))
        else:
            s = -a / 4
        
        # Eq. 83
        aq = a + 4*s
        bq = b + 3*s*(a + 2*s)
        cq = c + s*(2*b + s*(3*a + 4*s))
        dq = d + s*(c + s*(b + s*(a + s)))
        gg = sq(bq) / 9
        hh = aq * cq

        g = hh - 4*dq - 3*gg # Eq. 85
        h = (8*dq + hh - 2*gg)*bq/3 - sq(cq) - dq*sq(aq) # Eq. 86
        rmax = self._solve_depressed_cubic(g, h)
        if mpmath.isnan(rmax) or mpmath.isinf(rmax):
            rmax = self._solve_depressed_cubic_handleinf(g, h)
            if (mpmath.isnan(rmax) or mpmath.isinf(rmax)) and scaled:
                # Rescale again
                rfact = CUBIC_RESCAL_FACT
                rfact2 = sq(rfact)
                ggss = gg / rfact2 #TODO: Why does openmc's 1010 do these twice?
                hhss = hh / rfact2
                dqss = sq / rfact2
                aqs = aq / rfact
                bqs = bq / rfact
                cqs = cq / rfact
                ggss = sq(bqs) / mpf(9)
                hhss = aqs * cqs
                # TODO: do we need to get rid of any sq() instances here to preserve intended order of operations for precision?
                g = hhss - 4*dqss - 3*ggss
                h = (8*dqss + hhss - 2*ggss)*bqs/mpf(3) - cqs*(cqs/rfact) - (dq/rfact)*sq(aqs)
                rmax = self._solve_depressed_cubic(g, h)
                if mpmath.isnan(rmax) or mpmath.isinf(rmax):
                    rmax = self._solve_depressed_cubic_handleinf(g, h)
                rmax *= rfact
        
        #Use Newton-Raphson to refine phi0, see manuscript end of section 2.2
        x = rmax
        x2 = sq(x)
        x3 = x * x2
        gx = g * x
        f = x * (x2 + g) + h
        # TODO: ???? Do these need to be separate lines
        maxtt = max(abs(x3), abs(gx))
        if abs(h) > maxtt:
            maxtt = abs(h)
        
        if abs(f) > maxtt*MACHEPS:
            for iter_i in range(8):
                df = 3*x2 + g
                if df == 0:
                    break

                x_old = x
                x -= f/df
                f_old = f
                f = x*(x2 + g) + h
                if f == 0:
                    break
                
                if abs(f) >= abs(f_old):
                    x = x_old
                    break
        return x
    
    def _calc_err_ldlt(self, b, c, d, d2, l1, l2, l3) -> mpf:
        # Eq. 29 and 30
        err = abs(d2 + sq(l1) + 2*l3) if chop(b) == 0 else abs(((d2 + sq(l1) + 2*l3) - b) / b)
        err += abs(2*d2*l2 + 2*l1*l3) if chop(c) == 0 else abs(((2*d2*l2 + 2*l1*l3) - c) / c)
        err += abs(d2*sq(l2) + sq(l3)) if chop(d) == 0 else abs(((d2*sq(l2) + sq(l3)) - d) / d)
        return err

    def _calc_err_abcd_complex(self, a, b, c, d, aq, bq, cq, dq) -> mpf:
        '''abcd should be real, aq-dq can be complex'''
        # Eq. 68 and 69 for complex alpha1 (aq), beta1 (aq), alpha2 (cq) and beta2 (d1)
        err = abs(bq*dq) if chop(d) == 0 else abs((bq*dq - d) / d)
        err += abs(bq*cq + aq*dq) if chop(c) == 0 else abs(((bq*cq + aq*dq) - c) / c)
        err += abs(bq + aq*cq + dq) if chop(b) == 0 else abs(((bq + aq*cq + dq) - b) / b)
        err += abs(aq + cq) if a == 0 else abs(((aq + cq) - a) / a)
        return err

    def _calc_err_abcd(self, a, b, c, d, aq, bq, cq, dq) -> mpf:
        '''Where all inputs are real'''
        # Eq. 68 and 69 for real alpha1 (aq), beta1 (aq), alpha2 (cq) and beta2 (d1)
        err = abs(bq * dq) if chop(d) == 0 else abs((bq*dq - d) / d)
        err += abs(bq*cq + aq*dq) if chop(c) == 0 else abs(((bq*cq + aq*dq) - c) / c)
        err += abs(bq + aq*cq + dq) if chop(b) == 0 else abs(((bq + aq*cq + dq) - b) / b)
        err += abs(aq + cq) if chop(a) == 0 else abs(((aq + cq) - a) / a)
        return err

    def _calc_err_abc(self, a: mpf, b, c, aq: mpf, bq, cq, dq) -> mpf:
        # Eq. 48 through 51 
        err = abs(bq*cq + aq*dq) if chop(c) == 0 else abs(((bq*cq + aq*dq) - c) / c)
        err += abs(bq + aq*cq + dq) if chop(b) == 0 else abs(((bq + aq*cq + dq) - b) / b)
        err += abs(aq + cq) if chop(a) == 0 else abs(((aq + cq) - a) / a)
        return err
        
    def _newton_raphson(self, coeffs: list[mpf|mpc], roots: list[mpf|mpc]) -> list[mpf|mpc]:
        '''Refines the given list of roots for their matching coefficients.
        Defined in section 2.3 of manuscript. 
        '''
        assert all(isinstance(c, mpf|mpc) for c in coeffs)
        assert all(isinstance(r, mpf|mpc) for r in roots)

        a, b, c, d = self._coeffs[1:5]

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
#            print("fvec: ",fvec[k1],", vr: ",vr[k1])
            errf += abs(fvec[k1]) if chop(vr[k1]) == 0 else abs(fvec[k1]/vr[k1])
#            print("After adding: ",errf)
#        print("Original errf: ",errf)
        for iter_i in range(8):
#            print(f"x at start of iter {iter_i}: ",x)
            x02 = x[0] - x[2]
            det = x[1]*x[1] + x[1]*(-x[2]*x02 - mpf(2)*x[3]) + x[3]*(x[0]*x02 + x[3])
#            print("Determinant: ",det)
            if det == mpf(0): break
            Jinv: list[list[mpf|mpc]] = [[None,]*4,]*4 # You don't really need to do this in python but I want it and I don't want to figure out a whole numpy mixp setup for this
            Jinv = [[0,]*4,]*4
            Jinv = mpmath.matrix(Jinv)
            Jinv[0,0] = x02
#            print("Jinv[0,0] at start: ",Jinv[0,0])
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
#            print("Jinv[0,0] at end: ",Jinv[0,0])
#            print("Jinv: ",Jinv)
            
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
#                print("fvec: ",fvec[k1],", vr: ",vr[k1])
                errf += abs(fvec[k1]) if chop(vr[k1]) == 0 else abs(fvec[k1]/vr[k1])
#                print("After adding: ",errf)
#            print("New errf: ",errf)
                

            if (chop(errf) == 0):
                break

            if errf >= errf_old:
#                print("Converged already!")
                for k1 in range(4):
                    x[k1] = x_old[k1]
                break
            # else:
#                print("Yet to converge, iteration ",iter_i)

        # Save results
        roots.clear()
        for i in range(4):
            roots.append(x[i])
        # return locals()
        return roots

    def _solve_quadratic(self, a: mpf, b: mpf, roots: Iterable) -> Iterable[mpc]:
        diskr = sq(a) - 4*b
#        print("diskr: ",diskr)
        if (diskr >= 0):
            # sign_a = sign(a)
            # if sign_a == 0: sign_a = 1
            # div = -a - sign_a*sqrt(diskr) 
            div = -a - copysign(a, sqrt(diskr))
#            print("div: ",div)
            zmax = div / mpf(2)
            zmin = mpf(0) if chop(zmax) == 0 else b / zmax

            roots[0] = mpc(zmax)
            roots[1] = mpc(zmin)
        else:
            sqrt_d = sqrt(-diskr)
            roots[0] = mpc(-a + sqrt_d*1j) / mpf(2)
            roots[1] = mpc(-a - sqrt_d*1j) / mpf(2)
        return roots

    def _solve_normalized_quartic(self) -> list[mpc]:
        '''The central solve of the algorithm, performed on self._coeffs

        Returns 
        -------
        A list of the (potentially complex) roots of the given equation

        '''
        acx: mpc
        bcx: mpc
        ccx: mpc
        dcx: mpc
        l2m = mpmath.matrix([0,] * 12)
        d2m = mpmath.matrix([0,] * 12)
        res = mpmath.matrix([0,] * 12)
        errv = mpmath.matrix([0,] * 3)
        aqv = mpmath.matrix([0,] * 3)
        cqv = mpmath.matrix([0,] * 3)
        realcase: list[int] = [None, None]

        final_roots = [None,]*4

        # Assuming they've already been normalized
        a, b, c, d = self._coeffs[1:5]

        phi0 = self._calc_phi0(False)

        # Rescale polynomial if necessary
        rfact = mpf(1)
        if mpmath.isnan(phi0) or mpmath.isinf(phi0):
            rfact = QUART_RESCAL_FACT
            a /= rfact
            rfact2 = rfact * rfact
            b /= rfact2
            c /= rfact2*rfact
            d /= rfact2*rfact2
            self._coeffs[1:5] = a, b, c, d
            phi0 = self._calc_phi0(True) 
        
        l1 = a / 2 # Eq. 16
        l3 = b/mpf(6) + phi0/2 # Eq. 18
        del2 = c - a*l3 # Defined just after Eq. 27
        n_sol = 0 
        bl311 = 2*b/mpf(3) - phi0 - sq(l1) # d2 as defined in Eq. 20
        dml3l3 = d - sq(l3) # d3 as defined in Eq. 15 with d2 = 0
#        print(locals())

        # TODO: This section seems like it might need some revision for when to chop, with what precision, etc
        # 3 possible solutions for d2 and l2 (Eq. 28 and folowing discussion)
        if (chop(bl311) != 0):
            d2m[n_sol] = bl311
            l2m[n_sol] = del2 / (2*d2m[n_sol])
            res[n_sol] = self._calc_err_ldlt(b, c, d, d2m[n_sol], l1, l2m[n_sol], l3)
            n_sol += 1

        if (chop(del2) != 0):
            l2m[n_sol] = mpf(2) * dml3l3 / del2
            if chop(l2m[n_sol] != 0):
                d2m[n_sol] = del2 / (mpf(2)*l2m[n_sol])
                res[n_sol] = self._calc_err_ldlt(b, c, d, d2m[n_sol], l1, l2m[n_sol], l3)
                n_sol += 1
            
            d2m[n_sol] = bl311
            l2m[n_sol] = mpf(2) * dml3l3 / del2
            res[n_sol] = self._calc_err_ldlt(b, c, d, d2m[n_sol], l1, l2m[n_sol], l3)
            n_sol += 1
        
        # Pick just one l2 and d2 pair
        if n_sol == 0:
            l2 = d2 = mpf(0)
        else:
            # Pick the pair minimizing errors
            resmin = res[0]
            kmin = 0
            for k1 in range(1, n_sol):
                if res[k1] < resmin:
                    resmin = res[k1]
                    kmin = k1
            
            d2 = d2m[kmin]
            l2 = l2m[kmin]
        
#        print("d2m: ", d2m)
#        print("res: ", res)
        
        whichcase: int = 0 # Later used as an index
        # aq, bq, cq, dq # Just to clarify what variables we're about to assign to
        if d2 < 0:
            # Case I eq. 37 through 40
            gamma = sqrt(-d2)
#            print("gamma: ", gamma)
            aq = l1 + gamma
            bq = l3 + gamma*l2
            cq = l1 - gamma
            dq = l3 - gamma*l2
#            print("a1 b1 a2 b2 originally: ", aq, bq, cq, dq)

            if abs(dq) < abs(bq):
                dq = d / bq
            elif abs(dq) > abs(bq):
                bq = d / dq
            
            if abs(aq) < abs(cq):
                n_sol = 0
                if chop(dq) != 0:
                    aqv[n_sol] = (c - bq*cq) / dq # Eq. 47
                    errv[n_sol] = self._calc_err_abc(a, b, c, aqv[n_sol], bq, cq, dq)
                    n_sol += 1
                if chop(cq) != 0:
                    aqv[n_sol] = (b - dq - bq) / cq # Eq. 47
                    errv[n_sol] = self._calc_err_abc(a, b, c, aqv[n_sol], bq, cq, dq)
                    n_sol += 1
                aqv[n_sol] = a - cq # Eq. 47
                errv[n_sol] = self._calc_err_abc(a, b, c, aqv[n_sol], bq, cq, dq)
                n_sol += 1

                # Choose value of aq (alpha1 in manuscript) to minimize errors
                errmin = errv[0]
                kmin = 0
                for k in range(1, n_sol):
                    if (errv[k] < errmin):
                        kmin = k
                        errmin = errv[k]
                
                cq = cqv[kmin]
            realcase[0] = 1
        elif d2 > 0: # Should these be choperations? Probably don't need to, seeing as impl already handles "approximately zero"
            # Case II eq. 53 through 56
            gamma = sqrt(d2)
            acx = mpc(l1 + gamma*1j)
            bcx = mpc(l3 + gamma*l2*1j)
            ccx = mpmath.conj(acx)
            dcx = mpmath.conj(bcx)
            realcase[0] = 0
        else:
            realcase[0] = -1 # d2 is 0
        # Case III: d2 is 0 or approximately 0, check which solution is better
        if realcase[0] == -1 or abs(d2) < MACHEPS * (abs(mpf(2)*b/mpf(3)) + abs(phi0) + sq(l1)):
            d3 = d - sq(l3)
            err0 = mpf(0)
            if realcase[0] == 1: # I think it's possible this is a c++ typing thing and these can be condensed into one function since mpf and mpc are interchangable
                err0 = self._calc_err_abcd(a, b, c, d, aq, bq, cq, dq)
            elif realcase[0] == 0:
                err0 = self._calc_err_abcd_complex(a, b, c, d, acx, bcx, ccx, dcx)
            # aq1, bq1, cq1, dq1 # Real
            # acx1, bcx1, ccx1, dcx1 # Complwx
            err1 = mpf(0)
            if d3 <= 0:
                realcase[1] = 1
                aq1 = l1
                bq1 = l3 + sqrt(-d3)
                cq1 = l1
                dq1 = l3 - sqrt(-d3)
                if abs(dq1) < abs(bq1):
                    dq1 = d / bq1
                elif abs(dq1) > abs(bq1):
                    bq1 = d / dq1
                err1 = self._calc_err_abcd(a, b, c, d, aq1, bq1, cq1, dq1) # Eq. 68
            else:
                # i.e. complex
                realcase[1] = 0
                acx1 = l1
                bcx1 = l3 + mpc(0 + sqrt(d3)*1j)
                ccx1 = l1
                dcx1 = mpmath.conj(bcx1)
                err1 = self._calc_err_abcd_complex(a, b, c, d, acx1, bcx1, ccx1, dcx1)
            if realcase[0] == -1 or err1 < err0:
                whichcase = 1 # d2 = 0
                if realcase[1] == 1:
                    aq = aq1
                    bq = bq1
                    cq = cq1
                    dq = dq1
#                    print("a1 b1 a2 b2 after line 512ish error correction: ", aq, bq, cq, dq)
                else:
                    acx = acx1
                    bcx = bcx1
                    ccx = ccx1
                    dcx = dcx1
        if realcase[whichcase] == 1:
            # If alpha1, beta1, alpha2, and beta2 are real first refine them through a Newton-Ralphson
#            print("a1 b1 a2 b2 before refining: ", aq, bq, cq, dq)
            aq, bq, cq, dq = self._newton_raphson([a, b, c, d], [aq, bq, cq, dq])
#            print("a1 b1 a2 b2 after refining: ", aq, bq, cq, dq)
            # Finally calculate roots as roots of p1(x) and p2(x) (end of section 2.1)
            qroots = self._solve_quadratic(aq, bq, [None, None])
            final_roots[0:2] = qroots
            qroots = self._solve_quadratic(cq, dq, qroots)
            final_roots[2:4] = qroots
#            print("final roots: ",final_roots)
        else:
            # Complex coefficients of p1 and p2
            if whichcase == 0: # d2 != 0 
                cdiskr = 0.25*sq(acx) - bcx
                # Calculate roots as those of p1(x) and p2(x) (end of sec. 2.1)
                zx1 = -0.5*acx + sqrt(cdiskr)
                zx2 = -0.5*acx - sqrt(cdiskr)
                zxmax = zx1 if abs(zx1) > abs(zx2) else zx2
                zxmin = bcx / zxmax
                final_roots[0:4] = [
                    zxmin,
                    mpmath.conj(zxmin),
                    zxmax,
                    mpmath.conj(zxmax)
                ]
            else: # d2 ~= 0
                # Theoretically this path should never be reached
                cdiskr = sqrt(sq(acx) - 4*bcx)
                zx1 = -0.5 * (acx + cdiskr)
                zx2 = -0.5 * (acx - cdiskr)
                zxmax = zx1 if abs(zx1) > abs(zx2) else zx2
                zxmin = dcx / zxmax
                final_roots[0:2] = [
                    zxmax,
                    zxmin
                ]
                cdiskr = sqrt(sq(ccx) - 4*dcx)
                zx1 = -0.5 * (ccx + cdiskr)
                zx2 = -0.5 * (ccx - cdiskr)
                zxmax = zx1 if abs(zx1) > abs(zx2) else zx2
                zxmin = dcx / zxmax
                final_roots[2:4] = [
                    zxmax,
                    zxmin
                ]
        if rfact != mpf(1):
            for k in range(4):
                final_roots[k] *= rfact
        return final_roots


def sign_nz(val: MpfAble):
    '''
    More equivalent to 
    '''
    sign0 = sign(val)
    return 1 if sign0 == 0 else sign0

def copysign(sign_of: MpfAble, magn_of: MpfAble) -> mpf:
    '''
    Mimic std::copysign / math.copysign but make sure to keep it in mpf
    '''
    return mpf(math.copysign(1, mpf(sign_of))) * mpf(magn_of)

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