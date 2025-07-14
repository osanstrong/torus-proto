import math
import numpy as np
import mpmath
from mpmath import mpf, mpc
from src import solvers
from src.quartics import tenten_qs
from src.test.test_toroid import assert_close, mpfl

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

DOUBLE_PREC = 53
QUAD_PREC = 113
HIGH_PREC = 999

mpmath.mp.prec = HIGH_PREC

# Quick test script to verify that solvers have consistent responses
def test_rootfinders_random():

    num_trials = 100
    spread = 200
    for i in range(num_trials):
        roots_np = None
        roots_fr_hp = None

        coeffs = glob_rng.normal(0, spread / 2, 4)
        coeffs = [mpf(1)] + [mpf(n) for n in coeffs]

        roots_np = solvers.calc_real_roots_numpy(coeffs)
        roots_fr_hp = solvers.calc_real_roots_ferrari_highp(coeffs)
        roots_tt = solvers.calc_real_roots_1010(coeffs)

        # Remember this test isn't checking for precision, just that the math checks out
        roots_np = [float(r) for r in sorted(roots_np)]
        roots_fr_hp = [float(r) for r in sorted(roots_fr_hp)]
        roots_tt = [float(r) for r in sorted(roots_tt)]

        assert_close(roots_np, roots_fr_hp, abs_tol=1e-6)
        assert_close(roots_np, roots_tt, abs_tol=1e-6)


# Tests of individual components / helper functions


def test_1010_subcubics():
    num_trials = 100
    spread = 200
    for i in range(num_trials):
        coeffs = glob_rng.normal(0, spread / 2, 2)

        roots_np = np.roots([1, 0, coeffs[0], coeffs[1]])

        coeffs = [mpf(c) for c in coeffs]

        droot = tenten_qs.Solve1010([1,1,1,1,1])._solve_depressed_cubic(coeffs[0], coeffs[1])
        result = droot**3 + coeffs[0]*droot + coeffs[1]
        assert math.isclose(result, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec+20))

        droot_big = tenten_qs.Solve1010([1,1,1,1,1])._solve_depressed_cubic_handleinf(coeffs[0], coeffs[1])
        result_big = droot_big**3 + coeffs[0]*droot_big + coeffs[1]
        assert math.isclose(result_big, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec+20))

# def test_1010_NR():
#     '''Test the Newton Raphson implementation'''
#     num_trials = 100
#     spread = 200
#     for i in range(num_trials):
#         coeffs = glob_rng.normal(0, spread / 2, 4)

#         roots_np = np.roots([1] + [c for c in coeffs])

#         coeffs = [mpf(c) for c in coeffs]

#         roots_iter = [mpc(r) for r in roots_np]
        
#         minroot = roots_iter[0]
#         maxroot = roots_iter[-1]

#         delta = 1.5

#         miniter = minroot - delta
#         maxiter = maxroot + delta

#         roots_iter[0] = miniter
#         roots_iter[-1] = maxiter

#         # Make sure it converges upon to the original root
#         for iter in range(10):
#             roots_iter = tenten_qs.Solve1010([mpf(1)]+coeffs)._newton_raphson(coeffs, roots_iter)
        
#         assert math.isclose(roots_iter[0].real, minroot.real)
#         assert math.isclose(roots_iter[-1].real, maxroot.real)
        
    