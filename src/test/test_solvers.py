import pytest
import math
import numpy as np
import mpmath
from mpmath import mpf, mpc
from src import solvers
from src.quartics import alg1010
from src.test.test_toroid import assert_close

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

DOUBLE_PREC = 53
QUAD_PREC = 113
HIGH_PREC = 999

mpmath.mp.prec = HIGH_PREC


# -------------------------
# Holistic tests of solvers
# -------------------------


# Quick test script to verify that solvers have consistent responses
def test_solvers_random():

    num_trials = 100
    spread = 200
    for i in range(num_trials):
        roots_np = None
        roots_fr_hp = None

        coeffs = glob_rng.normal(0, spread / 2, 4)
        coeffs = [mpf(1)] + [mpf(n) for n in coeffs]

        roots_np = solvers.calc_real_roots(coeffs, "np")
        roots_fr_hp = solvers.calc_real_roots(coeffs, "fr")
        roots_tt = solvers.calc_real_roots(coeffs, "tt")

        # Remember this test isn't checking for precision, just that the math checks out
        roots_np = [float(r) for r in sorted(roots_np)]
        roots_fr_hp = [float(r) for r in sorted(roots_fr_hp)]
        roots_tt = [float(r) for r in sorted(roots_tt)]

        assert_close(roots_np, roots_fr_hp, abs_tol=1e-6)
        assert_close(roots_np, roots_tt, abs_tol=1e-6)


# -------------------------------------------------
# Tests of individual components / helper functions
# -------------------------------------------------


def test_1010_subcubics():
    num_trials = 100
    spread = 200
    for i in range(num_trials):
        coeffs = glob_rng.normal(0, spread / 2, 2)

        roots_np = np.roots([1, 0, coeffs[0], coeffs[1]])

        coeffs = [mpf(c) for c in coeffs]

        droot = alg1010.Alg1010Solver([1,1,1,1,1])._solve_depressed_cubic(coeffs[0], coeffs[1])
        result = droot**3 + coeffs[0]*droot + coeffs[1]
        assert math.isclose(result, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec+20))

        droot_big = alg1010.Alg1010Solver([1,1,1,1,1])._solve_depressed_cubic_handleinf(coeffs[0], coeffs[1])
        result_big = droot_big**3 + coeffs[0]*droot_big + coeffs[1]
        assert math.isclose(result_big, 0, abs_tol=mpmath.power(2, -mpmath.mp.prec+20))


def test_1010_nr_abab_converges():
    '''Ensures that the nr implementation for refining alphas and betas converges when there actually is an error'''
    num_trials = 100
    coeff_spread = 200
    wrench_spread = 1.5

    for i in range(num_trials):
        coeffs = [1] + [rand for rand in glob_rng.normal(0, coeff_spread / 2, 4)]

        roots_np = np.roots(coeffs)

        coeffs = [mpf(c) for c in coeffs]
        solver = alg1010.Alg1010Solver(coeffs)
        
        roots_tt = solver()

        if not solver._used_real_abab: # NR is only used for real alphas betas
            continue
        
        solver._max_nr_abab_iters = 8 # Uncap max iters
        wrench = [mpf(rand) for rand in glob_rng.uniform(-wrench_spread, wrench_spread, 4)]
        abab_raw = solver._abab_real_raw
        abab_refined = solver._abab_real_refined
        abab_wrenched = [abab_raw[i]+wrench[i] for i in range(4)]
        abab_wrenchfined = solver._newton_raphson_abab([p for p in abab_wrenched])

        with pytest.raises(AssertionError) as assertion: # Shouldn't start close
            assert_close(abab_refined, abab_wrenched)
        assert_close(abab_refined, abab_wrenchfined) # But should end close