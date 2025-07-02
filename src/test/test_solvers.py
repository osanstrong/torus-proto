import numpy as np
import mpmath
from mpmath import mpf
from src import solvers 
from src.test import toroid_util as tu
from src.test.test_toroid import assert_close

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

DIGIT_PRECISION = 64
mpmath.mp.dps = DIGIT_PRECISION

# Quick test script to verify that solvers have consistent responses
def test_rootfinders_random():

    num_trials = 100
    spread = 200
    for i in range(num_trials):
        roots_fr = None
        roots_np = None
        roots_frnp = None

        coeffs = glob_rng.normal(0, spread / 2, 4)
        coeffs = [mpf(1)] + [mpf(n) for n in coeffs]

        roots_np = solvers.calc_real_roots_numpy(coeffs)
        roots_fr = solvers.calc_real_roots_ferrari(coeffs)
        roots_fr_hp = solvers.calc_real_roots_ferrari_highp(coeffs, dps=DIGIT_PRECISION)

        # Remember this test isn't checking for precision, just that the math checks out
        roots_fr = [float(r) for r in sorted(roots_fr)]
        roots_np = [float(r) for r in sorted(roots_np)]
        roots_fr_hp = [float(r) for r in sorted(roots_fr_hp)]

        assert_close(roots_np, roots_fr, abs_tol=1e-6)
        assert_close(roots_np, roots_fr_hp, abs_tol=1e-6)

