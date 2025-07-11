import numpy as np
import mpmath
from mpmath import mpf
from src import solvers
from src.test.test_toroid import assert_close

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

FLOAT_PRECISION = 256 + (53 - 64) # Definitely not how it works, but I like keeping it just short of a power of two, like a 64-bit float having 53 bits of precision
mpmath.mp.prec = FLOAT_PRECISION

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
        roots_fr_hp = solvers.calc_real_roots_ferrari_highp(coeffs)

        # Remember this test isn't checking for precision, just that the math checks out
        roots_np = [float(r) for r in sorted(roots_np)]
        roots_fr_hp = [float(r) for r in sorted(roots_fr_hp)]

        assert_close(roots_np, roots_fr_hp, abs_tol=1e-6)

