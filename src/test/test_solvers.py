import numpy as np
import mpmath
from mpmath import mpf
from src import solvers 
from src.test.test_toroid import assert_close

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

# Quick test script to verify that solvers have consistent responses
def test_rootfinders_random():

    num_trials = 10000
    min_power = -48
    max_power = 48
    for i in range(num_trials):
        roots_fr = None
        roots_np = None
        roots_frnp = None
        # coeff_exps = glob_rng.uniform(min_power, max_power, 4)
        coeff_exps = glob_rng.normal(0, max_power / 2, 4)
        coeff_exps = [mpf(1)] + [mpf(n) for n in coeff_exps]

        roots_np = solvers.calc_real_roots_numpy(coeff_exps)
        # roots_fr, root_locals = toroid_util.real_roots_ferrari(coeff_exps)
        roots_fr = solvers.calc_real_roots_ferrari(coeff_exps)
        # roots_frnp, root_locals_np = toroid_util.real_roots_ferrari_npdebug(coeff_exps)

        roots_fr = [float(r) for r in sorted(roots_fr)]
        roots_np = [float(r) for r in sorted(roots_np)]
        assert_close(roots_np, roots_fr, abs_tol=1e-6)