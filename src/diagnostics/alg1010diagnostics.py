'''
A set of brief scripts which aren't outright tests per se, but reveal specific information
about the operation of the Algorithm 1010 implementation
'''

import numpy as np
import mpmath
from mpmath import mpf, mpc
from src.quartics.alg1010 import Alg1010Solver

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

DOUBLE_PREC = 53
QUAD_PREC = 113
HIGH_PREC = 999

mpmath.mp.prec = HIGH_PREC

def nr_diagnostics() -> dict:
    '''An algorithm to return information about the newton-raphson implementation 
    for refining alphas and betas, tested with random roots'''
    num_trials = 100
    spread = 200
    
    nr_trials = []

    z_compares: list[dict] = []

    a2_compares: list[dict] = []

    for i in range(num_trials):
        coeffs = [1] + [rand for rand in glob_rng.normal(0, spread / 2, 4)]

        roots_np = np.roots(coeffs)

        coeffs = [mpf(c) for c in coeffs]
        solver = Alg1010Solver(coeffs)
        roots_tt = solver()
        
        try:
            num_trials = solver._last_nr_iter
        except:
            num_trials = None
        nr_trials.append(num_trials)
        
        z_compare = None
        a2_compare = None
        if not num_trials is None and num_trials > 0:
            try:
                raw = solver._abab_real_raw
                refined = solver._abab_real_refined
                z_compare = {
                    "before": raw,
                    "after": refined,
                    "diff": [raw[i] - refined[i] for i in range(4)]
                }
            except:
                z_compare = "z compares not found"
            try:
                a2_compare = {
                    "case I/II": solver._a2_c12,
                    "case III": solver._a2_c3
                }
            except:
                a2_compare = "a2 compares not found"
        z_compares.append(z_compare)
        a2_compares.append(a2_compare)

    return {
        "Trials": nr_trials,
        "Refinements": z_compares,
        "a2 case I/II vs III": a2_compares
    }
