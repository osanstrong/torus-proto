'''
A script to compare and plot results of different solvers
'''

import sys
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import pandas as pd
from mpmath import mpf, mp, pi, power
from src.toroid import EllipticToroid
import src.analysis.ray_generator as rg
from src.solvers import get_solver, calc_real_roots
import src.quartics.alg1010 as alg1010


# ----------------
# Useful constants
# ----------------


DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999
MpfAble: type = mpf|float|str


# -----------
# Comparisons
# -----------


# 1: Compare intersection point at different distances for different solvers
def compare_distances(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-16, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    return_dtb: bool = False,
) -> dict:
    dists = [mpf(d) for d in dists]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u, v, dist = 1)[1]

    # Combos of dists and solvers
    sources = [rg.get_normal_ray(tor, u, v, dist = d)[0] for d in dists]
    solver_names = ["fr", "tt", "np"]
    test_solvers = [get_solver(s) for s in solver_names]

    # Final results include each dist and every solver, as well as arbitrarily high precision just to be sure
    final_results: dict = {}
    final_results["dists"] = dists
    # Also include the polynomial for diagnostics
    polynoms = [tor._ray_intersection_polynomial(ray_src, ray_dir) for ray_src in sources]
    final_results["polyn"] = polynoms

    # Get results at ludicrous precision just to check
    mp.prec = HIGH_PREC
    highp_results = []
    for i in range(len(dists)):
        ray_src = sources[i]
        dtb = tor.distance_to_boundary(ray_src, ray_dir, "tt")
        hit = ray_src + ray_dir*dtb
        highp_results.append(dtb if return_dtb else hit[2]) #Check just z coordinate for now?
    final_results["highp"] = highp_results

    #Then for each solver, repeat at normal precision
    mp.prec = prec
    for s in range(len(test_solvers)):
        solv = test_solvers[s]
        name = solver_names[s]
        results = []
        results_manual = []
        logs = []
        for i in range(len(sources)):
            ray_src = sources[i]
            polynom = polynoms[i]
            log = []
            sys.settrace(get_tracer_to_list(log))
            dtb = tor.distance_to_boundary(ray_src, ray_dir, solv)
            sys.settrace(None)
            logs.append(log)
            if dtb is None:
                results.append(None)
                continue
            hit = ray_src + ray_dir*dtb
            results.append(dtb if return_dtb else hit[2])
            
            man_solv = solv(polynom)
            man_roots = man_solv()
            man_roots = calc_real_roots(polynom, solv)
            results_manual.append(man_roots)
        final_results[name] = results
        final_results[name+'_man'] = results_manual
        final_results[name+'_log'] = logs
    
    return final_results
    

def to_db(d: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(d, orient="index").transpose()


def print_as_db(d: dict):
    print(to_db(d))




def get_tracer_to_list(l: list):
    '''Returns a new tracer instance, which logs certain key details to the given list.
    
    Logs 
    '''
    def tracer(frame, event, arg = None):
        code = frame.f_code
        func_name = code.co_name
        line_no = frame.f_lineno

        log = f"{line_no}:{func_name}:{event}"
        
        if func_name.startswith("__"):
            return tracer
        # if any([
        #     func_name.startswith("__"),
        #     func_name.startswith("gmpy"),
        #     func_name.startswith("mpf"),
        #     func_name in ["<genexpr>", "from_int", "to_float", "sq", 
        #     "l2norm2", "from_float", "convert", "make_mpf", "nthroot_fixed",
        #     "_mpf_", "to_mpfs", ""]
        # ]):
        #     return tracer
        #     l.append(log)
        if event == "return" and func_name in dir(alg1010.Alg1010Solver):
            locs = frame.f_locals
            locsstr = f":{locs}"
            log = log + locsstr + f":{arg}"

            log = {
                "event":f"{line_no}:{func_name}:{event}",
                "locals":locs,
                "returned":arg
            }
            l.append(log)
        return tracer
    return tracer