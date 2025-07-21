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
    return_logs: bool = False,
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
            # log = []
            # sys.settrace(get_tracer_to_list(log))
            tracer = AlgTracer(solv)
            tracer.begin()
            dtb = tor.distance_to_boundary(ray_src, ray_dir, solv)
            tracer.end()
            logs.append(tracer.simple_logstring())
            # logs.append(tracer.everything())
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


class AlgTracer():
    '''
    A class for tracing, such that an instance can be passed to sys.settrace().

    Usage:
    tracer = AlgTracer(namespace)
    tracer.begin()
    foo.bar()
    tracer.end()
    print(tracer.simple_logstring)
    '''

    def __init__(self, namespace):
        '''
        Parameters
        ----------
        namespace : something you can call dir() on
            The namespace of functions to trace. For instance, if the class
            Alg1010Solver were given, it would only trace functions defined
            there, such as _solve_normalized_quartic() or _calc_err_ldlt().
        '''
        self.namespace = namespace
        pass

    def begin(self):
        self._logs: list = []
        self._indent: int = 0
        sys.settrace(self)

    def end(self):
        sys.settrace(None)

    def __call__(self, frame, event, arg = None):
        if not event in ["call", "return"]:
            return self
        code = frame.f_code
        func_name = code.co_name
        line_no = frame.f_lineno
        if not func_name in dir(self.namespace):
            return self
        if func_name.startswith("__"):
            return self

        if event == "call":
            params = []
            for i in range(code.co_argcount):
                name = code.co_varnames[i]
                val = frame.f_locals[name]
                params.append(f"{name}:{val}")
            content = {
                "params": params
            }
        else:
            self._indent -= 1
            content = {
                "locals": frame.f_locals,
                "returned": arg
            }
        self._logs.append({
            "event": event,
            "name": func_name,
            "indent": self._indent,
            "content": content
        })
        if event == "call":
            self._indent += 1
        return self

    def simple_logstring(self, indent: str = "  "):
        return "\n".join([
            indent*e["indent"] + \
                (f"{e["name"]}({", ".join(e["content"]["params"][1:])})" if e["event"] == "call" \
            else f"-> {e["content"]["returned"]}") \
            for e in self._logs
        ])
        
