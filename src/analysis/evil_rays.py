'''
A script to compare and plot results of different solvers
'''

import sys
from collections.abc import Iterable
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import pandas as pd
from mpmath import mpf, matrix, mp, pi, power
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
    rays = [(src, ray_dir) for src in sources]
    results = compare_intersections(
        tor,
        rays,
        base_prec=prec
    )
    results["dists"] = dists
    return results


# 2: Compare a grazing ray at different distances
def compare_grazes_by_distance(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-16, 12)],
    u: mpf = 0, v: mpf = pi,
    ep: mpf = 0,
    return_logs: bool = True
) -> dict:
    dists = [mpf(d) for d in dists]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_grazing_ray(tor, u=u, v=v, distance = 1)[1]

    # Combos of dists and solvers
    sources = [rg.get_grazing_ray(tor, u=u, v=v, distance = d, pos_epsilon=ep)[0] for d in dists]
    rays = [(src, ray_dir) for src in sources]
    results = compare_intersections(
        tor,
        rays,
        get_result=get_first_intersection
    )
    results["dists"] = dists
    return results


def get_first_intersection(tor, ray, slv):
    inters = tor.ray_intersection_points(ray[0], ray[1], slv)
    if inters:
        return inters[0]
    else:
        return None


def get_first_intersection_z(tor, ray, slv):
    inters = tor.ray_intersection_points(ray[0], ray[1], slv)
    if inters:
        return inters[0][2]
    else:
        return None


def compare_intersections(
    toroids: EllipticToroid|Iterable[EllipticToroid], 
    rays: tuple[matrix,matrix]|Iterable[tuple[matrix, matrix]],
    solver_names: Iterable[str] = ["fr", "tt", "np", "fr_hp", "tt_hp"],
    get_result: callable = get_first_intersection_z,
    base_prec: int = DOUBLE_PREC,
    high_prec: int = HIGH_PREC,
) -> dict:
    '''
    Returns information about the result of every combination of the given toroids and rays.
    Typically, one would iterate over either of these at a time, to have a single variable comparison of some kind.
    
    Parameters
    ----------
    toroids : EllipticToroid | Iterable[EllipticToroid]
        A series of toroids to compare, or just one toroid, if toroids aren't being compared.
    rays : tuple[matrix, matrix] | Iterable[tuple[matrix, matrix]]
        A series of rays to compare, or just one, if rays aren't what's being compared.
        Ordered (source, direction)
    solver_names: Iterable[str], default ["fr", "tt", "np", "fr_hp", "tt_hp"]
        The names of the solvers to compare, with "_hp" suffixed to ones to be run in high precision.

    Returns
    -------
    A dictionary with results of each sequential combination of toroids and rays.
    '''
    solver_precs = [HIGH_PREC if name.endswith("_hp") else DOUBLE_PREC for name in solver_names]
    base_names = [name[:-3] if name.endswith("_hp") else name for name in solver_names]
    solver_funcs = [get_solver(name) for name in base_names]

    if isinstance(toroids, EllipticToroid):
        toroids = [toroids]
    if isinstance(rays[0], matrix): # I.e. it's only one pair of vectors, and not a list of pairs
        rays = [rays]

    mp.prec = HIGH_PREC
    final_results = {
        "rays":rays,
        "polynomials":[tor._ray_intersection_polynomial(ray[0],ray[1]) for tor in toroids for ray in rays]
    }
    for i in range(len(solver_names)):
        name = solver_names[i]
        prec = solver_precs[i]
        func = solver_funcs[i]

        mp.prec = prec
        
        slv_results = []
        slv_logs = []
        for tor in toroids:
            for ray in rays:
                tracer = AlgTracer(func)
                tracer.begin()
                result = get_result(tor, ray, func)
                tracer.end()

                slv_results.append(result)
                slv_logs.append(tracer.simple_logstring())
        
        final_results[name] = slv_results
        final_results[name+"_logs"] = slv_logs
    return final_results
    

def to_db(d: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(d, orient="index").transpose()


def print_as_db(d: dict):
    print(to_db(d))


def get_serialized_mpf(val: mpf) -> str:
    return repr(val)


def serialize_iter_mpfs(l: Iterable):
    for i in range(len(l)):
        item = l[i]
        if isinstance(item, mpf):
            l[i] = get_serialized_mpf(item)
        elif isinstance(item, matrix):
            l[i] = [get_serialized_mpf(n) for n in item]
        # elif isinstance(item, Iterable):
        #     if isinstance(item, tuple):
        #         l[i] = [n for n in item]
        #         item = l[i]
        #     serialize_iter_mpfs(item)
        # if isinstance(item, dict):
        #     serialize_dict_mpfs(item)
        # elif isinstance(item, Iterable):
        #     serialize_iter_mpfs(item)
        # elif isinstance(item, mpf):
        #     l[i] = get_serialized_mpf(item)


def serialize_dict_mpfs(d: dict):
    for key in d:
        item = d[key]
        if isinstance(item, dict):
            serialize_dict_mpfs(item)
        elif isinstance(item, Iterable):
            if isinstance(item, tuple):
                d[key] = [i for i in item]
                item = d[key]
            serialize_iter_mpfs(item)
        elif isinstance(item, mpf):
            d[key] = get_serialized_mpf(item)


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
        
