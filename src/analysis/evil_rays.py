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


# =--------------=
# Useful constants
# =--------------=


DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999
MpfAble: type = mpf|float|str


# =------------------=
# Experimental Presets
# =------------------=

# ITER_TOROID_CM: EllipticToroid = EllipticToroid() #Toroid the scale of ITER, though ITER is a different kind
# ITER_OUTER_DIVERTOR_CM: EllipticToroid = EllipticToroid() #Toroid matching the outer curved section of the ITER divertor; Has more extreme difference in dimensions


# =---------------=
# Final experiments
# =---------------=


def epsilon_avoid_backcollision():
    '''
    How close can a ray start to the surface of a toroid and reliably
    avoid intersecting with it. (While moving away)

    Cases considered for rays normal to the surface, and rays tangential
    to the surface.
    '''
    # First, normal rays
    special_uvs = [] 
    random_uvs = []
    # Iterate through different distances to try and find 
    full_results: dict = {}
    passfail_history: list[bool] = []

    epsilons = [power(10, i) for i in range] #Start with a given series
    # full_results = uvrand_normals(dists=epsilons, uv_count=)

    pass


# =--------------------------------------------------------=
# Experimental methods, mostly toying around in this section
# =--------------------------------------------------------=


def get_first_intersection(tor, ray_src, ray_dir, slv):
    inters = tor.ray_intersection_points(ray_src, ray_dir, slv)
    if inters:
        return inters[0]
    else:
        return None


# -----------
# Comparisons
# -----------


# 1: Compare intersection point at different distances for different solvers
def compare_normals_by_distances(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-16, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    return_dtb: bool = False,
    return_logs: bool = False,
    result_func: callable = get_first_intersection
) -> dict:
    dists = [mpf(d) for d in dists]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u, v, dist = 1)[1]

    # Combos of dists and solvers
    sources = [rg.get_normal_ray(tor, u, v, dist = d)[0] for d in dists]
    setups = [(tor, src, ray_dir) for src in sources]
    results = compare_intersections(
        setups,
        base_prec=prec,
        get_result=result_func
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
    setups = [(tor, src, ray_dir) for src in sources]
    results = compare_intersections(
        setups,
        get_result=get_first_intersection
    )
    results["dists"] = dists
    return results


# 3: Compare normals by distances by scale
def compare_normals_by_distance_by_scale(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-4, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    scales: list = [10**i for i in range(-3, 4)],
) -> dict:
    dists = [mpf(d) for d in dists]
    scales = [mpf(s) for s in scales]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u=u, v=v, dist=1)[1]

    toroids = [EllipticToroid(tor.tor_rad*s, tor.hor_rad*s, tor.ver_rad*s) for s in scales]
    sources = [rg.get_normal_ray(t, u=u, v=v, dist=d)[0] for t in toroids for d in dists]
    toroids = [t for t in toroids for d in dists]
    setups = [(toroids[i], sources[i], ray_dir) for i in range(len(sources))]
    results = compare_intersections(
        setups,
        get_result=get_first_intersection
    )
    results["dists"] = [d for s in scales for d in dists]
    results["scale"] = [s for s in scales for d in dists]
    return results

    
# 4: Compare normals by distances by scale of major radius
def compare_normals_by_distance_by_scale_r(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-4, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    scales: list = [10**i for i in range(-3, 4)],
) -> dict:
    dists = [mpf(d) for d in dists]
    scales = [mpf(s) for s in scales]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u=u, v=v, dist=1)[1]

    toroids = [EllipticToroid(max(tor.tor_rad*s, tor.hor_rad), tor.hor_rad, tor.ver_rad) for s in scales]
    sources = [rg.get_normal_ray(t, u=u, v=v, dist=d)[0] for t in toroids for d in dists]
    toroids = [t for t in toroids for d in dists]
    setups = [(toroids[i], sources[i], ray_dir) for i in range(len(sources))]
    results = compare_intersections(
        setups,
        get_result=get_first_intersection
    )
    results["dists"] = [d for s in scales for d in dists]
    results["scale"] = [s for s in scales for d in dists]
    return results


def compare_normals_by_distance_by_scale_b(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-4, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    scales: list = [10**i for i in range(-3, 4)],
) -> dict:
    dists = [mpf(d) for d in dists]
    scales = [mpf(s) for s in scales]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u=u, v=v, dist=1)[1]

    toroids = [EllipticToroid(tor.tor_rad, tor.hor_rad, tor.ver_rad*s) for s in scales]
    sources = [rg.get_normal_ray(t, u=u, v=v, dist=d)[0] for t in toroids for d in dists]
    toroids = [t for t in toroids for d in dists]
    setups = [(toroids[i], sources[i], ray_dir) for i in range(len(sources))]
    results = compare_intersections(
        setups,
        get_result=get_first_intersection
    )
    results["dists"] = [d for s in scales for d in dists]
    results["scale"] = [s for s in scales for d in dists]
    return results


# 6: Compare normals by distances by scaling r and a simultaneously
def compare_normals_by_distance_by_scale_ra(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-4, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    scales: list = [10**i for i in range(-3, 4)],
) -> dict:
    dists = [mpf(d) for d in dists]
    scales = [mpf(s) for s in scales]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u=u, v=v, dist=1)[1]

    toroids = [EllipticToroid(tor.tor_rad*s, tor.hor_rad*s, tor.ver_rad) for s in scales]
    sources = [rg.get_normal_ray(t, u=u, v=v, dist=d)[0] for t in toroids for d in dists]
    toroids = [t for t in toroids for d in dists]
    setups = [(toroids[i], sources[i], ray_dir) for i in range(len(sources))]
    results = compare_intersections(
        setups,
        get_result=get_first_intersection
    )
    results["dists"] = [d for s in scales for d in dists]
    results["scale"] = [s for s in scales for d in dists]
    return results


# 7: Compare intersection point at different distances for different solvers across the whole range of u and v
def uvspread_normals(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(4, 12)],
    prec: int = DOUBLE_PREC,
) -> dict:
    u_count = v_count = 5
    us = [i*2*mp.pi / mpf(u_count) for i in range(u_count)]
    vs = [i*2*mp.pi / mpf(v_count) for i in range(v_count)]
    uvs = [(u,v) for u in us for v in vs]
    return uvs_normals_by_distances(uvs, tor=tor, dists=dists,prec=prec)


# 7: Compare intersection point at different distances for different solvers across random spreads of u and v
def uvrand_normals(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(4, 12)],
    prec: int = DOUBLE_PREC,
    uv_count: int = 25
) -> dict:
    uvs = [(mp.rand()*2*mp.pi, mp.rand()*2*mp.pi) for i in range(uv_count)]
    return uvs_normals_by_distances(uvs, tor=tor, dists=dists,prec=prec)


def uvs_normals_by_distances(
    uvs,
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(4, 12)],
    prec: int = DOUBLE_PREC,
) -> dict:
    result_list = []
    
    for uv in uvs:
        u, v = uv
        uv_result = compare_normals_by_distances(tor=tor, dists=dists, prec=prec, u=u, v=v, result_func=get_distance)
        result_list.append(uv_result)
        print(f"Rays complete: {len(result_list)}")
    
    tt_devs = [] # Compared to the high precision result
    fr_devs = []
    tt_means = []
    fr_means = []
    tt_stddevs = [] # Compared to the average
    fr_stddevs = []

    tt_failrates = [] # At each distance, what percentage of the solves fail to reach a solution entirely
    fr_failrates = []

    for i in range(len(dists)):
        tt_dev_sum = mpf(0)
        fr_dev_sum = mpf(0)
        tt_mean_sum = mpf(0)
        fr_mean_sum = mpf(0)
    
        tt_failcount = 0 # Includes hp_failcount for these two
        fr_failcount = 0
        hp_failcount = 0
        for r in result_list:
            tt_result = r["tt"][i]
            fr_result = r["fr"][i]
            hp_result = r["tt_hp"][i]
            if hp_result is None: hp_result = r["fr_hp"][i]
            if hp_result is None: #This should be rare but it is possible
                tt_failcount += 1
                fr_failcount += 1
                hp_failcount += 1
                continue

            if not tt_result is None:
                tt_dev_sum += (tt_result-hp_result)**2
                tt_mean_sum += tt_result
            else:
                tt_failcount += 1

            if not fr_result is None:
                fr_dev_sum += (fr_result-hp_result)**2
                fr_mean_sum += fr_result
            else:
                fr_failcount += 1
        
        uv_count = len(result_list)
        tt_successes = mpf(uv_count - tt_failcount)
        tt_mean = None if tt_successes == 0 else tt_mean_sum / tt_successes
        tt_dev = None if tt_successes == 0 else mp.sqrt(tt_dev_sum / tt_successes)
        tt_devs.append(tt_dev)
        tt_means.append(tt_mean)
        tt_failrates.append(mpf(tt_failcount)/mpf(uv_count))

        fr_successes = mpf(uv_count - fr_failcount)
        fr_mean = None if fr_successes == 0 else fr_mean_sum / fr_successes
        fr_dev = None if fr_successes == 0 else mp.sqrt(fr_dev_sum / fr_successes)
        fr_devs.append(fr_dev)
        fr_means.append(fr_mean)
        fr_failrates.append(mpf(fr_failcount)/mpf(uv_count))
    
    return {
        "raw_results":result_list,
        "dists":dists,
        "tt_mean":tt_means,
        "tt_dev":tt_devs,
        "tt_fail":tt_failrates,
        "fr_mean":fr_means,
        "fr_dev":fr_devs,
        "fr_fail":fr_failrates
    }



def get_distance(tor, ray_src, ray_dir, slv):
    return tor.distance_to_boundary(ray_src, ray_dir, slv)


def get_first_intersection_z(tor, ray_src, ray_dir, slv):
    inters = tor.ray_intersection_points(ray_src, ray_dir, slv)
    if inters:
        return inters[0][2]
    else:
        return None


def compare_intersections(
    setups: Iterable[tuple[EllipticToroid, matrix, matrix]],
    solver_names: Iterable[str] = ["fr", "tt", "np", "fr_hp", "tt_hp"],
    get_result: callable = get_first_intersection,
    base_prec: int = DOUBLE_PREC,
    high_prec: int = HIGH_PREC,
) -> dict:
    solver_precs = [HIGH_PREC if name.endswith("_hp") else DOUBLE_PREC for name in solver_names]
    base_names = [name[:-3] if name.endswith("_hp") else name for name in solver_names]
    solver_funcs = [get_solver(name) for name in base_names]
    
    mp.prec = HIGH_PREC
    final_results = {
        "tor": [s[0] for s in setups],
        "ray_src": [s[1] for s in setups],
        "ray_dir": [s[2] for s in setups],
        "polynomials":[s[0]._ray_intersection_polynomial(s[1], s[2]) for s in setups]
    }
    for i in range(len(solver_names)):
        name = solver_names[i]
        prec = solver_precs[i]
        func = solver_funcs[i]

        mp.prec = prec
        
        slv_results = []
        slv_logs = []
        for setup in setups:
            tor = setup[0]
            ray_src = setup[1]
            ray_dir = setup[2]

            tracer = AlgTracer(func)
            tracer.begin()
            result = get_result(tor, ray_src, ray_dir, func)
            tracer.end()

            slv_results.append(result)
            slv_logs.append(tracer.simple_logstring())
        
        final_results[name] = slv_results
        final_results[name+"_logs"] = slv_logs
    mp.prec = base_prec
    return final_results
    

def to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(d, orient="index").transpose()


def print_as_df(d: dict):
    print(to_df(d))


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
        if not func_name in dir(self.namespace)+dir(EllipticToroid):
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


def concat_dicts(dict1, dict2) -> dict:
    if not len(dict1) == len(dict2) or \
        not all(key1 in dict2 for key1 in dict1):
        raise ValueError(f"Dictionaries must have identical keys. d1: {dict1}, d2: {dict2}")
    if not all(isinstance(dict1[k], list) and isinstance(dict2[k], list) for k in dict1):
        raise ValueError(f"Dictionaries must have list data to concatenate with each other. d1: {dict1}, d2: {dict2}")

    new_dict = {}
    for key in dict1:
        new_dict[key] = dict1[key].copy()
        new_dict[key].extend(dict2[key])
    
    return new_dict


# Takes the two given dicts of parallel lists, and inserts those of one into those of the other at a specified index.
def insert_dict_at_index(base_dict, insert_dict, idx) -> dict:
    if not len(base_dict) == len(insert_dict) or \
        not all(key1 in insert_dict for key1 in base_dict):
        raise ValueError(f"Dictionaries must have identical keys. d1: {base_dict}, d2: {insert_dict}")
    if not all(isinstance(base_dict[k], list) and isinstance(insert_dict[k], list) for k in base_dict):
        raise ValueError(f"Dictionaries must have list data to concatenate with each other. d1: {base_dict}, d2: {insert_dict}")

    new_dict = {}
    for key in base_dict:
        new_dict[key] = base_dict[key].copy()
        new_dict[key][idx:idx] = insert_dict[key]
    
    return new_dict
