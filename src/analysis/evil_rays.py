'''
A script to compare and plot results of different solvers
'''

import sys
from collections.abc import Iterable
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
from numpy import arange
import pandas as pd
from mpmath import mpf, matrix, mp, pi, power, log
from src.toroid import EllipticToroid
import src.analysis.ray_generator as rg
from src.solvers import get_solver, calc_real_roots
import src.quartics.alg1010 as alg1010


# =-----------=
# Random caches
# =-----------=


try:
    _geab_cache = _geab_cache #Rough check to not reload if it's already been loaded
except NameError:
    _geab_cache = {} #Do we really need to cache EVERYTHING?
    _exp1_final = {} #Final information about experiment 1, e.g. graphs n stuff
    _dbd_cache = {}
    _exp2_final = {}
    _dbdg_cache = {}
    _exp2b_final = {}
    _dbdt_cache = {}
    _exp3_final = {}
    _esct_cache = {}
    _exp3b_final = {}

# =--------------=
# Useful constants
# =--------------=


SINGLE_PREC: int = 23
DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999
MpfAble: type = mpf|float|str


# =------------------=
# Experimental Presets
# =------------------=

BAND_RING_CM: EllipticToroid = EllipticToroid(0.5, mpf("0.02"), 0.25) #idk, the size of finger or smt
THIN_BAGEL_CM: EllipticToroid = EllipticToroid(2.5, 2, 0.2) #or could make it thinner and say it's a vinyl record
ITER_TOROID_CM: EllipticToroid = EllipticToroid(6.2, 4.5, 8.75) #Toroid roughly the scale of ITER, though ITER is a different kind
LHC_TUNNEL_CM: EllipticToroid = EllipticToroid(4.25 * 1_000 * 100, 190, 190) #LHC tunnel because yknow why not

UNIT_TOR = EllipticToroid(1,1,1) #Also critically degenerate, as it would happen


# Makes a copy of the given torus, with radii scaled by the specified amounts
def scaled_copy(tor: EllipticToroid, scales: list[MpfAble]):
    scales = [mpf(s) for s in scales]
    return EllipticToroid(
        tor.tor_rad*scales[0],
        tor.hor_rad*scales[1],
        tor.ver_rad*scales[2]
    )

ITER_SCALED: dict = {
    "base": ITER_TOROID_CM,
    "0.1x": scaled_copy(ITER_TOROID_CM, ["0.1",]*3),
    "0.01x": scaled_copy(ITER_TOROID_CM, ["0.01",]*3),
    "10x": scaled_copy(ITER_TOROID_CM, [10,]*3),
    "100x": scaled_copy(ITER_TOROID_CM, [100,]*3),
    # "r0.1x": scaled_copy(ITER_TOROID_CM, [0.1,1,1]), # These ones are degenerate whoops
    # "r0.01x": scaled_copy(ITER_TOROID_CM, [0.01,1,1]),
    "ra0.1x": scaled_copy(ITER_TOROID_CM, [0.1,0.1,1]),
    "ra0.01x": scaled_copy(ITER_TOROID_CM, [0.01,0.01,1]),
    "b0.1x": scaled_copy(ITER_TOROID_CM, [1,1,0.1]),
    "b0.01x": scaled_copy(ITER_TOROID_CM, [1,1,0.01]),
    "r10x": scaled_copy(ITER_TOROID_CM, [10,1,1]),
    "r100x": scaled_copy(ITER_TOROID_CM, [100,1,1]),
    "ra10x": scaled_copy(ITER_TOROID_CM, [10,10,1]),
    "ra100x": scaled_copy(ITER_TOROID_CM, [100,100,1]),
    "b10x": scaled_copy(ITER_TOROID_CM, [1,1,10]),
    "b100x": scaled_copy(ITER_TOROID_CM, [1,1,100]),
}

SOME_ITER_SCALED: dict = {
    "base": ITER_TOROID_CM,
    "0.1x": scaled_copy(ITER_TOROID_CM, ["0.1",]*3),
    "0.01x": scaled_copy(ITER_TOROID_CM, ["0.01",]*3),
    "10x": scaled_copy(ITER_TOROID_CM, [10,]*3),
    "100x": scaled_copy(ITER_TOROID_CM, [100,]*3),
}

BASIC_RANGES: dict = {
    "outside":[0, 2*mp.pi, -0.1, 1],
    # "inside":[0, 2*mp.pi, mp.pi-0.1, mp.pi+0.1], #Revisit once we get a check on 
    "top":[0, 2*mp.pi, 0.5*mp.pi-0.1, 0.5*mp.pi+0.1],
    "bottom":[0, 2*mp.pi, 1.5*mp.pi-0.1, 1.5*mp.pi+0.1],
    "diag_top":[0, 2*mp.pi, 0.25*mp.pi-0.1, 0.25*mp.pi+0.1],
    "diag_bottom":[0, 2*mp.pi, 1.75*mp.pi-0.1, 1.75*mp.pi+0.1],
    "full_outside":[0, 2*mp.pi, 1.5*mp.pi, 2.5*mp.pi],
}


# =---------------=
# Final experiments
# =---------------=


def graph_eps_avoid_back(
    num_rays: int = 10,
    use_cache: bool = False, #Whether to use the cache data or run calcs all over again from scratch
    log_resolution = mpf("0.01"),
    solvers = ("tt", "fr"),
    precs = {
        "single":SINGLE_PREC,
        "double":DOUBLE_PREC,
        "quad":QUAD_PREC
    },
    verbosity = 1,
):
    ''''''
    closest_escapes = _geab_cache
    if not use_cache: #If we want to recalculate everything
        uv_pairs = [(mp.rand()*2*mp.pi, (mp.rand()+1.5)*mp.pi) for i in range(num_rays)]
        
        closest_escapes["uv_pairs"] = uv_pairs
        closest_escapes["solvers"] = solvers
        closest_escapes["precs"] = precs
                
        for prec in precs:
            slv_results = [None,]*len(solvers)
            for i in range(len(solvers)):
                slv_res = epsilon_avoid_backcollision(
                    uv_pairs=uv_pairs, 
                    prec=precs[prec],
                    solver_code=solvers[i],
                    log_convergence=log_resolution
                )
                closest_escapes[f"{solvers[i]}_{prec}_full"] = slv_res
                last_fail_idx = max([i if not slv_res["valid"][i] else -99999 for i in range(len(slv_res["valid"]))])
                closest = slv_res["dists"][last_fail_idx+1]
                slv_results[i] = closest
            closest_escapes[prec] = slv_results
    else:
        solvers = closest_escapes["solvers"]
        precs = closest_escapes["precs"]
    
    _exp1_final["solvers"] = solvers
    _exp1_final["precs"] = precs
    _exp1_final["uv_pairs"] = [(repr(pair[0]),repr(pair[1])) for pair in closest_escapes["uv_pairs"]]
    for prec in precs:
        _exp1_final[prec] = [float(ep) for ep in closest_escapes[prec]]

    # The actual graphing
    x = arange(len(precs)) #Label locations
    width = 0.25 #Width of bars

    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(solvers)):
        # prec = precs[i]
        slv = solvers[i]
        offset = width*i
        pos = x+offset
        # print(f"width: {width}, i: {i}, offset: {offset}, x: {x}, pos: {pos}")
        # print(f"width: {type(width)}, i: {type(i)}, offset: {type(offset)}, x: {type(x)}, pos: {type(pos)}")
        rects = ax.bar(pos, [float(-log(_exp1_final[prec][i], b=10)) for prec in precs], width, label=slv)
        # print(rects[:])
        print(f"location: {pos}, data: {[_exp1_final[prec][i] for prec in precs]}, width: {width}")
        ax.bar_label(rects, padding=3)

    # Add text
    ax.set_ylabel("Closest escape distance d, -log(d)")
    ax.set_title(f"Closest safe distances from Toroid 50, 10, 20, for {num_rays} rays with different solvers at different precisions")
    ax.set_xticks(x+width, precs)
    ax.legend(loc="upper left", ncols=3)
    plt.show()
    # print(_geab_cache)


def graph_dev_by_distance(
    num_rays: int = 10,
    dists: list = [10**i for i in range(12)],
    solvers: list = ["tt", "fr"],
    precs: dict = {"single":SINGLE_PREC, "double":DOUBLE_PREC, "quad":QUAD_PREC},
    use_cache: bool = False,
):
    if not use_cache:
        random_uvs = [(mp.rand()*2*mp.pi, (mp.rand()+1.5)*mp.pi) for i in range(num_rays)]
        full_res = {}
        for slv in solvers:
            full_res[slv] = {}
            for prec in precs:
                res = uvs_normals_by_distances(random_uvs, dists=dists, prec=precs[prec], solver_code=slv)
                full_res[slv][prec] = res
                _exp2_final[f"{slv}_{prec}"] = {
                    "mean":res["tp_mean"],
                    "dev":res["tp_dev"],
                    "fail":res["tp_fail"]
                }
        _dbd_cache.update(full_res)
        _exp2_final["dists"] = dists
    # If we are using cache, we just assume those are already in place
    x = [float(log(d)) for d in dists]
    _exp2_final["dists"] = [float(d) for d in dists]
    for slv in solvers:
        for prec in precs:
            for dataset in ["mean", "dev", "fail"]: #Convert to floats for serializability
                _exp2_final[f"{slv}_{prec}"][dataset] = [float(n) for n in _exp2_final[f"{slv}_{prec}"][dataset]]
            err = [float(log(n, b=10)) for n in _exp2_final[f"{slv}_{prec}"]["dev"]]
            plt.plot(x, err, drawstyle='steps-mid', label=f"{slv} at {prec}")
    plt.legend(title="Solver-precision combo:")
    plt.xlabel("Log of distance")
    plt.ylabel("Log of error (deviation, same units as distance)")
    plt.title("Error by distance for toroid 50x10x20 for normal rays approaching at different solvers and precisions")
    plt.show()


def graph_dev_by_distance_graze(
    num_rays: int = 10,
    dists: list = [10**i for i in range(12)],
    solvers: list = ["tt", "fr"],
    precs: dict = {"single":SINGLE_PREC, "double":DOUBLE_PREC, "quad":QUAD_PREC},
    use_cache: bool = False,
    surf_dist: mpf = mpf(-1) #Go a little into the torus so it's supposed to hit
):
    tor = EllipticToroid(50,10,20)
    dist_coords = [(d, surf_dist) for d in dists]
    hole_radius = tor.tor_rad - tor.hor_rad
    if not use_cache:
        random_uvs = [(mp.rand()*2*mp.pi, (mp.rand()+1.5)*mp.pi) for i in range(num_rays)]
        full_res = {}
        for slv in solvers:
            full_res[slv] = {}
            for prec in precs:
                res = uvs_grazes_by_distances(
                    random_uvs, dist_coords, slv, precs[prec],
                    tor=tor,
                )
                full_res[slv][prec] = res
                _exp2b_final[f"{slv}_{prec}"] = {
                    "mean":res["tp_mean"],
                    "dev":res["tp_dev"],
                    "fail":res["tp_fail"],
                    "hp_fail":res["hp_fail"]
                }
                print(f"Combo {slv}_{prec} complete")
        _dbdg_cache.update(full_res)
        _exp2b_final["dists"] = dists
    # If we are using cache, we just assume those are already in place
    x = [float(log(d)) for d in dists]
    _exp2b_final["dists"] = [float(d) for d in dists]
    for slv in solvers:
        for prec in precs:
            for dataset in ["mean", "dev", "fail", "hp_fail"]: #Convert to floats for serializability
                _exp2b_final[f"{slv}_{prec}"][dataset] = [0 if n is None else float(n) for n in _exp2b_final[f"{slv}_{prec}"][dataset]]
            err = [float(log(n, b=10)) for n in _exp2b_final[f"{slv}_{prec}"]["dev"]]
            plt.plot(x, err, drawstyle='steps-mid', label=f"{slv} at {prec}")
    plt.legend(title="Solver-precision combo:")
    plt.xlabel("Log of distance")
    plt.ylabel("Log of error (deviation, same units as distance)")
    plt.title("Error by distance for toroid 50x10x20 for grazing rays approaching at different solvers and precisions")
    plt.show()
    

def graph_dbd_by_toroid(
    num_rays: int = 10,
    toroids: dict = {
        "ring":BAND_RING_CM,
        "disc":THIN_BAGEL_CM,
        "iter":ITER_TOROID_CM,
        "lhc":LHC_TUNNEL_CM,
    },
    dists: list = [10**i for i in range(-10,20,2)],
    solver: str = "tt",
    prec: int = DOUBLE_PREC,
    use_cache: bool = False,
):
    if not use_cache:
        random_uvs = [(mp.rand()*2*mp.pi, (mp.rand()+1.5)*mp.pi) for i in range(num_rays)]
        full_res = {}
        for tor_name in toroids:
            tor = toroids[tor_name]
            res = uvs_normals_by_distances(random_uvs, tor=tor, dists=dists, prec=prec, face_outwards=True, solver_code=solver)
            full_res[tor_name] = res
            _exp3_final[tor_name] = {
                "mean":res["tp_mean"],
                "dev":res["tp_dev"],
                "fail":res["tp_fail"],
                "hp_fail":res["hp_fail"]
            }
            print(" "*40, end="\r")
            print(f"Toroid {tor_name} complete")
        _dbdg_cache.update(full_res)
        _exp3_final["dists"] = dists
    # If we are using cache, we just assume those are already in place
    x = [float(log(d)) for d in dists]
    _exp3_final["dists"] = [float(d) for d in dists]
    for tor_name in toroids:
        for dataset in ["mean", "dev", "fail", "hp_fail"]: #Convert to floats for serializability
            _exp3_final[tor_name][dataset] = [0 if n is None else float(n) for n in _exp3_final[tor_name][dataset]]
        err = [float(log(n, b=10)) for n in _exp3_final[tor_name]["dev"]]
        plt.plot(x, err, drawstyle='steps-mid', label=f"{tor_name}:{toroids[tor_name]}")
    plt.legend(title="Toroid:")
    plt.xlabel("Log of distance")
    plt.ylabel("Log of error (deviation, same units as distance)")
    plt.title(f"Error by distance for toroids for grazing rays approaching at precision {prec}")
    plt.show()


def graph_escape_by_toroid_random_ranges(
    num_rays: int = 10,
    raysets: dict[str, Iterable[Iterable[tuple[mpf, mpf]]]] = None, #Override the random ray generation and reuse existing ones. Or set to "cache" to try and find the ones in the cache
    toroids: dict = {
        "ring":BAND_RING_CM, #This one's so small and weird that the given escape distances are actually too big
        "disc":THIN_BAGEL_CM,
        "iter":ITER_TOROID_CM,
        "lhc":LHC_TUNNEL_CM,
    },
    uv_ranges: dict = {
        "outside":[0, 2*mp.pi, -0.1, 1],
        # "inside":[0, 2*mp.pi, mp.pi-0.1, mp.pi+0.1], #Revisit once we get a check on 
        "top":[0, 2*mp.pi, 0.5*mp.pi-0.1, 0.5*mp.pi+0.1],
        "bottom":[0, 2*mp.pi, 1.5*mp.pi-0.1, 1.5*mp.pi+0.1]
    },
    start_dists: list = [power(10,i) for i in [-300,-40,-0.01, 10]],
    solver: str = "tt",
    prec: int = DOUBLE_PREC,
    use_cache: bool = False,
    verbosity: int = 1,
    log_convergence = mpf("0.01")
):
    og_params = locals()
    _esct_cache["params"]=og_params
    _exp3b_final["params"]=og_params
    start_dists = start_dists.copy()
    mp.prec = prec
    start_dists.insert(0,-1)
    if not use_cache:
        if raysets == 'cache':
            uv_sets = deepcopy(_esct_cache["rays"])
        elif raysets == 'cachefill': #Niche: keep existing rays but add however many more are necessary to meet ray quota
            uv_sets = deepcopy(_esct_cache["rays"])
            for range_name in uv_ranges:
                uvr = uv_ranges[range_name]
                newrand_uvs = [(uvr[0] + (uvr[1]-uvr[0])*mp.rand(), uvr[2] + (uvr[3]-uvr[2])*mp.rand()) for i in range(num_rays-len(uv_sets[range_name]))]
                uv_sets[range_name].extend(newrand_uvs)
        elif raysets is None: #Default, generate new rays
            uv_sets = {}

            for range_name in uv_ranges:
                uvr = uv_ranges[range_name]
                random_uvs = [(uvr[0] + (uvr[1]-uvr[0])*mp.rand(), uvr[2] + (uvr[3]-uvr[2])*mp.rand()) for i in range(num_rays)]
                uv_sets[range_name] = random_uvs
        elif isinstance(raysets, dict):
            uv_sets = deepcopy(raysets)
        else:
            raise ValueError(f"Unknown raysets provided of {raysets}")

        _esct_cache["rays"] = uv_sets
        print(toroids)
        for tor_name in toroids:
            tor = toroids[tor_name]
            for range_name in uv_ranges:
                print(f"Analyzing toroid {tor_name} for range {range_name}")
                res = epsilon_avoid_backcollision(
                    tor=tor,
                    epsilons=start_dists,
                    uv_pairs=uv_sets[range_name], 
                    prec=prec, solver_code=solver,
                    verbosity=verbosity,
                    log_convergence=log_convergence,
                )
                _esct_cache[f"{tor_name}_{range_name}"] = res
                NONE_FOUND = -999
                last_fail_idx = max([i if not res["valid"][i] else NONE_FOUND for i in range(len(res["valid"]))])
                closest = 10 if last_fail_idx == NONE_FOUND else res["dists"][last_fail_idx+1]
                _exp3b_final[f"{tor_name}_{range_name}"] = closest

    range_names = [rn for rn in uv_ranges]
    tor_names = [tn for tn in toroids]
    x = arange(len(tor_names)) #Label locations
    width = 1.0/float(1+len(range_names)) #Width of bars

    for tor_name in tor_names:
        for range_name in range_names: #Convert to floats for serializability
            _exp3b_final[f"{tor_name}_{range_name}"] = float(_exp3b_final[f"{tor_name}_{range_name}"])

    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(range_names)):
        # prec = precs[i]
        range_name = range_names[i]
        offset = width*i
        pos = x+offset
        rects = ax.bar(pos, [float(-log(_exp3b_final[f"{tor_names[j]}_{range_name}"], b=10)) for j in range(len(tor_names))], width, label=f"{range_name}")
        # print(rects[:])
        print(f"location: {pos}, data: {[_exp3b_final[f"{tor_names[j]}_{range_name}"] for j in range(len(tor_names))]}, width: {width}")
        ax.bar_label(rects, padding=3)

    # Add text
    ax.set_ylabel("Closest escape distance d, -log(d)")
    ax.set_title(f"Closest safe distances from toroids, for {num_rays} rays each in distinct regions, using {solver} at {prec} prec")
    ax.set_xticks(x+width, tor_names)
    ax.legend(loc="upper left", ncols=3)
    plt.show()






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

def epsilon_avoid_backcollision(
    tor: EllipticToroid = EllipticToroid(50,10,20),
    epsilons: list = [power(10, i) for i in [-400,-100,-50,-25,0]],
    uv_count: int = 10,
    uv_pairs: list = None, #If given specific uv pairs, override random assignment
    prec: int = DOUBLE_PREC,
    log_convergence: MpfAble = "0.1",
    polyn_calc_prec: int = None,
    solver_code: str = "tt",
    verbosity: int = 1,
) -> dict:
    '''
    How close can a ray start to the surface of a toroid and reliably
    avoid intersecting with it. (While moving away)
    Iteratively compares different starting distances until it settles on the lowest stable one.

    Cases considered for rays normal to the surface, and rays tangential
    to the surface.
    
    Parameters
    ----------
    epsilons : list[mpf]
        The distances away from the torus to test, in ascending order
    '''
    epsilons = epsilons.copy()
    log_convergence = mp.convert(log_convergence)
    # First, normal rays
    special_uvs = []
    random_uvs = [(mp.rand()*2*mp.pi, (mp.rand()+1.5)*mp.pi) for i in range(uv_count)]
    if uv_pairs is None:
        uv_pairs = random_uvs
    # Iterate through different distances to try and find 
    full_results: dict = {}
    passfail_history: list[bool] = []
    print(f"Operating on distances: {epsilons}")
    #Start with the given series
    dist_results = lambda dists: uvs_normals_by_distances(
        uv_pairs, dists=dists, face_outwards=True, 
        prec=prec, polyn_calc_prec=polyn_calc_prec,
        verbosity=verbosity,
        solver_code=solver_code,
        tor=tor
    )
    escaped = lambda result, idx=0: result["tp_mean"][idx] is None
    full_results = dist_results(epsilons)
    passfail_history = [escaped(full_results, i) for i in range(len(epsilons))]
    
    highest_fail = max([i if not passfail_history[i] else -1 for i in range(len(passfail_history))])
    # after_hf = dist_results([full_results["dists"][highest_fail]*power(10, log_convergence)])
    
    def insert_results(new_res, idx, full_res, pf_res):
        insert_dict_at_index(full_res, new_res, idx)
        pf_res[idx:idx] = [escaped(new_res)]

    # insert_results(after_hf, highest_fail+1, full_results, passfail_history)

    # Convergence condition: the result a certain log distance ahead of the largest failure succeeds
    max_iter = 100
    c = 0
    converge_ratio = power(10, log_convergence) #Required ratio between the highest failing epsilon and the next epsilon higher (first guaranteed success)
    while (full_results["dists"][highest_fail+1]/full_results["dists"][highest_fail] > converge_ratio) and (c < max_iter):
        highest_dist = full_results["dists"][highest_fail]
        next_dist = full_results["dists"][highest_fail+1]
        if verbosity >= 0: print(f"Iteration {c} complete after dist {highest_dist}, further iteration needed.")
        if verbosity >= 1: mp.prec=prec;print(to_df(full_results))

        betw_dist = mp.sqrt(highest_dist*next_dist)
        betw_result = dist_results([betw_dist])

        insert_results(betw_result, highest_fail+1, full_results, passfail_history)
        if not escaped(betw_result): #If the bisection also fails, next bisection goes above it
            highest_fail += 1


        
    # while (not passfail_history[highest_fail+1]) and (c < max_iter):
    #     highest_dist = full_results["dists"][highest_fail+1]
    #     if verbosity >= 0: print(f"Iteration {c} complete after dist {highest_dist}, further iteration needed.")
    #     if verbosity >= 1: mp.prec=prec;print(to_df(full_results))

    #     if (highest_fail+2 >= len(full_results["dists"])): #biggest, go a little bigger
    #         next_dist = full_results["dists"][highest_fail+1]*power(10, log_convergence)
    #     else:
    #         next_dist = full_results["dists"][highest_fail+2]
    #     betw_dist = mp.sqrt(highest_dist*next_dist)
    #     betw_result = dist_results([betw_dist])
    #     betw_passfail = escaped(betw_result)

    #     insert_results(betw_result, highest_fail+2, full_results, passfail_history)

    #     if not passfail_history[highest_fail+2]: #The bisection also fails, bisect again above
    #         highest_fail += 2
    #     else: #The bisection succeeds, so bisect again below
    #         highest_fail += 1

    #     # Check right above it again
    #     after_hf = dist_results([full_results["dists"][highest_fail]*power(10, log_convergence)])
    #     insert_results(after_hf, highest_fail+1, full_results, passfail_history)

        c += 1
    if verbosity >= 0:
        print(f"Complete after {c} iterations")
        print(" "*20, end="\r")
    if verbosity >= 1: mp.prec=prec;print(to_df(full_results))
    full_results["valid"] = passfail_history
    return full_results


# 1: Compare intersection point at different distances for different solvers
def compare_normals_by_distances(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(-16, 12)],
    prec: int = DOUBLE_PREC,
    u: mpf = 0, v: mpf = pi / 2,
    return_dtb: bool = False,
    return_logs: bool = False,
    result_func: callable = get_first_intersection,
    face_outwards: bool = False,
    polyn_calc_prec: int = None,
    solver_names: list = ["tt", "tt_hp", "fr", "fr_hp"]
) -> dict:
    dists = [mpf(d) for d in dists]
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u, v, dist = 1)[1]
    if face_outwards: ray_dir *= -1

    # Combos of dists and solvers
    sources = [rg.get_normal_ray(tor, u, v, dist = d)[0] for d in dists]
    setups = [(tor, src, ray_dir) for src in sources]
    results = compare_intersections(
        setups,
        base_prec=prec,
        get_result=result_func,
        polyn_calc_prec=polyn_calc_prec,
        solver_names=solver_names,
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


def uvs_grazes_by_distances(
    uvs: list[tuple],
    dists: list[tuple[MpfAble]], # Ray distances, then normal distances
    solver_code: str,
    prec: int,
    tor: EllipticToroid = EllipticToroid(50,10,20),
    verbosity: int = 1,
    fail_condition: callable = None
) -> dict:
    '''
    Compares the results of different distance pairs (along ray and away from surface)
    on a set of uvs, against their high-precision counterparts
    '''
    hole_radius = tor.tor_rad - tor.hor_rad
    if fail_condition is None: fail_condition = lambda dist, set_idx, ray_idx: dist is None or abs(dist-dists[set_idx][0]) > hole_radius # Any further out and we assume it might be (correctly) detecting the intersection on the other side
    raysets = [[rg.get_grazing_ray(tor, u=uv[0], v=uv[1], distance=d[0], pos_epsilon=d[1]) for uv in uvs] for d in dists]
    return compare_raysets_vs_hp(raysets, solver_code, prec, toroid=tor, is_failure=fail_condition, verbosity=verbosity)
    


def uvs_normals_by_distances(
    uvs,
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(4, 12)],
    prec: int = DOUBLE_PREC,
    face_outwards: bool = False,
    polyn_calc_prec: int = None,
    solver_code: str = "tt",
    verbosity: int = 1
) -> dict:
    result_list = []

    raysets = [[rg.get_normal_ray(tor, uv[0], uv[1], d) for uv in uvs] for d in dists]
    if face_outwards:
        raysets = [[(ray[0], ray[1]*-1) for ray in rayset] for rayset in raysets]
    full_res = compare_raysets_vs_hp(raysets, solver_code, prec, tor, verbosity=verbosity)
    full_res["dists"] = dists
    return full_res
    
    # for uv in uvs:
    #     u, v = uv
    #     if verbose: print(f"Comparing u: {u}, v: {v}, on distances: {dists}")
    #     uv_result = compare_normals_by_distances(
    #         tor=tor, dists=dists, prec=prec, 
    #         u=u, v=v, result_func=get_distance, 
    #         face_outwards=face_outwards, polyn_calc_prec=polyn_calc_prec, 
    #         solver_names=[solver_code, f"{solver_code}_hp"])
    #     if verbose: print(f"{solver_code} result: {uv_result[solver_code]}, vs {uv_result["dists"]}")
    #     result_list.append(uv_result)
    #     if verbose: print(f"Rays complete: {len(result_list)}")
    #     elif not silent: 
    #         print(" "*20, end="\r")
    #         print(f"Rays completed: {len(result_list)}", end='\r')
    # tp_devs = [] # Compared to the high precision result
    # tp_means = []

    # tp_stddevs = [] # Compared to the average
    # tp_failrates = [] # At each distance, what percentage of the solves fail to reach a solution entirely

    # hp_failrates = [] # At each distance, what percentage of high-precision solves failed

    # for i in range(len(dists)):
    #     tp_dev_sum = mpf(0)
    #     tp_mean_sum = mpf(0)
    
    #     tp_failcount = 0 # Includes hp_failcount for these two
    #     hp_failcount = 0
    #     both_succeedcount = 0
    #     for r in result_list:
    #         tp_result = r[solver_code][i]
    #         hp_result = r[f"{solver_code}_hp"][i]
    #         if hp_result is None:
    #             hp_failcount += 1

    #         if tp_result is None:
    #             tp_failcount += 1
    #         else:
    #             tp_mean_sum += tp_result
    #             if not hp_result is None:
    #                 both_succeedcount += 1
    #                 tp_dev_sum += (tp_result-hp_result)**2    
        
    #     uv_count = len(result_list)
    #     tp_successes = mpf(uv_count - tp_failcount)
    #     tp_mean = None if tp_successes == 0 else tp_mean_sum / tp_successes
    #     tp_dev = None if tp_successes-hp_failcount == 0 else mp.sqrt(tp_dev_sum / mpf(both_succeedcount))
    #     tp_devs.append(tp_dev)
    #     tp_means.append(tp_mean)
    #     tp_failrates.append(mpf(tp_failcount)/mpf(uv_count))

    #     hp_failrates.append(mpf(hp_failcount)/mpf(uv_count))
    # return {
    #     "raw_results":result_list,
    #     "dists":dists,
    #     "tp_mean":tp_means,
    #     "tp_dev":tp_devs,
    #     "tp_fail":tp_failrates,
    #     "hp_fail":hp_failrates
    # }



def get_distance(tor, ray_src, ray_dir, slv):
    return tor.distance_to_boundary(ray_src, ray_dir, slv)


def get_first_intersection_z(tor, ray_src, ray_dir, slv):
    inters = tor.ray_intersection_points(ray_src, ray_dir, slv)
    if inters:
        return inters[0][2]
    else:
        return None


def compare_raysets_vs_hp(
    raysets: Iterable[Iterable[tuple[matrix, matrix]]],
    solver_name: str,
    target_prec: int,
    toroid: EllipticToroid = EllipticToroid(50,10,20),
    high_prec: int = HIGH_PREC,
    polyn_calc_prec: int = None, 
    is_failure: callable = lambda dist, set_idx, ray_idx: dist is None,
    verbosity: int = 0
) -> dict:
    '''
    Parameters
    ----------
    raysets : Iterable[Iterable[tuple[matrix src, matrix dir] ray] rayset]
        Sets of comparable rays; The results will be averaged over each set (inner iterable)
        and returned as lists
    is_failure : callable(dist, set_idx, ray_idx)
        For the given ray in the given rayset, what defines if it were to fail?
        By default, just if it didn't find an intersection, but this could be adjusted to 
        exclude intersections that would belong to other parts of the toroid, for example
    '''
    final_results = {
        "full_logs":[],
        "tp_mean":[],
        "tp_dev":[],
        "tp_fail":[],
        "hp_fail":[]
    }
    solver = get_solver(solver_name)
    for set_idx in range(len(raysets)):
        rayset = raysets[set_idx]
        mean_sum = 0
        dev_sum = 0
        tp_failcount = 0
        hp_failcount = 0
        both_succeedcount = 0

        num_rays = len(rayset)
        logs = []
        c = 0
        for ray_idx in range(num_rays):
            ray = rayset[ray_idx]
            tracer = AlgTracer(solver)
            tracer.begin()
            mp.prec = target_prec
            tp_dtb = toroid.distance_to_boundary(ray[0], ray[1], solver)
            # print(f"Testing precision: {mp.prec} (should be {target_prec})")
            tracer.end()
            target_log = f"{solver_name} at prec {mp.prec}\n"+tracer.simple_logstring()
            tracer.begin()
            mp.prec = high_prec
            hp_dtb = toroid.distance_to_boundary(ray[0], ray[1], solver)
            tracer.end()
            high_log = f"{solver_name} at prec {mp.prec}\n"+tracer.simple_logstring()

            logs.append(f"{target_log}\n{high_log}")

            tp_fail = is_failure(tp_dtb, set_idx, ray_idx)
            hp_fail = is_failure(hp_dtb, set_idx, ray_idx)

            if tp_fail: tp_failcount+=1
            if hp_fail: hp_failcount+=1

            # Average: add if it succeeds
            if not tp_fail:
                mean_sum += tp_dtb
                # Deviation: add if there's also a hp to compare to
                if not hp_fail:
                    both_succeedcount += 1
                    dev_sum += (tp_dtb-hp_dtb)**2
            
            c += 1
            if verbosity >= 0: 
                print(" "*40, end="\r")
                print(f"Set {set_idx} rays completed: {c}", end='\r')

        
        mean = None if (num_rays-tp_failcount)==0 else mean_sum / mpf(num_rays - tp_failcount)
        dev = None if both_succeedcount==0 else mp.sqrt(dev_sum / mpf(both_succeedcount))
        tp_failrate = mpf(tp_failcount) / mpf(num_rays)
        hp_failrate = mpf(hp_failcount) / mpf(num_rays)

        final_results["full_logs"].append(logs)
        final_results["tp_mean"].append(mean)
        final_results["tp_dev"].append(dev)
        final_results["tp_fail"].append(tp_failrate)
        final_results["hp_fail"].append(hp_failrate)
    return final_results



def compare_intersections(
    setups: Iterable[tuple[EllipticToroid, matrix, matrix]],
    solver_names: Iterable[str] = ["fr", "tt", "np", "fr_hp", "tt_hp"],
    get_result: callable = get_first_intersection,
    base_prec: int = DOUBLE_PREC,
    high_prec: int = HIGH_PREC,
    polyn_calc_prec: int = None, # if given, use this instead of case by case precision
) -> dict:
    solver_precs = [high_prec if name.endswith("_hp") else base_prec for name in solver_names]
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
            tor._polyn_calc_prec = prec if polyn_calc_prec is None else polyn_calc_prec
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
                "returned": f"{arg}"
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
        raise ValueError(f"Dictionaries must have identical keys. d1: {[key for key in base_dict]}, d2: {[key for key in insert_dict]}")
    if not all(isinstance(base_dict[k], list) and isinstance(insert_dict[k], list) for k in base_dict):
        raise ValueError(f"Dictionaries must have list data to concatenate with each other. d1: {[key for key in base_dict]}, d2: {[key for key in insert_dict]}")

    # new_dict = {}
    for key in base_dict:
        # base_dict[key] = base_dict[key].copy()
        base_dict[key][idx:idx] = insert_dict[key]
    
    return base_dict


# Returns a nested copy of the given dictionary/list combination. Currently only operates on dicts and lists
def deepcopy(item):
    if isinstance(item, dict):
        new_dict = {}
        for key in item:
            new_dict[key] = deepcopy(item[key])
        return new_dict
    elif isinstance(item, list):
        return [deepcopy(n) for n in item]
    else:
        return item


# Returns a new set of random uvs in the given range
def random_uvs(uv_range: list[MpfAble], num_rays: int) -> list[tuple[mpf, mpf]]:
    return [(uv_range[0] + mp.rand()*(uv_range[1]-uv_range[0]), uv_range[2] + mp.rand()*(uv_range[3]-uv_range[2])) for i in range(num_rays)]


# Returns sets of random uvs 
