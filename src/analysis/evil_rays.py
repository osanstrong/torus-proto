'''
A script to compare and plot results of different solvers
'''

import sys
import builtins as blt
import json
from collections.abc import Iterable
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import Annotation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pandas as pd
from mpmath import mpf, matrix, mp, pi, power, log
import mpmath.ctx_mp_python as ctx_mp_python
import mpmath.libmp.backend as mpbackend
from src.toroid import EllipticToroid
import src.toroid as toroid
import src.analysis.ray_generator as rg
from src.solvers import get_solver, calc_real_roots
import src.quartics.alg1010 as alg1010
from src.prec_util import mp_const, CONST_PREC
import src.prec_util as pu


# =-----------=
# Random caches
# =-----------=


try:
    _geab_cache = _geab_cache #Rough check to not reload if it's already been loaded
except NameError:
    _geab_cache = {} #Do we really need to cache EVERYTHING?
    _exp1_final = {} #Final information about experiment 1, e.g. graphs n stuff
    _exp1b_cache = {}
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
CSS_COLS: dict = mcolors.CSS4_COLORS
XKCD_COLS: dict = mcolors.XKCD_COLORS
SLV_DISPLAY: dict = {
    "tt": "Algorithm 1010",
    "fr": "Ferrari Method"
}
# Representing solver by color and precision by line style
SLV_COLS: dict = {
    "tt": XKCD_COLS['xkcd:teal'],
    "fr": XKCD_COLS['xkcd:orange'],
}
PREC_LINESTYLES: dict = {
    "single": ":",
    "double": "--",
    "quad": "-"
}
# Representing solver by linestyle and precision by color
SLV_LINESTYLES: dict = {
    "tt": "-",
    "fr": "--",
}
SLV_POINTSIZE: float = 3
SLV_POINTSTYLE: dict = {
    "tt": "o",
    "fr": "o",
}
PREC_COLS: dict = {
    "single":CSS_COLS["darkviolet"],
    "double":CSS_COLS["red"],
    "quad":CSS_COLS["darkorange"],
}
PREC_SLV_COLS: dict = {
    "single":{
        "tt": CSS_COLS["indigo"],
        "fr": CSS_COLS["darkviolet"] 
    },
    "double":{
        "tt": CSS_COLS["maroon"],
        "fr": CSS_COLS["red"]
    },
    "quad":{
        "tt": CSS_COLS["saddlebrown"], 
        "fr": CSS_COLS["darkorange"]
    },
}
RAY_DISPLEN = 20


# =------------------=
# Experimental Presets
# =------------------=

BAND_RING_CM: EllipticToroid = EllipticToroid(0.5, mpf("0.02"), 0.25) #idk, the size of finger or smt
THIN_BAGEL_CM: EllipticToroid = EllipticToroid(2.5, 2, 0.2) #or could make it thinner and say it's a vinyl record
ITER_TOROID_CM: EllipticToroid = EllipticToroid(6.2, 4.5, 8.75) #Toroid roughly the scale of ITER, though ITER is a different kind
LHC_TUNNEL_CM: EllipticToroid = EllipticToroid(4.25 * 1_000 * 100, 190, 190) #LHC tunnel because yknow why not

UNIT_TOR = EllipticToroid(1,1,1) #Also critically degenerate, as it would happen


# Makes a copy of the given torus, with radii scaled by the specified amounts
def scaled_copy(tor: EllipticToroid, scales: list[MpfAble], prec: int = None):
    prev_prec = mp.prec
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

KNOWN_TORII_CM: dict = {
    "jet_plasma": EllipticToroid(300, 125, 200),
    "inner_dune_fsc": EllipticToroid("2.3", 0.5, 0.5),
    "outer_dune_fsc": EllipticToroid("2.3", "2.285", "2.285"),
    "lz_pmtConduitBend": EllipticToroid("37.5", "10.96", "10.96"),
    "lz_thermoConduitBend": EllipticToroid(25, 8, 8),
    "xlzd_0x1c60140": EllipticToroid("141.699993610382", "37.2999995946884","37.2999995946884"),
    "inner_xlzd_0x1ccaf60": EllipticToroid("141.699993610382", "35.7999980449677", "35.7999980449677"),
    "xlzd_0x1c8eef0": EllipticToroid("126.199996471405", "33.799996972084", "33.799996972084"),
    "outer_xlzd_0x1cc8960": EllipticToroid("123.259997367859", "32.9399973154068","329.399973154068"),
    "inner_xlzd_0x1cc8960": EllipticToroid("123.259997367859", "31.2399983406067","31.2399983406067"),
    "outer_xlzd_0x1cc75a0": EllipticToroid("127.639997005463", "34.1599971055984", "34.1599971055984"),
    "inner_xlzd_0x1cc75a0": EllipticToroid("127.639997005463", "32.3599994182587", "32.3599994182587"),
}

BASIC_RANGES: dict = {
    "outside":[0, 2, -0.1, 1],
    # "inside":[0, 2, mp.pi-0.1, mp.pi+0.1], #Revisit once we get a check on 
    "top":[0, 2, 0.5-0.1, 0.5+0.1],
    "bottom":[0, 2, 1.5-0.1, 1.5+0.1],
    "diag_top":[0, 2, 0.25-0.1, 0.25+0.1],
    "diag_bottom":[0, 2, 1.75-0.1, 1.75+0.1],
    "full_outside":[0, 2, 1.5, 2.5],
    "full":[0,2,0,2],
}


# =---------------=
# Final experiments
# =---------------=


def graph_eps_avoid_back(
    num_rays: int = 10,
    use_cache: bool = False, #Whether to use the cache data or run calcs all over again from scratch,
    uv_pairs: Iterable[tuple[mpf, mpf]] = None, #If not using cache, reuse a specific set of rays for repeatability
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
        if uv_pairs is None:
            mp.prec += 200
            uv_pairs = [(mp_const(mp.rand()*2), mp_const(mp.rand()+1.5)) for i in range(num_rays)]
            mp.prec -= 200
        
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
        if closest_escapes:
            solvers = closest_escapes["solvers"]
            precs = closest_escapes["precs"]
        else:
            solvers = _exp1_final["solvers"]
            precs = _exp1_final["precs"]
    
    _exp1_final["solvers"] = solvers
    _exp1_final["precs"] = precs
    if closest_escapes:_exp1_final["uv_pairs"] = [(repr(pair[0]),repr(pair[1])) for pair in closest_escapes["uv_pairs"]]
    for prec in precs:
        _exp1_final[prec] = [float(ep) for ep in _exp1_final[prec]]

    # The actual graphing
    x = np.arange(len(precs)) #Label locations
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


def graph_geab_indvuv( #experiment 1b
    tor: EllipticToroid = KNOWN_TORII_CM["jet_plasma"],
    raygen_type: str = "fill", #Other option is 'rand', which uses num_rays instead of num_u, num_v, and uv_range
    num_rays: int = 10, # for rand
    num_u: int = 20, # for fill
    num_v: int = 20, # for fill
    uv_range: Iterable[mpf] = BASIC_RANGES['full'], # for fill
    use_cache: bool = False, #Whether to use the cache data or run calcs all over again from scratch,
    uv_pairs: Iterable[tuple[mpf, mpf]] = None, #If not using cache, reuse a specific set of rays for repeatability
    ray_type: str = "normal", #Other option: grazing for a grazing ray
    log_resolution = mpf("0.01"),
    solvers = ("tt", "fr"),
    precs = {
        "single":SINGLE_PREC,
        "double":DOUBLE_PREC,
        "quad":QUAD_PREC
    },
    verbosity = 1,
):
    if not use_cache:
        if uv_pairs is None:
            if raygen_type == "rand":
                uv_pairs = [(mp_const(mp.rand()*2), mp_const(mp.rand()+1.5)) for i in range(num_rays)]
            elif raygen_type == "fill":
                uv_pairs = uv_fillrange(uv_range, num_u-1, num_v-1)
                _exp1b_cache['uv_range'] = uv_range
            else:
                raise ValueError(f"Unrecognized ray generation type {raygen_type}")
        _exp1b_cache["uv_pairs"] = uv_pairs
        _exp1b_cache['ray_type'] = ray_type
        
        _exp1b_cache["graph"] = {
            "uv_pairs": uv_pairs,
            "ray_type": ray_type,
            "solvers": solvers,
            "precs": precs,
            "toroid":tor,
        }
        if raygen_type=="fill":
            _exp1b_cache["graph"]['uv_range'] = uv_range

        for slv in solvers:
            for prec_name in precs:
                prec = precs[prec_name]
                fullres_list = []
                mineps_list = []
                for i in range(len(uv_pairs)):
                    uv = uv_pairs[i]
                    full_res = singuv_epsilon_avoid_backcollision(tor, uv, solver_code=slv, prec=prec, ray_type=ray_type)
                    fullres_list.append(full_res)
                    mineps_list.append(full_res[0])
                    if verbosity >= 0: print(f"{" "*76}\r{slv}_{prec_name} rays complete: {i+1}", end="\r")
                mineps = max(mineps_list)
                _exp1b_cache[f"{slv}_{prec_name}_full"] = fullres_list
                _exp1b_cache[f"{slv}_{prec_name}"] = mineps
                _exp1b_cache["graph"][f"{slv}_{prec_name}"] = float(mineps)
                if verbosity >= 0: print(f"{slv}_{prec_name} complete with mineps of {mineps}")
    
    tor = _exp1b_cache['graph']['toroid']
    uv_pairs = _exp1b_cache['graph']['uv_pairs']
    num_rays = len(uv_pairs)
    ray_type = _exp1b_cache['graph']['ray_type']
    # The actual graphing
    x = np.arange(len(precs)) #Label locations
    width = 0.35 #Width of bars

    # fig, ax = plt.subplots(layout='constrained')
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    fig.set_layout_engine('constrained')

    min_y = max_y = 1
    for i in range(len(solvers)):
        # prec = precs[i]
        slv = solvers[i]
        offset = width*i
        pos = x+offset
        # print(f"width: {width}, i: {i}, offset: {offset}, x: {x}, pos: {pos}")
        # print(f"width: {type(width)}, i: {type(i)}, offset: {type(offset)}, x: {type(x)}, pos: {type(pos)}")
        bottoms = [float(_exp1b_cache["graph"][f"{slv}_{prec}"]) for prec in precs]
        min_y = min(min_y, min(bottoms))
        heights = [1-b for b in bottoms]
        # rects = ax.bar(pos, heights, width, bottom=bottoms, label=SLV_DISPLAY[slv], log=True, color=[PREC_SLV_COLS[prec][slv] for prec in precs])
        rects = ax.bar(pos, heights, width, bottom=bottoms, label=SLV_DISPLAY[slv], log=True, color=SLV_COLS[slv])
        # print(rects[:])
        # print(f"location: {pos}, data: {[_exp1b_cache[f"{slv}_{prec}"] for prec in precs]}, width: {width}")
        ax.bar_label(rects, labels=['{:0.2e}'.format(b) for b in bottoms], label_type='center')

    # Add text
    ax.set_ylabel("Closest escape distance (cm)")
    ax.set_xlabel("Precision level of calculations")
    ax.yaxis.set_inverted(True)
    ax.set_ylim([max_y, min_y*0.1])
    # ax.set_yscale('log')
    ray_desc = "??? rays"
    if raygen_type == "fill":
        ray_desc = f"{num_rays} {ray_type} rays spanning {_exp1b_cache['graph']['uv_range']}"
    elif raygen_type == "rand":
        ray_desc = f"{num_rays} random {ray_type} rays"

    # fig.suptitle(f"Closest safe distances from Toroid {tor}, for {ray_desc} with different solvers at different precisions")
    ax.set_xticks(x+width, precs)
    ax.set_title("Escapable distances")
    multicol_patchlist = [[mpatches.Patch(facecolor=PREC_SLV_COLS[prec][slv], label=SLV_DISPLAY[slv]) for prec in precs] for slv in solvers]
    ax.legend(
        handler_map = {list: HandlerTuple(None)},
        # handles=multicol_patchlist, 
        labels=[SLV_DISPLAY[slv] for slv in solvers],
        loc="upper left"
    )

    ax2 = fig.add_subplot(1,2,2, projection="3d", computed_zorder=False)
    _plot_pincushion_on_ax(ax2, _exp1b_cache['graph'])
    ax2.set_title("Rays")
    plt.show()


def _plot_pincushion_on_ax(ax, graph_cache: dict):
    tor = graph_cache['toroid']
    pad = float(1.5*(tor.tor_rad+tor.hor_rad))
    X, Y, Z = plot_toroid(tor)
    ax.axes.set_xlim3d(left=-pad, right=pad)
    ax.axes.set_ylim3d(bottom=-pad, top=pad)
    ax.axes.set_zlim3d(bottom=-pad, top=pad)
    # ax.plot_surface(X, Y, Z, antialiased=True, color="orange", zorder=0)
    ax.plot_surface(X, Y, Z, antialiased=True, color="orange")

    ray_type = graph_cache['ray_type']
    uv_pairs = graph_cache['uv_pairs']
    print(f"ray type: {ray_type}")
    if ray_type == 'normal':
        get_ray = lambda uv: rg.get_normal_ray(tor, uv[0], uv[1], 0)
    elif ray_type == 'grazing':
        get_ray = lambda uv: rg.get_grazing_ray(tor, uv[0], uv[1])

    raylen = RAY_DISPLEN
    raylen = (tor.tor_rad*tor.hor_rad)**0.5
    backsend = 4
    rays = [get_ray(uv) for uv in uv_pairs]
    rays = [(ray[0]+backsend*ray[1], ray[1]*-raylen) for ray in rays]
    for ray in rays:
        s = [float(n) for n in ray[0]]
        d = [float(n) for n in ray[1]]
        ax.arrow3D(s, d)    
    

def graph_double_pincushion(
        tor = KNOWN_TORII_CM["jet_plasma"],
        uv_range = [0,2,0,2],
        num_u = 10,
        num_v = 10,
    ):
    fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw={"projection":"3d"})
    uv_pairs = uv_fillrange(uv_range, num_u-1, num_v-1)
    mockres1 = {
        'toroid':tor,
        "uv_pairs":uv_pairs,
        'ray_type':'normal',
    }
    mockres2 = {
        "toroid":tor,
        "uv_pairs":uv_pairs,
        'ray_type':'grazing',
    }
    _plot_pincushion_on_ax(ax1, mockres1)
    ax1.set_title('Normal Rays')
    _plot_pincushion_on_ax(ax2, mockres2)
    ax2.set_title("Grazing Rays")
    plt.show()



def _plot_dbd(results: dict,
    # prec_slv_colors: dict = {
    #     "single":[CSS_COLS["darkviolet"], CSS_COLS["rebeccapurple"]],
    #     "double":[CSS_COLS["red"], CSS_COLS["darkred"]],
    #     "quad":[CSS_COLS["darkorange"], CSS_COLS["chocolate"]],
    # },
    ):
    x = results['dists']
    precs = results['precs']
    solvers = results['solvers']

    # fig, ax = plt.subplots()
    fig = plt.figure(layout="compressed")
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="major")
    ax.grid(visible=True, which="minor", color="0.9")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Error (cm)")
    locmin = mticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=120)
    ax.yaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_locator(locmin)
    ax.minorticks_on()

    for prec in precs:
        for slv_i in range(len(solvers)):
            slv = solvers[slv_i]
            for dataset in ["mean", "dev", "fail"]: #Convert to floats for serializability
                results[f"{slv}_{prec}"][dataset] = [0 if n is None else float(n) for n in _exp2_final[f"{slv}_{prec}"][dataset]]
            err = results[f"{slv}_{prec}"]["dev"]
            ax.scatter(x, err, marker=SLV_POINTSTYLE[slv], color=PREC_SLV_COLS[prec][slv], label=f"{SLV_DISPLAY[slv]}, {prec}", zorder=5)
            # ax.plot(x, err, '', label=f"{SLV_DISPLAY[slv]}, {prec}", linestyle=PREC_LINESTYLES[prec], color=SLV_COLS[slv])
            # ax.plot(x, err, drawstyle='steps-mid', label=f"{SLV_DISPLAY[slv]}, {prec}", linestyle=PREC_LINESTYLES[prec], color=SLV_COLS[slv])
    
    
    ax.legend(title="Solver, precision:", loc='lower left', bbox_to_anchor=(1.01, 0.3))

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_pincushion_on_ax(ax2, results)
    
    # plt.title("Error by distance for toroid 50x10x20 for normal rays approaching at different solvers and precisions")
    plt.show()  


def graph_dev_by_distance(
    tor: EllipticToroid = KNOWN_TORII_CM["jet_plasma"],
    uv_pairs = None,
    raygen_type: str = "fill", #Other option is 'rand', which uses num_rays instead of num_u, num_v, and uv_range
    num_rays: int = 10, # for rand
    num_u: int = 10, # for fill
    num_v: int = 10, # for fill
    uv_range: Iterable[mpf] = BASIC_RANGES['full_outside'], # for fill
    dists: list = [mp_const(f"1e{i}") for i in range(-12, 12)],
    solvers: list = ["tt", "fr"],
    precs: dict = {"single":SINGLE_PREC, "double":DOUBLE_PREC, "quad":QUAD_PREC},
    use_cache: bool = False,
    ray_type: str = "normal", # Other option is grazing
    verbosity: int = 1
):
    if not use_cache:
        full_res = {}
        if uv_pairs is None:
            if raygen_type == "rand":
                uv_pairs = [(mp_const(mp.rand()*2), mp_const(mp.rand()+1.5)) for i in range(num_rays)]
            elif raygen_type == "fill":
                uv_pairs = uv_fillrange(uv_range, num_u-1, num_v-1)
                _exp2_final['uv_range'] = uv_range
            else:
                raise ValueError(f"Unrecognized ray generation type {raygen_type}")
        _exp2_final['uv_pairs'] = uv_pairs
        _exp2_final['ray_type'] = ray_type
        _exp2_final['toroid'] = tor
        full_res.update(_exp2_final)

        for slv in solvers:
            full_res[slv] = {}
            for prec in precs:
                if ray_type == "normal":
                    res = uvs_normals_by_distances(uv_pairs, tor=tor, dists=dists, prec=precs[prec], solver_code=slv, verbosity=verbosity)
                elif ray_type == "grazing":
                    res = uvs_grazes_by_distances(uv_pairs, [(d, 0) for d in dists], slv, precs[prec],  tor=tor, verbosity=verbosity)
                full_res[f"{slv}_{prec}_full"] = res
                _exp2_final[f"{slv}_{prec}"] = {
                    "mean":res["tp_mean"],
                    "dev":res["tp_dev"],
                    "fail":res["tp_fail"]
                }
                if verbosity >= 0: print(f"{slv}_{prec} complete                 ")
        _dbd_cache.update(full_res)
        _exp2_final["dists"] = dists
    else:
        dists = _exp2_final["dists"].copy()
        tor = _exp2_final["toroid"]

    # If we are using cache, we just assume those are already in place
    _exp2_final["dists"] = [float(d) for d in dists]
    _exp2_final["precs"] = precs
    _exp2_final["solvers"] = solvers
    _plot_dbd(_exp2_final)
    


def graph_dev_by_distance_graze(
    uv_pairs = None,
    raygen_type: str = "fill", #Other option is 'rand', which uses num_rays instead of num_u, num_v, and uv_range
    num_rays: int = 10, # for rand
    num_u: int = 30, # for fill
    num_v: int = 10, # for fill
    uv_range: Iterable[mpf] = BASIC_RANGES['full'], # for fill
    dists: list = [mp_const(f"1e{i}") for i in range(-12,12)],
    solvers: list = ["tt", "fr"],
    precs: dict = {"single":SINGLE_PREC, "double":DOUBLE_PREC, "quad":QUAD_PREC},
    use_cache: bool = False,
    surf_dist: mpf = 0 #Go a little into the torus so it's supposed to hit
):
    tor = EllipticToroid(50,10,20)
    dist_coords = [(d, surf_dist) for d in dists]
    hole_radius = tor.tor_rad - tor.hor_rad
    if not use_cache:
        if uv_pairs is None:
            if raygen_type == "rand":
                uv_pairs = [(mp_const(mp.rand()*2), mp_const(mp.rand()+1.5)) for i in range(num_rays)]
            elif raygen_type == "fill":
                uv_pairs = uv_fillrange(uv_range, num_u-1, num_v-1)
            else:
                raise ValueError(f"Unrecognized ray generation type {raygen_type}")
        full_res = {}
        for slv in solvers:
            full_res[slv] = {}
            for prec in precs:
                res = uvs_grazes_by_distances(
                    uv_pairs, dist_coords, slv, precs[prec],
                    tor=tor
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
        random_uvs = [(mp.rand()*2, (mp.rand()+1.5)) for i in range(num_rays)]
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
        _dbdt_cache.update(full_res)
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
        "outside":[0, 2, -0.1, 1],
        # "inside":[0, 2, mp.pi-0.1, mp.pi+0.1], #Revisit once we get a check on 
        "top":[0, 2, 0.5-0.1, 0.5+0.1],
        "bottom":[0, 2, 1.5-0.1, 1.5+0.1]
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
    x = np.arange(len(tor_names)) #Label locations
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

def singuv_epsilon_avoid_backcollision(
    tor: EllipticToroid,
    uv: tuple[mpf, mpf],
    epsilons: list = [mp_const(f"1e{i}") for i in [-400, -300,0]],
    prec: int = DOUBLE_PREC,
    solver_code: str = "tt",
    log_convergence: MpfAble = "0.1",
    verbosity: int = 1,
    ray_type: str = "normal", #Other option: grazing for a grazing ray
    avoid_condition: callable = lambda dtb, hole_radius: dtb is None or dtb > 1.9*hole_radius #Also discount rays that, say you're on the inside region, hit the other side again
) -> tuple[mpf, dict, int]: #The min epsilon, the full results, and the exit code (0 for full iteration, -1 for all avoided and 1 for all hit)
    prev_prec = mp.prec
    solver = get_solver(solver_code)
    hole_radius = tor.tor_rad-tor.hor_rad
    def avoidance_result(eps) -> dict:
        if ray_type == "normal":
            ray = rg.get_normal_ray(tor, uv[0], uv[1], eps)
        elif ray_type == "grazing":
            ray = rg.get_grazing_ray(tor, uv[0], uv[1], pos_epsilon=eps)
        elif ray_type == "mix45":
            mp.prec += 200
            ray_norm = rg.get_normal_ray(tor, uv[0], uv[1], eps)
            ray_graz = rg.get_grazing_ray(tor, uv[0], uv[1], pos_epsilon=eps)
            cos45 = mp.cospi(0.25)
            ray = (ray_norm[0], tuple(mp_const(c) for c in (cos45*ray_norm[1]+cos45*ray_graz[1])))

        tracer = AlgTracer(solver)
        tracer.begin()
        mp.prec = prec
        tp_dtb = tor.distance_to_boundary(ray[0], -ray[1], solver)
        tp_dtb = None if tp_dtb is None else mp_const(tp_dtb)
        tracer.end()
        tp_logstr = tracer.simple_logstring()

        mp.prec = HIGH_PREC
        tracer.begin()
        hp_dtb = tor.distance_to_boundary(ray[0], -ray[1], solver)
        hp_dtb = None if hp_dtb is None else mp_const(hp_dtb)
        tracer.end()
        hp_logstr = tracer.simple_logstring()

        return {
            "tp_log": [tp_logstr],
            "tp_dtb": [tp_dtb],
            "hp_log": [hp_logstr],
            "hp_dtb": [hp_dtb],
            "eps": [eps]
        }
    full_res = {
            "tp_log": [],
            "tp_dtb": [],
            "hp_log": [],
            "hp_dtb": [],
            "eps": []
        }
    farthest_hit = -1
    exit_code = 0
    for i in range(len(epsilons)):
        new_res = avoidance_result(epsilons[i])
        if not avoid_condition(new_res["tp_dtb"][0], hole_radius): farthest_hit = i #I.e. at this distance, we didn't avoid hitting the toroid on the way out
        full_res = concat_dicts(full_res, new_res)

    iterate_further = True
    if farthest_hit == -1: #None of the distances hit, for now just stop iterating at all
        iterate_further = False
        exit_code = -1
    if farthest_hit == len(epsilons)-1: #All of the distances hit, likewise iteration probably won't help
        iterate_further = False
        exit_code = 1
    
    # Iterate until the farthest distance that hit, and the next distance after that (which avoids) are a certain magnitude apart
    max_count = 100; c = 0
    mp.prec = HIGH_PREC
    ratio_threshold = mp.power(10, log_convergence)

    while iterate_further and (full_res["eps"][farthest_hit+1]/full_res["eps"][farthest_hit] > ratio_threshold) and (c < max_count):
        next_eps = mp.sqrt(full_res["eps"][farthest_hit+1]*full_res["eps"][farthest_hit])
        next_res = avoidance_result(next_eps)
        full_res = insert_dict_at_index(full_res, next_res, farthest_hit+1)
        if not avoid_condition(next_res["tp_dtb"][0], hole_radius): farthest_hit += 1 #I.e. if it avoided the new result, get closer to the hit, and if it hit, get closer to the miss
    mp.prec = prev_prec
    return full_res["eps"][farthest_hit+1], full_res, exit_code


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
    random_uvs = [(mp.rand()*2, (mp.rand()+1.5)) for i in range(uv_count)]
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
    us = [i*2 / mpf(u_count) for i in range(u_count)]
    vs = [i*2 / mpf(v_count) for i in range(v_count)]
    uvs = [(u,v) for u in us for v in vs]
    return uvs_normals_by_distances(uvs, tor=tor, dists=dists,prec=prec)


# 7: Compare intersection point at different distances for different solvers across random spreads of u and v
def uvrand_normals(
    tor: EllipticToroid = EllipticToroid(50, 10, 20),
    dists: list[MpfAble] = [power(10, i) for i in range(4, 12)],
    prec: int = DOUBLE_PREC,
    uv_count: int = 25
) -> dict:
    uvs = [(mp.rand()*2, mp.rand()*2) for i in range(uv_count)]
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
    verbosity: int = 1,
    fail_condition: callable = None,
) -> dict:
    if fail_condition is None:
        fail_condition = lambda dist, set_idx, ray_idx: dist is None or abs(dist-dists[set_idx]) > min(tor.hor_rad, tor.ver_rad) # Any further out and we assume it might be (correctly) detecting the intersection on the other side

    raysets = [[rg.get_normal_ray(tor, uv[0], uv[1], d) for uv in uvs] for d in dists]
    if face_outwards:
        raysets = [[(ray[0], ray[1]*-1) for ray in rayset] for rayset in raysets]
    full_res = compare_raysets_vs_hp(raysets, solver_code, prec, tor, verbosity=verbosity, is_failure=fail_condition)
    full_res["dists"] = dists
    return full_res



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
def deepcopy(item, indiv_item_result: callable = lambda item: item):
    match type(item):
        case blt.dict:
            new_dict = {}
            for key in item:
                new_dict[key] = deepcopy(item[key], indiv_item_result=indiv_item_result)
            return new_dict
        case blt.list:
            return [deepcopy(n, indiv_item_result=indiv_item_result) for n in item]
        case blt.tuple:
            return {
                    "__tuple__": True,
                    "items": [deepcopy(n, indiv_item_result=indiv_item_result) for n in item]
                }
        case _:
            return indiv_item_result(item)


# Serializes some mp data types
def serialize_indiv_mpitem(item):
    match type(item):
        case mpbackend.MPZ_TYPE:
            return {"mpz":int(item)}
        case mp.mpf:
            return {"mpf":serialize(item._mpf_)}
        case mp.mpc:
            return {"mpc":serialize(item._mpc_)}
        case pu.CONST_TYPE: #NOTE: Only serializes to custom constants, with oct precision
            prev = mp.prec
            mp.prec = CONST_PREC
            s = {"mpconst":serialize(item._mpf_)}
            mp.prec = prev
            return s
        case toroid.EllipticToroid:
            return {"toroid":serialize([item.tor_rad, item.hor_rad, item.ver_rad])} 
        case _:
            return item


# Returns a nested copy, where certain mpmath data types are manually serialized
def serialize(item):
    return deepcopy(item, indiv_item_result=serialize_indiv_mpitem)


# Deserialize a dictionary, given that it might represent an individual mpmath item or a tuple
def deserialize_dict(item: dict):
    if len(item) == 1:
        mtype, content = item.popitem()
        match mtype:
            case "mpz": 
                return mpbackend.MPZ_TYPE(content)
            case "mpf":
                return mp.mpf(deserialize(content))
            case "mpc":
                return mp.mpc(deserialize(content[0]), deserialize(content[1]))
            case "mpconst":
                return mp_const(mp.mpf(deserialize(content)))
            case "toroid":
                r, a, b = deserialize(content) # Should return a list
                return EllipticToroid(r, a, b)
    if "__tuple__" in item:
        return tuple(deserialize(it) for it in item["items"])
    else:
        new_dict = {}
        for key in item:
            new_dict[key] = deserialize(item[key])
        return new_dict


# Deserialize an otherwise json-compatible item with mpf types
def deserialize(item):
    match type(item):
        case blt.dict:
            return deserialize_dict(item)
        case blt.list:
            return [deserialize(it) for it in item]
        case _:
            return item


# Returns a new set of random uvs in the given range
def random_uvs(uv_range: list[MpfAble], num_rays: int) -> list[tuple[mpf, mpf]]:
    return [(uv_range[0] + mp.rand()*(uv_range[1]-uv_range[0]), uv_range[2] + mp.rand()*(uv_range[3]-uv_range[2])) for i in range(num_rays)]


# Returns a spread of uvs in the given range with the given u and v densities
def uv_fillrange(uvr: list[mpf, mpf, mpf, mpf], u_density=5, v_density=5):
    prev = mp.prec
    mp.prec = CONST_PREC
    uvs = [
        (
            mp_const(uvr[0] + (uvr[1]-uvr[0])*mpf(u_i) / mpf(u_density)),
            mp_const(uvr[2] + (uvr[3]-uvr[2])*mpf(v_i) / mpf(v_density)),
        ) for u_i in range(u_density+1) for v_i in range(v_density+1)
    ]
    mp.prec = prev
    return uvs


# Returns sets of 3d coordinates of the toroid in numpy arrays
def plot_toroid(toroid: EllipticToroid, precision: int = 1000):
    U = np.linspace(0, 2 * np.pi, precision)
    V = np.linspace(0, 2 * np.pi, precision)
    U, V = np.meshgrid(U, V)

    X = (float(toroid.tor_rad) + float(toroid.hor_rad) * np.cos(V)) * np.cos(U)
    Y = (float(toroid.tor_rad) + float(toroid.hor_rad) * np.cos(V)) * np.sin(U)
    Z = float(toroid.ver_rad) * np.sin(V)
    return X, Y, Z


# Taken from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c to make drawing rays easier
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, xyz, dxyz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    x, y, z = xyz
    dx, dy, dz = dxyz
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)