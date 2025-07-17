'''
A script to compare and plot results of different solvers
'''
import matplotlib.pyplot as plt
from mpmath import mpf, mp, pi
from src.toroid import EllipticToroid
import src.analysis.ray_generator as rg


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
    dists: list[MpfAble] = ["0.00001", "0.01", 1, 10, 100, 1000]+[10**i for i in range(4, 16)],
    prec: int = DOUBLE_PREC
) -> dict:
    dists = [mpf(d) for d in dists]
    # One torus setup
    tor = EllipticToroid(5, 1, 2)
    u, v = 0, pi / 2 # Let's try directly above the torus to start
    mp.prec = HIGH_PREC
    ray_dir = rg.get_normal_ray(tor, u, v, dist = 1)[1]

    # Combos of dists and solvers
    sources = [rg.get_normal_ray(tor, u, v, dist = d)[0] for d in dists]
    test_solvers = ["fr", "tt", "np"]

    # Final results include each dist and every solver, as well as arbitrarily high precision just to be sure
    final_results: dict = {}
    final_results["dists"] = dists
    # Get results at ludicrous precision just to check
    mp.prec = HIGH_PREC
    highp_results = []
    for i in range(len(dists)):
        ray_src = sources[i]
        hit = ray_src + ray_dir*tor.distance_to_boundary(ray_src, ray_dir, "tt")
        highp_results.append(hit[2]) #Check just z coordinate for now?
    final_results["highp"] = highp_results

    #Then for each solver, repeat at normal precision
    mp.prec = prec
    for solv in test_solvers:
        results = []
        for ray_src in sources:
            dist = tor.distance_to_boundary(ray_src, ray_dir, solv)
            if dist is None:
                results.append(None)
                continue
            hit = ray_src + ray_dir*tor.distance_to_boundary(ray_src, ray_dir, solv)
            results.append(hit[2])
        final_results[solv] = results
    
    return final_results
        