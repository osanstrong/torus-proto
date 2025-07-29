'''
A testing module to make sure that the ray generating functions in ray_generator work properly
'''


import numpy as np
from mpmath import mpf, matrix as mat, pi, mp, chop, norm, cos
import src.analysis.ray_generator as rg
from src.toroid import EllipticToroid


# Ensure the generated grazing ray is orthogonal to the normal vector
def test_grazing_ray_grazes():
    mp.prec = 113
    num_trials = 100
    globr = np.random.default_rng(1997)
    r_range = 10
    for i in range(num_trials):
        rands = [abs(r) for r in globr.normal(0, r_range, 3)]
        # Random torus, but make sure major radius is larger than horizontal minor radius
        tor = EllipticToroid(rands[0]+rands[1], rands[1], rands[2])

        u, v = globr.uniform(0, 2*pi, 2)
        norm = rg.uv_norm(tor, u, v)
        graze = rg.get_grazing_ray(tor, u=u, v=v)[1]
        
        dot = norm.T * graze
        assert chop(dot[0]) == 0

        
# Ensure the donut hole ray goes through the center
def test_donut_hole_ray_is_center():
    mp.prec = 113
    num_trials = 100
    globr = np.random.default_rng(1984)
    r_range = 10
    for i in range(num_trials):
        rands = [abs(r) for r in globr.normal(0, r_range, 3)]
        tor = EllipticToroid(rands[0]+rands[1], rands[1], rands[2])

        u = globr.uniform(0, 2*pi)
        src, should_clear = rg.get_donut_hole_ray(tor, u, 25, ang_epsilon=0.01)

        inters = tor.ray_intersection_distances(src, should_clear, "tt")
        assert len(inters) == 0


# Ensure generated normal rays are normal to the surface, and intersect at the surface
def test_normal_rays():
    mp.prec = 113
    num_trials = 100
    globr = np.random.default_rng(2001)
    r_range = 10
    for i in range(num_trials):
        rands = [abs(r) for r in globr.normal(0, r_range, 3)]
        tor = EllipticToroid(rands[0]+rands[1], rands[1], rands[2])

        u, v = globr.uniform(0, 2*pi, 2)
        u = 0
        dist = 100
        src, should_hit = rg.get_normal_ray(tor, u, v, dist)
        surf = rg.point_on_toroid(tor, u, v)
        nor = rg.uv_norm(tor, u, v)

        assert isclose(norm(should_hit, 2), norm(should_hit.T*nor, 2))
        if not pi/2 < v < 3*pi/2: # From outside: should be first hit
            hits = tor.ray_intersection_distances(src, should_hit, "tt")
            hit_points = [src + dist*should_hit for dist in hits]
            first_hit = tor.distance_to_boundary(src, should_hit, "tt")
            assert isclose(first_hit, dist, tol=1e-16)
        else: # Otherwise, should always be second to last hit (yes sometimes these mean the same thing)
            hits = tor.ray_intersection_distances(src, should_hit, "tt")
            assert isclose(sorted(hits)[-2], dist, tol=1e-16)


# ---------------------
# Misc helper functions
# ---------------------


def isclose(a: mpf, b: mpf, tol: mpf = None) -> bool:
    return chop(abs(a-b), tol=tol) == 0