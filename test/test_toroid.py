# A script to run test cases of toroid-ray.py


# TODO: do this in a more pythonic way
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)
#

import math
import random
import numpy as np
import numpy.linalg as la
import pytest
import inspect
import toroid
from toroid import Toroid

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

# Quick shorthand to check if two arrays are equivalent
def check_equal(a, b, rel_tol=1e-09, abs_tol=0.0):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert math.isclose(a[i], b[i], rel_tol=rel_tol, abs_tol=abs_tol)

# In lieu of any rootfinder implementations, just use numpy for now
def real_roots_numpy(coeffs: list[float]) -> (list[float], dict):
    """
    Solves for roots of the polynomial using numpy's eigenvalue / matrix implementation
    """
    all_roots = np.roots(coeffs)
    real = all_roots[np.isreal(all_roots)]
    real_list = [float(np.real(r)) for r in real]
    return real_list, locals()

# Secondary shorthand to test all intersection methods for a given toroid-ray combo.
def assert_intersections(
    tor: Toroid,
    ray_src: np.array,
    ray_dir: np.array,
    known_t_list: list,
):
    t_lists = [known_t_list]
    # 1st: basic numpy root method
    np_solver = real_roots_numpy
    np_t_list, np_info = tor.ray_intersections(ray_src, ray_dir, np_solver)
    t_lists.append(np_t_list)
    # run actual test
    check_equal(sorted(np_t_list), known_t_list)


# Compare basic generated polynomial to a known desmos test case (https://www.desmos.com/3d/3fdrpdcjjw)
def test_polynom():
    tor = Toroid(3.05, 1, 0.5)
    s = np.array([1.4, 2.9, 2.6]) - np.array([0.96, -0.25, 1.3])
    u = np.array([0.63, -0.2, -1.66])
    u /= la.norm(u)

    poly = tor._ray_intersection_polynomial(s, u)
    desmos = np.array([1, -5.60371151562, 21.4844076830, -38.1674209108, 19.9891068153])

    check_equal(poly, desmos)


# Ray through the center shouldn't intersect with the toroid
def test_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 0, 1])
    u = np.array([0, 0, -1])
    assert len(tor.ray_intersections(s, u, real_roots_numpy)[0]) == 0


# Ray starting inside and going out away from center should have 1 intersection
def test_inside_out():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5, 0])
    u = np.array([0, 1, 0])
    assert_intersections(tor, s, u, [1])


# Ray starting inside and going towards center should have 3 intersections
def test_inside_through_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    assert_intersections(tor, s, u, [1, 9, 11])


# Repeat but along the a 45 degree diagonal
def test_inside_through_center_diag():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    diag = np.array([0.5**0.5, 0.5**0.5, 0])
    s = 5 * diag
    u = -1 * diag
    assert_intersections(tor, s, u, [1, 9, 11])


# Repeat but with a further offset
def test_inside_through_center_diagoffset():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    diag = np.array([0.5**0.5, 0.5**0.5 * 0.8, 0.03])
    diag /= la.norm(diag)
    s = 5 * diag
    u = -1 * diag
    inters, inter_locals = tor.ray_intersections(
        s, u, real_roots_numpy
    )
    assert len(inters) == 3


