# A script to run test cases of toroid-ray.py


# TODO: do this in a more pythonic way
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)
#

import math
import itertools
import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pprint
import pytest
import inspect
import toroid_ray
from toroid_ray import Toroid
import toroid_util

printer = pprint.PrettyPrinter(indent=4)

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)

PLOT_RESULTS: bool = True
PLOT_COLORS: list[tuple] = [
    (0, 0, 0.7),
    (0, 0.5, 0.5),
    (0.3, 0.6, 0),
    (0.7, 0.5, 0),
    (0.8, 0, 0),
]


# Returns arrays x y z representing a toroidal surface with the given precision, that matplotlib can then plot
# randomize: how much to randomly vary each u and v by, relative to their step sizes
def plot_toroid(toroid: Toroid, precision: int = 1000, randomize: float = 0):
    rand_seed = 1997
    rng = np.random.default_rng(seed=rand_seed)
    rng.uniform(-randomize, randomize, precision)

    U = np.linspace(0, 2 * np.pi, precision)
    V = np.linspace(0, 2 * np.pi, precision)
    U, V = np.meshgrid(U, V)
    U += rng.uniform(-randomize, randomize, (precision, precision))
    V += rng.uniform(-randomize, randomize, (precision, precision))

    X = (toroid.r + toroid.a * np.cos(V)) * np.cos(U)
    Y = (toroid.r + toroid.a * np.cos(V)) * np.sin(U)
    Z = toroid.b * np.sin(V)
    return X, Y, Z


# Quick shorthand to check if two arrays are equivalent
def check_equal(a, b, rel_tol=1e-09, abs_tol=0.0):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert math.isclose(a[i], b[i], rel_tol=rel_tol, abs_tol=abs_tol)


# Plots the given toroid and matching lists of ray sources, ray directions, and intersection t lists.
# Can also have showing deferred, to display anything else on top of it.
def display_intersections(
    tor: Toroid,
    rays_src: list[np.array],
    rays_dir: list[np.array],
    t_lists: list[list],
    defer_showing: bool = False,
    manual_colors: list[tuple] = None,
):
    # print("directions given: " + str(rays_dir))
    if manual_colors is None:
        manual_colors = [None] * len(t_lists)

    pad = (tor.r + tor.a) * 1.25
    x, y, z = plot_toroid(tor, precision=100)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.axes.set_xlim3d(left=-pad, right=pad)
    ax.axes.set_ylim3d(bottom=-pad, top=pad)
    ax.axes.set_zlim3d(bottom=-pad, top=pad)
    ax.plot_surface(x, y, z, antialiased=True, color="orange", zorder=0)
    for i in range(len(rays_src)):
        ts = np.array(t_lists[i])
        # Check out the manual color,
        col = manual_colors[i]
        if col is None:  # And set procedurally if unspecified
            col = PLOT_COLORS[len(ts)]
        s = rays_src[i]
        d = rays_dir[i].astype(np.dtype("float64"))
        # print("direction pre-norm: " + str(d))
        d /= la.norm(d)
        # print("direction post-norm: " + str(d))
        t_ray = pad * 2
        e = s + d * t_ray
        # ax.set_color_cycle([col])
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=col, zorder=1)  # Ray
        # ax.text(e[0], e[1], e[2],"Ray #"+str(i)+", "+str(len(ts))+" int.s",color=col) #Label

        # Intersections
        ax.scatter(
            s[0] + ts * d[0],
            s[1] + ts * d[1],
            s[2] + ts * d[2],
            "zorder=3\nalpha=1",
            c=to_hex(col),
        )
    # ax.legend()
    method = inspect.stack()[1]
    supermethod = inspect.stack()[2]
    plt.title(
        f"Test: {method.function} (in {supermethod.function})"
    )  # Label with name of method that called this one
    if not defer_showing:
        plt.show()
    return ax


# Secondary shorthand to test all intersection methods for a given toroid-ray combo.
def assert_intersections(
    tor: Toroid,
    ray_src: np.array,
    ray_dir: np.array,
    known_t_list: list,
    show_results: bool = PLOT_RESULTS,
):
    # print("direction: " + str(ray_dir))
    # ray_dir /= la.norm(ray_dir)
    # poly = tor.ray_intersection_polynomial(ray_src, ray_dir)
    t_lists = [known_t_list]
    # 1st: basic numpy root method
    np_solver = toroid_util.real_roots_numpy
    np_t_list, np_info = tor.ray_intersections(ray_src, ray_dir, np_solver)
    t_lists.append(np_t_list)
    # 2nd: ferrari limited, as seen on https://www.desmos.com/3d/2ba985c474
    fr_solver = toroid_util.real_roots_ferrari
    fr_t_list, fr_info = tor.ray_intersections(ray_src, ray_dir, fr_solver)
    t_lists.append(fr_t_list)
    # 3rd: ferrari full, which can edge around complex numbers and was taken from StackExchange
    fr_solver_SE = toroid_util.real_roots_ferrari_SE
    fr_t_list_SE, fr_info_SE = tor.ray_intersections(ray_src, ray_dir, fr_solver_SE)
    t_lists.append(fr_t_list_SE)
    # TODO: Test more root methods? Should each one have a distinct color,
    # or is showing the number of intersections more important?

    # Colors, draw known t in a specific color to distinguish from the other ones
    colors = [(0, 0, 0)] + [None] * (len(t_lists) - 1)

    if show_results:
        display_intersections(
            tor, [ray_src] * len(t_lists), [ray_dir] * len(t_lists), t_lists=t_lists
        )

    # After display, run actual test
    check_equal(np_t_list, known_t_list)
    check_equal(fr_t_list, known_t_list)
    check_equal(fr_t_list_SE, known_t_list)


# Test 1: a toroidal surface is graphed as expected
def tesnt_create():
    # tor = Toroid(5, 1, 0.3, pos=np.array([5, 5, 5]))
    tor = Toroid(5, 1, 0.3)
    pad = 7
    x, y, z = plot_toroid(tor)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection="3d")
    ax.axes.set_xlim3d(left=-pad, right=pad)
    ax.axes.set_ylim3d(bottom=-pad, top=pad)
    ax.axes.set_zlim3d(bottom=pad, top=pad)
    ax.plot_surface(x, y, z, antialiased=True, color="orange")
    plt.show()


# Test 2: compare basic generated polynomial to a known desmos test case (https://www.desmos.com/3d/3fdrpdcjjw)
def test_polynom():
    tor = Toroid(3.05, 1, 0.5)
    s = np.array([1.4, 2.9, 2.6]) - np.array([0.96, -0.25, 1.3])
    u = np.array([0.63, -0.2, -1.66])
    u /= la.norm(u)

    poly = tor.ray_intersection_polynomial(s, u, verbose=True)
    desmos = np.array([1, -5.60371151562, 21.4844076830, -38.1674209108, 19.9891068153])

    check_equal(poly, desmos)
    # assert np.allclose(poly, desmos)
    # assert poly[0] == 1
    # assert poly[1] == -5.60371151562
    # assert poly[2] ==  21.4844076830
    # assert poly[3] == -38.1674209108
    # assert poly[4] ==  19.9891068153


# Test 3: make sure all the polynomial solvers agree with each other to a certain extent
def test_solvers():
    return None


# Ray through the center shouldn't intersect with the toroid
def test_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 0, 1])
    u = np.array([0, 0, -1])
    assert len(tor.ray_intersections(s, u, toroid_util.real_roots_ferrari_SE)[0]) == 0


# Ray starting inside and going out away from center should have 1 intersection
def test_inside_out():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5, 0])
    u = np.array([0, 1, 0])
    # assert len(tor.ray_intersections_np(s, u)) == 1
    # check_equal(tor.ray_intersections_np(s, u), [1])
    assert_intersections(tor, s, u, [1])


# Ray starting inside and going towards center should have 3 intersections
def test_inside_through_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    assert_intersections(tor, s, u, [1, 9, 11])


def test_inside_through_center_diag():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    diag = np.array([0.5**0.5, 0.5**0.5, 0])
    s = 5 * diag
    u = -1 * diag
    assert_intersections(tor, s, u, [1, 9, 11])


def test_inside_through_center_diagoffset():
    tor = Toroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    diag = np.array([0.5**0.5, 0.5**0.5 * 0.8, 0.03])
    diag /= la.norm(diag)
    s = 5 * diag
    u = -1 * diag
    inters, inter_locals = tor.ray_intersections(
        s, u, toroid_util.real_roots_ferrari_SE
    )
    display_intersections(tor, [s], [u], [inters])
    assert len(inters) == 3


# We wanted a test case of a ray going through the center of a toroid but at an angle so it grazes two edges of the toroid hole
def internal_graze_angle(tor: Toroid):
    th0 = math.asin(tor.a / tor.r)  # Angle if a and b were equal
    v0 = math.cos(th0) * tor.a
    v1 = v0 * (tor.b / tor.a)
    w = math.sin(th0) * tor.a
    th1 = math.atan(v1 / (tor.r - w))
    return th1


# Test some grazing cases


# Test that normals are working properly, just display for now
def test_normals_pointshot():
    tor = Toroid(9.8733, 4.387, 1.73)
    # Fan ray source
    s = np.array([-15, 0, 0])
    thetas = np.linspace(-0.2, 0.2, 3)
    phis = np.linspace(-0.1, 0.1, 3)
    us = []
    t_sets = []
    p_sets = []
    for th in thetas:
        for phi in phis:
            u = np.array(
                [
                    math.cos(phi) * math.cos(th),
                    math.cos(phi) * math.sin(th),
                    math.sin(phi),
                ]
            )
            us.append(u)
            t_set, t_info = tor.ray_intersections(
                s, u, toroid_util.real_roots_ferrari_SE
            )
            t_sets.append(t_set)
            p_set, p_info = tor.ray_intersections(
                s, u, toroid_util.real_roots_ferrari_SE, return_points=True
            )
            p_sets.append(p_set)
    ss = [s] * len(us)
    ax = display_intersections(tor, ss, us, t_sets, defer_showing=True)
    # ax = display_intersections(tor, ss, us, t_sets, defer_showing=True)
    for p_set in p_sets:
        for p in p_set:
            n = tor.surface_normal(p)
            o = p + n * 4
            ax.plot([p[0], o[0]], [p[1], o[1]], [p[2], o[2]], zorder=3)
    plt.show()


# Second normals test, this time using parametric points surrounding the toroid
def test_normals_wrap():
    tor = Toroid(8.31, 1.12, 2.3)
    density = 15
    X, Y, Z = plot_toroid(tor, density)
    ax = display_intersections(tor, [], [], [], defer_showing=True)
    for i, j in itertools.product(range(density), range(density)):
        pos = np.array([X[i, j], Y[i, j], Z[i, j]])
        norm = tor.surface_normal(pos)
        if norm is None:
            continue
        outer_norm = pos + norm * 3
        p = pos
        o = outer_norm
        ax.plot([p[0], o[0]], [p[1], o[1]], [p[2], o[2]], zorder=3)
    plt.show()


# Second normals test, same as second but each being slightly offset
def test_normals_randwrap():
    tor = Toroid(6.77, 1.31, 0.324)
    density = 5
    X, Y, Z = plot_toroid(tor, density, randomize=1)
    ax = display_intersections(tor, [], [], [], defer_showing=True)
    for i, j in itertools.product(range(density), range(density)):
        pos = np.array([X[i, j], Y[i, j], Z[i, j]])
        norm = tor.surface_normal(pos)
        if norm is None:
            continue
        outer_norm = pos + norm * 3
        p = pos
        o = outer_norm
        ax.plot([p[0], o[0]], [p[1], o[1]], [p[2], o[2]], zorder=3)
    plt.show()


# Create random assortment of points and test if they're in the toroid
def test_random_piv():
    rand_seed = 1997
    rng = np.random.default_rng(seed=rand_seed)

    c = np.array([3.1, 123, 9.77])
    tor = Toroid(13.11, 2.71, 1.997)

    w = (tor.r + tor.a) * 1.3
    h = tor.b + w - (tor.r + tor.a)

    num_points = 1000
    x = rng.uniform(c[0] - w, c[0] + w, num_points)
    y = rng.uniform(c[1] - w, c[1] + w, num_points)
    z = rng.uniform(c[2] - h, c[2] + h, num_points)

    inside = [[], [], []]
    outside = [[], [], []]
    for i in range(num_points):
        point = np.array([x[i], y[i], z[i]])
        if tor.point_in_volume(point):
            inside[0].append(x[i])
            inside[1].append(y[i])
            inside[2].append(z[i])
        else:
            outside[0].append(x[i])
            outside[1].append(y[i])
            outside[2].append(z[i])

    inside = np.array(inside)
    outside = np.array(outside)
    ax = display_intersections(tor, [], [], [], defer_showing=True)
    ax.scatter(inside[0, :], inside[1, :], inside[2, :], c=to_hex((0, 0.8, 0)))
    ax.scatter(outside[0, :], outside[1, :], outside[2, :], c=to_hex((0.9, 0, 0)))
    plt.show()


# Test if points on the edge of the toroid are considered in / out
def test_random_edge():
    rand_seed = 1997
    rng = np.random.default_rng(seed=rand_seed)

    tor = Toroid(13.11, 2.71, 1.997)

    w = (tor.r + tor.a) * 1.3
    h = tor.b + w - (tor.r + tor.a)

    num_points = 1000
    U = rng.uniform(0, 2 * np.pi, num_points)
    V = rng.uniform(0, 2 * np.pi, num_points)
    x = (tor.r + tor.a * np.cos(V)) * np.cos(U)
    y = (tor.r + tor.a * np.cos(V)) * np.sin(U)
    z = tor.b * np.sin(V)

    inside = [[], [], []]
    outside = [[], [], []]
    for i in range(num_points):
        point = np.array([x[i], y[i], z[i]])
        if tor.point_in_volume(point):
            inside[0].append(x[i])
            inside[1].append(y[i])
            inside[2].append(z[i])
        else:
            outside[0].append(x[i])
            outside[1].append(y[i])
            outside[2].append(z[i])

    inside = np.array(inside)
    outside = np.array(outside)
    ax = display_intersections(tor, [], [], [], defer_showing=True)
    ax.scatter(inside[0, :], inside[1, :], inside[2, :], c=to_hex((0, 0.8, 0)))
    ax.scatter(outside[0, :], outside[1, :], outside[2, :], c=to_hex((0.9, 0, 0)))
    plt.show()


def test_rootfinders_random():

    num_trials = 10000
    min_power = -48
    max_power = 48
    for i in range(num_trials):
        roots_fr = None
        roots_np = None
        roots_frnp = None
        # coeff_exps = glob_rng.uniform(min_power, max_power, 4)
        coeff_exps = glob_rng.normal(0, max_power / 2, 4)
        coeff_exps = [1] + [float(n) for n in coeff_exps]

        roots_np, info_np = toroid_util.real_roots_numpy(coeff_exps)
        # roots_fr, root_locals = toroid_util.real_roots_ferrari(coeff_exps)
        roots_fr, root_locals = toroid_util.real_roots_ferrari_SE(coeff_exps)
        # roots_frnp, root_locals_np = toroid_util.real_roots_ferrari_npdebug(coeff_exps)
        root_locals = printer.pformat(root_locals)

        roots_fr = sorted(roots_fr)
        roots_np = sorted(roots_np)
        check_equal(roots_np, roots_fr, abs_tol=1e-6)


def test_rootfinders_desmos():
    desmos_coeffs = [1, 10.3487176734, 48.5955496218, 102.422553557, 69.3726451627]
    np_roots, np_info = toroid_util.real_roots_numpy(desmos_coeffs)
    fr_roots, fr_locals = toroid_util.real_roots_ferrari(desmos_coeffs)
    fr_roots = sorted(fr_roots)
    np_roots = sorted(np_roots)

    check_equal(np_roots, fr_roots)
