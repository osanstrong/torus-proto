# A script to run test cases of toroid-ray.py

from math import isclose
import numpy as np
from numpy.linalg import norm
import pytest
import src.toroid
from src.toroid import EllipticToroid

glob_rand_seed = 1999
glob_rng = np.random.default_rng(seed=glob_rand_seed)
COS_45 = np.sqrt(0.5)

# Quick shorthand to check if two arrays are equivalent
def assert_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    assert np.allclose(a, b, rtol=rel_tol, atol=abs_tol)

# In lieu of any rootfinder implementations, just use numpy for now
def calc_real_roots_numpy(coeffs: list[float]) -> list[float]:
    """
    Solves for roots of the polynomial using numpy's eigenvalue / matrix implementation
    """
    all_roots = np.roots(coeffs)
    real_roots = all_roots[np.isreal(all_roots)]
    return [float(np.real(r)) for r in real_roots]

# Secondary shorthand to test all intersection methods for a given toroid-ray combo.
def assert_intersections(
    tor: EllipticToroid,
    ray_src: np.array,
    ray_dir: np.array,
    known_t_list: list,
):
    np_t_list = tor.ray_intersection_distances(ray_src, ray_dir, calc_real_roots_numpy)
    assert_close(sorted(np_t_list), known_t_list)

# Like assert_intersections, but testing the method which returns final points instead of distances
def assert_intersection_points(
    tor: EllipticToroid,
    ray_pos: np.array,
    ray_dir: np.array,
    known_point_list: list[np.array]
):
    np_point_list = tor.ray_intersection_points(ray_pos, ray_dir, calc_real_roots_numpy)
    for i in range(len(np_point_list)):
        assert_close(np_point_list[i], known_point_list[i])


# Ray through the center shouldn't intersect with the toroid
def test_center():
    tor = EllipticToroid(5, 1, 1)
    s = np.array([0, 0, 1])
    u = np.array([0, 0, -1])
    assert len(tor.ray_intersection_distances(s, u, calc_real_roots_numpy)) == 0
    assert tor.distance_to_boundary(s, u, calc_real_roots_numpy) is None


# Ray starting inside and going out away from center should have 1 intersection
def test_inside_out():
    tor = EllipticToroid(5, 1, 1)
    s = np.array([0, 5, 0])
    u = np.array([0, 1, 0])
    assert_intersections(tor, s, u, [1])
    assert_intersection_points(tor, s, u, [np.array([0,6,0])])
    assert isclose(tor.distance_to_boundary(s, u, calc_real_roots_numpy), 1)


# Ray starting inside and going towards center should have 3 intersections
def test_inside_through_center():
    tor = EllipticToroid(5, 1, 1)
    s = np.array([0, 5.0, 0])
    u = np.array([0, -1.0, 0])
    assert_intersections(tor, s, u, [1, 9, 11])
    assert_intersection_points(tor, s, u, [
        np.array([0,4,0]),
        np.array([0,-4,0]),
        np.array([0,-6,0])
    ])
    assert isclose(tor.distance_to_boundary(s, u, calc_real_roots_numpy), 1)


# Repeat but along the a 45 degree diagonal
def test_inside_through_center_diag():
    tor = EllipticToroid(5, 1, 1)
    base_ang = np.pi / 4
    for e in [s * 10**-p for s in [-1, 1] for p in range(1,10)]:
        ang = base_ang + e
        diag = np.array([np.cos(ang), np.sin(ang), 0])
        s = 5 * diag
        u = -1 * diag
        assert_intersections(tor, s, u, [1, 9, 11])
        assert_intersection_points(tor, s, u, [4*diag, -4*diag, -6*diag])
        assert isclose(tor.distance_to_boundary(s, u, calc_real_roots_numpy), 1)


# Repeat but with a further offset
def test_inside_through_center_diagoffset():
    tor = EllipticToroid(5, 1, 1)
    diag = np.array([COS_45, COS_45 * 0.8, 0.03])
    diag /= norm(diag)
    s = 5 * diag
    u = -1 * diag
    inters = tor.ray_intersection_distances(
        s, u, calc_real_roots_numpy
    )
    assert len(inters) == 3


# Ray straight up from above the torus shouldn't intersect, and ray straight down from the same should intersect twice
def test_vertical():
    tor = EllipticToroid(5, 1, 1)
    s = np.array([0, 5.0, 2.3])
    u_up = np.array([0, 0, 1.0])
    u_down = np.array([0, 0, -1.0])
    assert len(tor.ray_intersection_distances(s, u_up, calc_real_roots_numpy)) == 0
    assert tor.distance_to_boundary(s, u_up, calc_real_roots_numpy) is None
    assert_intersections(tor, s, u_down, [1.3, 3.3])
    assert_intersection_points(tor, s, u_down, [np.array([0,5.0,1.0]), np.array([0,5.0,-1.0])])
    assert isclose(tor.distance_to_boundary(s, u_down, calc_real_roots_numpy), 1.3)


# Points that should be inside
def test_inside_points():
    tor = EllipticToroid(5, 1, 1)
    should_be_inside = [
        [5.0, 0, 0],
        [0, 5.0, 0],
        [5.0, 0, 0.9],
        [5.0 * COS_45, 5.0 * COS_45, 0.9]
    ]
    for pos in should_be_inside:
        assert tor.point_sense(pos) == -1


# Points that should be outside
def test_outside_points():
    tor = EllipticToroid(5, 1, 1)
    should_be_outside = [
        [0,0,0],
        [0,3.9,0],
        [3.9,0,0],
        [-3.9,0,0],
        [5.0,0,1.1],
        [6.1,0,0]
    ]
    for pos in should_be_outside:
        assert tor.point_sense(pos) == 1


#Points that should be on the edge
def test_edge_points():
    tor = EllipticToroid(5,1,1)
    should_be_on = [
        [5.0,0,1.0],
        [4.0,0,0],
        [6.0,0,0]
    ]
    for pos in should_be_on:
        assert tor.point_sense(pos) == 0

#Easily tested normal vectors
def test_normals_basic():
    tor = EllipticToroid(5,1,1)
    assert_close(tor.surface_normal([5.0,0,1.0]), np.array([0,0,1.0]))
    assert_close(tor.surface_normal([5.0,0,-1.0]), np.array([0,0,-1.0]))
    assert_close(tor.surface_normal([6.0,0,0]), np.array([1.0,0,0]))
    assert_close(tor.surface_normal([4.0,0,0]), np.array([-1.0,0,0]))
    assert_close(tor.surface_normal([0,6.0,0]), np.array([0,1.0,0]))
    assert_close(tor.surface_normal([0,4.0,0]), np.array([0,-1.0,0]))


# Verify input tests
def test_value_errors():
    # Each possible way to mess up a toroid
    with pytest.raises(ValueError): tor = EllipticToroid(-1, 1, 1)
    with pytest.raises(ValueError): tor = EllipticToroid(1, -1, 1)
    with pytest.raises(ValueError): tor = EllipticToroid(1.2, 1, -1)
    with pytest.raises(ValueError): tor = EllipticToroid(0, 1, 1)
    with pytest.raises(ValueError): tor = EllipticToroid(1, 0, 1)
    with pytest.raises(ValueError): tor = EllipticToroid(1.2, 1, 0)
    with pytest.raises(ValueError): tor = EllipticToroid(0.9, 1, 1)
    with pytest.raises(ValueError): tor = EllipticToroid(1, 1, 1)

    tor = EllipticToroid(2, 1, 1)
    start = np.array([0.2,1,3])
    zero = np.array([0,0,0])
    
    # Intersectors should not acccept zero vectors for direction
    with pytest.raises(ValueError): inters = tor.ray_intersection_distances(start, zero, calc_real_roots_numpy)
    with pytest.raises(ValueError): inters = tor.ray_intersection_points(start, zero, calc_real_roots_numpy)
    with pytest.raises(ValueError): dist = tor.distance_to_boundary(start, zero, calc_real_roots_numpy)

    # Surface normal should only work on surface points
    with pytest.raises(ValueError): norm = tor.surface_normal([0,0,0])
    with pytest.raises(ValueError): norm = tor.surface_normal([2,0,12])
    with pytest.raises(ValueError): norm = tor.surface_normal([0,12,0]) 

    mag_one = [0,1,0]
    not_one = [1,2,3]
    # Intersection methods should only accept vectors with mag 1
    with pytest.raises(ValueError): inters = tor.ray_intersection_distances(start, not_one, calc_real_roots_numpy)
    with pytest.raises(ValueError): inters = tor.ray_intersection_points(start, not_one, calc_real_roots_numpy)
    with pytest.raises(ValueError): dist = tor.distance_to_boundary(start, not_one, calc_real_roots_numpy) 
    inters = tor.ray_intersection_distances(start, mag_one, calc_real_roots_numpy)
    inters = tor.ray_intersection_points(start, mag_one, calc_real_roots_numpy)
    dist = tor.distance_to_boundary(start, mag_one, calc_real_roots_numpy)
    


#### Helper function tests


# Compare basic generated polynomial to a known desmos test case (https://www.desmos.com/3d/3fdrpdcjjw)
def test_polynom():
    tor = EllipticToroid(3.05, 1, 0.5)
    s = np.array([1.4, 2.9, 2.6]) - np.array([0.96, -0.25, 1.3])
    u = np.array([0.63, -0.2, -1.66])
    u /= norm(u)

    poly = tor._ray_intersection_polynomial(s, u)
    desmos = np.array([1, -5.60371151562, 21.4844076830, -38.1674209108, 19.9891068153])

    assert_close(poly, desmos)