'''A module with functions to make it simpler to analyze different permutations of algorithms
Compare:
 - Solvers
 - Precision levels
 - Edge cases/epsilon variation therefrom
 - Starting distance of otherwise equivalent rays
'''

from collections.abc import Iterable, Callable
import numpy as np
import mpmath
from mpmath import mpf, mpc, matrix, cos, sin
import toroid

# ---------
# Constants
# ---------


# Common precision levels to test
SINGLE_PREC: int = 23
DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999


# ------------------------------------------------------------
# Utility functions to generate rays which might be edge cases
# ------------------------------------------------------------


# TODO: Could these all be thought of more as a class?
# RayGenerator for a given torus
# function get_normal_ray(u, v, d)
# Rename class to ray_generator possibly
def origins_for_arriving_ray(
    toroid: toriod.EllipticToroid,
    u: mpf,
    v: mpf,
    ray_dir: matrix,
    distances: list[mpf],
) -> list[matrix]:
    '''
    Takes a point on a torus, a direction (defaults to towards the origin), and produces a 
    list of points shifted backwards along the given direction by the given distances
    '''
    start = point_on_toroid(toroid, u, v)
    return [start - dist*ray_dir for dist in distances]


def epsilon_shifts(
    shift_dir: matrix,
    n: int|list[int] = 6,
) -> list[matrix]:
    '''
    Produces a list of vectors in the given direction, logarithmically spaced (base 10)

    Parameters
    ----------
    shift_dir : matrix (3x1)
        The direction the shifts should face. The smallest vectors might not quite match direction
    n : int | list[int]
        The powers of 10 to scale epsilon by. Must be positive. If an int is given, ranges from 0-n
    '''
    eps = mpmath.mp.eps
    if isinstance(n, int):
        n = [i for i in range(n)]
    return [shift_dir * (eps*pow(10, ni)) for ni in n]


def calc_grazing_ray(
    toroid: toroid.EllipticToroid,
    u: mpf = 0,
    v: mpf = 0,
    yaw: mpf = None,
    distance: mpf = None,
    epsilon: mpf = None,
) -> tuple[matrix, matrix]:
    '''
    Calculates a ray of ray_src and ray_dir which grazes the given toroid at a specified location.

    Parameters
    ----------
    u, v : mpf
        Position on the toroid surface, which is parameterized as follows:
         - x = cos(u)*(tor_rad + hor_rad*cos(v))
         - y = sin(u)*(tor_rad + hor_rad*cos(v))
         - z = ver_rad * sin(v)
    yaw : mpf, default None
        If desired, rotate the resulting direction around the surface normal by this many radians.
        Note: for v < pi/2, v > 3pi/2, some yaw values pierce the toroid, and at v = 0, only yaw=0 won't.
        If None, leaves ray on the r-z plane
    distance : mpf, default None
        How far backwards to shift the ray origin from the surface point. I.e., for a distance 5,
        then ray_src + distance*ray_dir approaches the surface.
        If None, leaves ray origin at exactly the surface point.
    epsilon : mpf, default None
        How far away from the surface to shift the grazing ray. Negative values go into the toroid.
        If None, leaves ray exactly grazing the surface
    
    Returns
    -------
    (ray_src, ray_dir) : tuple[matrix, matrix]
        The source and direction of the ray.
    '''
    # Find point from uv
    ray_src = matrix([

    ])
    # Find normal of that point
    # Rotate normal vector in r-z plane 90˚ to get a direction vector of the ray
    # If required, rotate that vector by yaw
    # Find a point along the ray such that traveling distance from that point along the way arrives at the point
    # Shift in position along epsilon
    pass

def calc_donut_hole_ray(
    toroid: toroid.EllipticToroid,
    u: mpf = 0,
    distance: mpf = 1,
    start_at_origin: bool = False
) -> tuple[matrix, matrix]:
    '''
    Finds a "donut hole ray", a special case of calc_grazing_ray() which goes through the origin.
    This also means that it scrapes the toroid's inside in the opposite direction.
    '''
    r = toroid.tor_rad
    a = toroid.hor_rad
    v = mpmath.pi - mpmath.acos(a / r)
    surf_point, ray_dir = calc_grazing_ray(toroid, u = u, v = v)
    if start_at_origin:
        ray_src = 0 - distance*ray_dir
    else:
        ray_src = surf_point - distance*ray_dir
    return ray_src, ray_dir


# -------------------------------------------------------------
# Utility functions to compare solutions from different solvers
# -------------------------------------------------------------


def intersections_by_solver(
    toroid: toroid.EllipticToroid,
    ray_src: Iterable[mpf], 
    ray_dir: Iterable[mpf],
    solvers: list[type|str] = ["Ferrari", "Alg1010"],
    prec: int = DOUBLE_PREC
) -> dict:
    '''
    Returns a dictionary of intersection results for the given ray torus combination.

    Parameters
    ----------
    toroid
    '''
    pass


# -------------------------------------
# Other miscellaneous utility functions
# -------------------------------------


def rotation_matrix_3d(
    axis: Iterable[mpf],
    theta: mpf
) -> matrix:
    pass


def points_along_ray(
    ray_src: matrix,
    ray_dir: matrix,
    distances: list[mpf]
) -> list[matrix]:
    '''
    Shorthand to find points at the given distance from a start point in a given direction
    '''
    return [ray_src + dist*ray_dir for dist in distances]


def point_on_toroid(toroid: toroid.EllipticToroid, u: mpf, v: mpf) -> matrix:
    '''
    Shorthand to find a point on the given torus using parameterized surface coordinates u & v
    '''
    r = toroid.tor_rad
    a = toroid.hor_rad
    b = toroid.ver_rad
    return matrix([
       cos(u) * (r + a*cos(v)),
       sin(u) * (r + a*cos(v)),
       b * sin(v)
    ])