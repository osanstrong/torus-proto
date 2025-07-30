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
from mpmath import mp
from mpmath import mpf, mpc, matrix, cos, sin, sqrt, sign, norm
from src.toroid import EllipticToroid, sq

# ---------
# Constants
# ---------


# Common precision levels to test
SINGLE_PREC: int = 23
DOUBLE_PREC: int = 53
QUAD_PREC: int = 113
HIGH_PREC: int = 999


# ------------------------------------------------------------
# Utility functions to generate notable kinds of rays
# ------------------------------------------------------------


def get_normal_ray(
    toroid: EllipticToroid,
    u: mpf,
    v: mpf,
    dist: mpf,
) -> tuple[matrix, matrix]:
    '''
    Takes a point on a torus, a direction (defaults to towards the origin), and produces a ray
    towards it whose source is shifted backwards along the given direction by the given distances

    Parameters
    ----------
    toroid : EllipticToroid
        The toroidal surface to generate the ray for
    u, v : mpf
        The coordinates on the toroidal surface for the ray to aim at
    dist : mpf
        How far back the ray should start; negative distance means it starts after the surface
    '''
    start = point_on_toroid(toroid, u, v)
    mp.prec += 200
    ray_dir = -toroid.surface_normal(start)
    start -= dist*ray_dir
    mp.prec -= 200
    return start, ray_dir


def get_grazing_ray(
    toroid: EllipticToroid,
    u: mpf = 0,
    v: mpf = 0,
    yaw: mpf = None,
    distance: mpf = None,
    pos_epsilon: mpf = None,
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
    pos_epsilon : mpf, default None
        How far away from the surface to shift the grazing ray. Negative values go into the toroid.
        If None, leaves ray exactly grazing the surface
    
    Returns
    -------
    (ray_src, ray_dir) : tuple[matrix, matrix]
        The source and direction of the ray.
        Picks the direction such that ray points toward the z axis. 
    '''
    # Find point from uv
    ray_src = point_on_toroid(toroid, u, v)
    # Find normal of that point
    mp.prec += 200
    srf_nor = toroid.surface_normal(ray_src)
    nor_mag = norm(srf_nor, 2)
    # Rotate normal vector in r-z plane 90˚ to get a direction vector of the ray
    x, y, z = srf_nor
    r = sqrt(sq(x) + sq(y)) * -sign(cos(v)) # If the vector is on 'hole' of donut, r is negative
    zr = z/r
    graze_dir = matrix([x*zr, y*zr, -r])
    graze_mag = norm(graze_dir, 2)
    if sin(v) > 0: # If the vector is on 'topside', flip so that the ray is always approaching the z axis
        graze_dir *= 1
    # If required, rotate that vector by yaw
    # Find a point along the ray such that traveling distance from that point along the way arrives at the point
    if not distance is None:
        ray_src -= distance*graze_dir
    # Shift in position along epsilon
    if not pos_epsilon is None:
        ray_src += pos_epsilon*matrix(srf_nor)
    mp.prec -= 200
    return ray_src, graze_dir


def get_donut_hole_ray(
    toroid: EllipticToroid,
    u: mpf,
    distance: mpf,
    ang_epsilon: mpf = None
) -> tuple[matrix, matrix]:
    '''
    Finds a "donut hole ray", a special case of calc_grazing_ray() which goes through the origin.
    This also means that it scrapes the toroid's inside in the opposite direction.
    Starts at origin instead of at the surface point.

    Parameters
    ----------
    toroid : EllipticToroid
        The toroidal surface to find the donut hole ray for
    u : mpf
        The 'theta' position on the toroidal surface. There's only one pair of vs that produce
        this donut hole ray.
    distance : mpf
        How far back to position the ray from the origin
    ang_epsilon : mpf, default None
        If specified, rotate the ray by this much about the origin, in radians
    '''
    r = toroid.tor_rad
    a = toroid.hor_rad
    v = mpmath.pi - mpmath.acos(a / r)
    surf_point, ray_dir = get_grazing_ray(toroid, u = u, v = v)
    mag0 = norm(ray_dir, 2)
    prev = mpmath.mp.prec
    mpmath.mp.prec = 999
    if not ang_epsilon is None:
        dx, dy, dz = ray_dir
        dr = sqrt(sq(dx) + sq(dy))
        zr = dz/dr
        ray_dir = matrix([
            dx * (cos(ang_epsilon) + zr*sin(ang_epsilon)),
            dy * (cos(ang_epsilon) + zr*sin(ang_epsilon)),
            dz*cos(ang_epsilon) - dr*sin(ang_epsilon)
        ])
    mpmath.mp.prec = prev
    ray_src = 0 - distance*ray_dir
    mag = norm(ray_dir, 2)
    return ray_src, ray_dir


# -------------------------------------------------------------
# Utility functions to compare solutions from different solvers
# -------------------------------------------------------------


def intersections_by_solver(
    toroid: EllipticToroid,
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


def point_on_toroid(toroid: EllipticToroid, u: mpf, v: mpf) -> matrix:
    '''
    Shorthand to find a point on the given torus using parameterized surface coordinates u & v
    '''
    mp.prec += 100
    r = toroid.tor_rad
    a = toroid.hor_rad
    b = toroid.ver_rad
    point = matrix([
       cos(u) * (r + a*cos(v)),
       sin(u) * (r + a*cos(v)),
       b * sin(v)
    ])
    mp.prec -= 100
    return point


def uv_norm(toroid: EllipticToroid, u: mpf, v: mpf) -> matrix:
    '''
    Shorthand for the normal vector to the toroid at a given u, v
    '''
    return matrix(toroid.surface_normal(point_on_toroid(toroid, u, v)))


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