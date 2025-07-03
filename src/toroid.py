from collections.abc import Callable, Iterable, Mapping
from typing import Union
import math
import numpy as np
from numpy import linalg as la
import mpmath
from mpmath import mpf

'''A module for modeling Elliptic Toroid surfaces for ray tracing-like applications, 
specifically Celeritas and ORANGE.

This module contains a class "EllipticToroid" which represents an elliptical toroid surface, with three
functions necessary for ray tracing:
 - The first intersection (if any) with the surface along a given ray
 - The sense of a point relative to the torus, whether inside, on, or outside
 - The normal vector to the surface at a given point

Note
----
Ray-torus intersection and normal formulae taken from Graphics Gems II, Gem V.2 by Joseph M Cychosz, 
Purdue University. Graphics Gems II: ISBN 0-12-064481-9, Published 1991 Academic Press, Inc.

'''


# Type hint for acceptable arguments to make an mpf mixed-precision float instance
type MpfAble = float|int|str|mpf


class EllipticToroid:
    '''A class representing a z-axis oriented, origin-centered elliptical toroid in terms of: 
        tor_rad, the major radius (along xy plane)
        hor_rad, the radius of the rotated ellipse which is parallel to the major radius (xy plane)
        ver_rad, the ellipse radius orthogonal to the major radius (z axis).
        
    Class objects also contain three methods for the three surface functions: distance from point
    along ray, surface sense at point, and surface normal at point.

    Attributes 
    ----------
    tor_rad : MpfAble
        The radius from toroid origin to the center of the revolved ellipse, along the xy plane
    hor_rad : MpfAble
        The horizontal radius of the revolved ellipse, along the xy plane
    ver_rad : MpfAble
        The vertical radius of the revolved ellipse, aligned with the z-axis

    Note
    ----
    The class also contains a private, secondary representation of its properties in terms of 3 parameters
    p, a0, and b0, such that (x^2+y^2 + pz^2 + b0^2) - a0(x^2+y^2), for the purpose of solving 
    ray intersection. These parameters are currently only used internally.
    
    '''

    __slots__ = ("_tor_rad", "_hor_rad", "_ver_rad", "_p", "_a0", "_b0")

    def __init__(self, tor_rad: MpfAble, hor_rad: MpfAble, ver_rad: MpfAble):
        '''
        Parameters
        ----------
        tor_rad : MpfAble
            The radius from toroid origin to the center of the revolved ellipse, along the xy plane
        hor_rad : MpfAble
            The horizontal radius of the revolved ellipse, along the xy plane
        ver_rad : MpfAble
            The ellipse radius orthogonal to the major radius (z axis).

        Note
        ----
        When inputting non-binary fractions/decimals, either mpf or str instances may be preferred
        to maintain higher precision
        '''
        tor_rad = mpf(tor_rad)
        hor_rad = mpf(hor_rad)
        ver_rad = mpf(ver_rad)

        if not tor_rad > 0: 
            raise ValueError(f"Toroid radius must be greater than 0 (tor_rad={tor_rad})")
        if not hor_rad > 0: 
            raise ValueError(f"Ellipse radii must be greater than 0 (hor_rad={hor_rad})")
        if not ver_rad > 0: 
            raise ValueError(f"Ellipse radii must be greater than 0 (ver_rad={ver_rad})")
        if not tor_rad > hor_rad: 
            raise ValueError(f"Degenerate toroids not supported (tor_rad={tor_rad} < {hor_rad}=hor_rad))")
        self._tor_rad = tor_rad
        self._hor_rad = hor_rad
        self._ver_rad = ver_rad

        # From Graphics Gems, form which is more convenient for solving ray intersection
        self._p = (hor_rad*hor_rad) / (ver_rad*ver_rad)
        self._a0 = 4*tor_rad*tor_rad
        self._b0 = tor_rad*tor_rad  - hor_rad*hor_rad

    @property
    def tor_rad(self) -> mpf:
        return self._tor_rad

    @property
    def hor_rad(self) -> mpf:
        return self._hor_rad

    @property
    def ver_rad(self) -> mpf:
        return self._ver_rad

    def ray_intersection_distances(
        self,
        ray_pos: Iterable[MpfAble],
        ray_dir: Iterable[MpfAble],
        solve_quartic: Callable[[list[mpf]], Iterable[mpf]],
    ) -> list[mpf]:
        '''Solves for intersection distances (aka t-values, where 'end = pos + t*dir') using 
        the given quartic solver, and returns them in a list.

        Returns
        -------
        A list of distances along the given ray to its intersections, if any, with this toroid. 
        This list is not guaranteed to be sorted.

        Parameters
        ----------
        ray_pos : Iterable[MpfAble] (length 3)
            An array corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[MpfAble] (length 3)
            An array corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        solve_quartic : Callable[[list[mpf]], Iterable[mpf]]
            A method which takes a quartic polynomial as a list of mpf instances (ordered c4, c3 ... c0 
            as returned by ray_intersection_polynomial()), and returns its real roots.
        '''
        if all(comp == 0 for comp in ray_dir): raise ValueError("Ray direction cannot be 0")
        if not math.isclose(hypot2(ray_dir), 1): 
            raise ValueError(f"ray_dir must have magnitude 1 (Current mag: {mpmath.sqrt(hypot2(ray_dir))})")
        
        poly = self._ray_intersection_polynomial(ray_pos, ray_dir)
        t_vals = solve_quartic(to_mpfs(poly))
        return [t for t in t_vals if t > 0]

    def ray_intersection_points( 
        self,
        ray_pos: Iterable[mpf],
        ray_dir: Iterable[mpf],
        solve_quartic: Callable[[list[mpf]], Iterable[mpf]],
    ) -> list[list[mpf]]:
        '''Solves for intersection points using the given quartic solver with ray_intersections(), 
        and returns them in a list, sorted by increasing distance.

        Returns
        -------
        A list, sorted in increasing distance, of the given ray's intersection points, if any, 
        with this toroid.

        Parameters
        ----------
        ray_pos : Iterable[mpf] (length 3)
            An array corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[mpf] (length 3)
            An array corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        solve_quartic : Callable[[list[mpf]], Iterable[mpf]]
            A method which takes a quartic polynomial as a list of mpfs (ordered c4, c3 ... c0 
            as returned by ray_intersection_polynomial()), and returns its real roots.
        '''
        if all(comp == 0 for comp in ray_dir): raise ValueError("Ray direction cannot be 0")
        if not math.isclose(hypot2(ray_dir), 1): 
            raise ValueError(f"ray_dir must have magnitude 1 (Current mag: {mpmath.sqrt(hypot2(ray_dir))})")
        
        t_vals = self.ray_intersection_distances(ray_pos, ray_dir, solve_quartic)
        return [add(ray_pos, scl(ray_dir, t)) for t in sorted(t_vals)]

    def distance_to_boundary( 
        self,
        ray_pos: Iterable[mpf],
        ray_dir: Iterable[mpf],
        solve_quartic: Callable[[list[mpf]], Iterable[mpf]],
    ) -> float | None:
        '''Solves for the distance to the first intersection of the given ray with this torus.
        If no intersection is found, returns None.
        
        Returns
        -------
        A float value representing the distance to the first intersesction of the given ray with 
        this torus. Only 'forward' intersections, where distance > 0, will be included.
        If no such intersections are found, returns None instead.

        Parameters
        ----------
        ray_pos : Iterable[mpf] (length 3)
            An array corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[mpf] (length 3)
            An array corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        solve_quartic : Callable[[list[mpf]], Iterable[mpf]]
            A method which takes a quartic polynomial as a list of mpfs (ordered c4, c3 ... c0 
            as returned by ray_intersection_polynomial()), and returns its real roots.
        
        '''
        if all(comp == 0 for comp in ray_dir): raise ValueError("Ray direction cannot be 0")
        if not math.isclose(hypot2(ray_dir), 1): 
            raise ValueError(f"ray_dir must have magnitude 1 (Current mag: {mpmath.sqrt(hypot2(ray_dir))})")
        
        distances = self.ray_intersection_distances(ray_pos, ray_dir, solve_quartic)
        if not distances: 
            return None
        return min(distances)

    def surface_normal(self, pos: Iterable[MpfAble]) -> list[mpf]:
        '''Solves for the vector normal to the torus surface at the given x, y, and z, and
        returns an np array containing that vector.

        Returns
        -------
        A list of mpfs (length 3) representing the surface normal vector.

        Parameters
        ----------
        pos : Iterable[MpfAble], length 3
            The x, y, and z of the position to find a surface vector at. 
            Must be on the surface (point_sense(pos) == 0).
        '''
        if not self.point_sense(pos) == 0: raise ValueError("Point must be on surface.")
    
        [x, y, z] = to_mpfs(pos)

        r = self.tor_rad
        a = self.hor_rad
        b = self.ver_rad

        d = mpmath.sqrt(sq(x) + sq(y))
        f = 2 * (d-r) / (d*sq(a))
        n = [x*f, y*f, (2*z) / sq(b)]
        length = mpmath.sqrt(hypot2(n))
        if length == 0:
            return None
        return [comp/length for comp in n]

    def point_sense(self, pos: Iterable[MpfAble]) -> int:
        '''Evaluates if the given point is inside, outside, or on the toroid surface.
        
        Returns
        -------
        -1 if the point is inside the surface, 0 if the point is exactly on the surface, 
        and 1 if the point is outside of the surface.

        Parameters
        pos : Iterable[MpfAble], length 3
            The x, y, and z of the position to evaluate.
        '''
        [x, y, z] = to_mpfs(pos)

        r = self.tor_rad
        a = self.hor_rad
        b = self.ver_rad

        val = (sq(sq(x) + sq(y) + sq(z*a/b) + (sq(r) - sq(a)))
               - (4*sq(r)) * (sq(x) + sq(y)))
        threshold = 1e-9
        if math.isclose(val, 0, abs_tol=threshold): return 0
        elif val > 0: return 1
        else: return -1

    def _ray_intersection_polynomial(
        self, ray_pos: Iterable[MpfAble], ray_dir: Iterable[MpfAble]
    ) -> list[mpf]:
        '''Finds the coefficients of a polynomial representing the given ray's intersection 
        with this torus. The real roots, if any, of this polynomial represent distances along
        the ray to each intersection. The ray direction is assumed to be normalized.

        Parameters
        ----------
        ray_pos : Iterable[MpfAble] (length 3)
            An array corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[MpfAble] (length 3)
            An array corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.

        Returns
        -------
        A list of mpfs corresponding to the quartic polynomial whose roots are the distances 
        along the given ray to its intersections with the torus. Returned in order 
        [c4, c3, c2, c1, c0] where the polynomial would be written c4x^4, c3x^3, ..., c0. The first
        coefficient is always 1.
        '''
        assert math.isclose(hypot2(ray_dir), mpf(1))

        [x0, y0, z0] = to_mpfs(ray_pos)

        [ax, ay, az] = to_mpfs(ray_dir)

        # Intermediate terms, from Graphics Gems
        f = 1 - sq(az)
        g = f + self._p*sq(az)
        l = 2 * (x0*ax + y0*ay)
        t = sq(x0) + sq(y0)
        q = self._a0 / sq(g)
        m = (l + 2*self._p*z0*az) / g
        u = (t + self._p*sq(z0) + self._b0) / g
        

        # Final polynomial coeffs
        c4 = mpf(1)
        c3 = 2*m
        c2 = sq(m) + 2*u - q*f
        c1 = 2*m*u - q*l
        c0 = sq(u) - q*t
        return [c4, c3, c2, c1, c0]


# misc util functions

def sq(val: mpf|float) -> mpf|float:
    '''
    Returns
    -------
    The square of a float val, i.e. val*val
    
    Parameters
    ----------
    val : mpf|float
        The value to square
    '''
    return val*val


def hypot2(vals: Iterable[mpf|float]) -> mpf|float:
    '''
    Returns
    -------
    The sum of the squares of every value in the given Iterable

    Parameters
    ----------
    vals : Iterable[mpf|float]
        The vector/list of components to find the squared hypotenuse of
    '''
    return sum([sq(val) for val in vals])


def to_mpfs(vals: Iterable[MpfAble]) -> list[mpf]:
    '''
    Returns
    -------
    A list of mpf instances corresponding to each original value
    
    Parameters
    vals : Iterable[MpfAble] 
        A list of mpf-compatible values to convert to mpf instances
    '''
    return [mpf(val) for val in vals]


def scl(vals: Iterable[mpf|float], factor: mpf|float) -> list[mpf]:
    '''
    Returns
    -------
    A list of each value scaled by the given factor

    Parameters
    ----------
    vals : Iterable[mpf|float]
        A list of values to scale
    factor : mpf | float
        The factor to scale each value by
    '''
    return [val*factor for val in vals]


def add(a: Iterable[mpf], b: Iterable[mpf]) -> list[mpf]:
    '''
    Returns
    -------
    A list containing the sums of each corresponding pair of values in a and b

    Parameters
    ----------
    a, b : Iterable[mpf] of same length
        The lists of values to add
    '''
    assert len(a) == len(b)
    return [a[i]+b[i] for i in range(len(a))]