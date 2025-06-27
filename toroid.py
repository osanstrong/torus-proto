from collections.abc import Callable, Iterable, Mapping
import math
import numpy as np
from numpy import linalg as la

'''A module for modeling Elliptic Toroid surfaces for raytracing-like applications, specifically Celeritas and ORANGE.

This module contains a class "Toroid" which represents an elliptical Toroid surface, with three functions necessary for raytracing:
 - The first intersection (if any) with the surface along a given ray
 - The sense of a point relative to the torus, whether inside, on, or outside
 - The normal vector to the surface at a given point

Note
----
Ray-torus intersection and normal formulae taken from Graphics Gems II, Gem V.2 by Joseph M Cychosz, Purdue University

'''


class Toroid:
    '''A class representing a z-axis oriented, origin-centered elliptical toroid in terms of: 
        r, the major radius (along xy plane); 
        a, the radius of the rotated ellipse which is parallel to the major radius (xy plane); and 
        b, the ellipse radius orthogonal to the major radius (z axis). 
        
    Class objects also contain three methods for the three surface functions: distance from point
    along ray, surface sense at point, and surface normal at point.

    Note
    ----
    The class also contains a secondary representation of its properties in terms of 3 parameters
    p, A, and B, for the purpose of solving ray intersection. However, accessing these parameters
    directly is highly discouraged, as their precise implementation is non-final.
    
    '''
    def __init__(self, r: float, a: float, b: float):
        '''
        Parameters
        ----------
        r : float
            The radius from the origin of the toroid to the center of the revolved ellipse, or the "Major Radius", along the xy plane
        a : float
            The radius of the revolved ellipse aligned with the major radius, or the "parallel minor radius", along the xy plane
        b : float
            The radius of the revolved ellipse orthogonal to the major radius, or the "orthogonal minor radius", aligned with the z-axis

        Attributes
        ----------
        TODO: Do you define the effective attributes that the user sees here, or include the private facing ones? I assume the former, in which case it should basically be identical
        '''
        self.r = r
        self.a = a
        self.b = b

        # From Graphics Gems, form which is more convenient for solving ray intersection
        self._p = (a*a) / (b*b)
        self._A = 4*r*r
        self._B = r*r  - a*a

    def _ray_intersection_polynomial(
        self, ray_src: Iterable[float], ray_dir: Iterable[float]
    ):
        '''Finds the coefficients of a polynomial representing the given ray's intersection 
        with this torus. The real roots, if any, of this polynomial represent distances along
        the ray to each intersection. The ray direction is assumed to be normalized.

        Parameters
        ----------
        ray_src : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.

        Returns
        -------
        An np array of floats corresponding to the quartic polynomial whose roots are the distances 
        along the given ray to its intersections with the torus. Returned in order 
        [c4, c3, c2, c1, c0] where the polynomial would be written c4x^4, c3x^3, ..., c0. The first
        coefficient is always 1.
        '''
        x0 = ray_src[0]
        y0 = ray_src[1]
        z0 = ray_src[2]

        ax = ray_dir[0]
        ay = ray_dir[1]
        az = ray_dir[2]

        # Intermediate terms, from Graphics Gems
        f = 1 - az*az
        g = f + self._p*az*az
        l = 2 * (x0*ax + y0*ay)
        t = x0*x0 + y0*y0
        q = self._A / (g*g)
        m = (l + 2*self._p*z0*az) / g
        u = (t + self._p*z0*z0 + self._B) / g
        

        # Final polynomial coeffs
        c4 = 1
        c3 = 2*m
        c2 = m*m + 2*u - q*f
        c1 = 2*m*u - q*l
        c0 = u*u - q*t
        return np.array([c4, c3, c2, c1, c0])


    def ray_intersections(
        self,
        ray_src: Iterable[float],
        ray_dir: Iterable[float],
        quart_solver: Callable[[list[float]], (Iterable[float], Mapping)],
    ) -> (list[float], dict):
        '''Solves for intersection t values using the given quartic solver, and returns them in a
        list alongside the locals from the solver.

        Returns
        -------
        A list of distances along the given ray to its intersections, if any, with this toroid. 
        This list is not guaranteed to be sorted.

        Parameters
        ----------
        ray_src : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        quart_solver : Callable[[list[float]], (Iterable[float], dict)]
            A method which takes a quartic polynomial as a list of floats (ordered c4, c3 ... c0 
            as returned by ~ray_intersection_polynomial), and returns its real roots, alongside
            a copy of its local variables
        '''
        poly = self._ray_intersection_polynomial(ray_src, ray_dir)
        t_vals, solver_locals = quart_solver(poly)
        intersections: list = [t for t in t_vals if t > 0]
        return intersections, solver_locals

    def ray_intersection_points( 
        self,
        ray_src: np.array,
        ray_dir: np.array,
        quart_solver: Callable[[list[float]], (list[float], dict)],
    ) -> tuple[list[np.array], Mapping]:
        '''Solves for intersection points using the given quartic solver with ~ray_intersections, 
        and returns them in a list alongside the locals from the solver.

        Returns
        -------
        A list of the given ray's intersection points, if any, with this toroid.
        As with ~ray_intersections, this list is not guaranteed to be sorted.

        Parameters
        ----------
        ray_src : np.array (length 3)
            An array of length 3 corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : np.array (length 3)
            An array of length 3 corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        quart_solver : Callable[[list[float]], (Iterable[float], dict)]
            A method which takes a quartic polynomial as a list of floats (ordered c4, c3 ... c0 
            as returned by ~ray_intersection_polynomial), and returns its real roots, alongside
            a copy of its local variables
        '''
        t_vals, t_locals = self.ray_intersections(ray_src, ray_dir, quart_solver)
        points = [ray_src + t*ray_dir for t in t_vals]
        return points, t_locals

    def distance_to_boundary( 
        self,
        ray_src: Iterable[float],
        ray_dir: Iterable[float],
        quart_solver: Callable[[list[float]], (list[float], dict)],
    ) -> float | None:
        '''Solves for the distance to the first intersection of the given ray with this torus.
        If no intersection is found, returns None.
        
        Returns
        -------
        A float value representing the distance to the first intersesction of the given ray with 
        this torus. An intersection with a distance of 0 or below (backwards) will be excluded.
        If no such intersections are found, returns None instead.

        Parameters
        ray_src : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z coordinates of the ray's origin.
        ray_dir : Iterable[float] (length 3)
            An array of length 3 corresponding to the x, y, and z components of the ray's 
            direction. This vector is assumed to be normalized to a magnitude of 1.
        quart_solver : Callable[[list[float]], (Iterable[float], dict)]
            A method which takes a quartic polynomial as a list of floats (ordered c4, c3 ... c0 
            as returned by ~ray_intersection_polynomial), and returns its real roots, alongside
            a copy of its local variables
        
        '''
        return min(self.ray_intersections(ray_src, ray_dir, quart_solver))

    def surface_normal(self, pos: Iterable[float]) -> np.array:
        '''Solves for the vector normal to the torus surface at the given x, y, and z, and
        returns an np array containing that vector.

        Returns
        -------
        A numpy array (1d, length 3) array of the surface normal vector.

        Parameters
        ----------
        pos : Iterable[float], length 3
            The x, y, and z of the position to find a surface vector at. Presumed to be on the surface.
        '''
        x = pos[0]
        y = pos[1]
        z = pos[2]

        d = (x*x + y*y) ** 0.5
        f = 2 * (d-self.r) / (d*self.a*self.a)
        n = np.array([x*f, y*f, (2*z) / (self.b*self.b)])
        length = la.norm(n)
        if length == 0:
            return None
        n /= length
        return n

    def point_in_volume(self, pos: Iterable[float]) -> int:
        '''Evaluates if the given point is inside, outside, or on the toroid surface.
        
        Returns
        -------
        -1 if the point is inside the surface, 0 if the point is exactly on the surface, 
        and 1 if the point is outside of the surface.

        Parameters
        pos : Iterable[float], length 3
            The x, y, and z of the position to evaluate.
        '''
        x = pos[0]
        y = pos[1]
        z = pos[2]
        r = self.r
        a = self.a
        b = self.b
        val = ((x*x + y*y + z*z*(a*a)/(b*b) + (r*r - a*a))**2
               - (4*r*r) * (x*x + y*y))
        threshold = 1e-9
        if math.isclose(val, 0, abs_tol=threshold): return 0
        elif val > 0: return 1
        else: return -1