# A python file/module for prototyping Celeritas/ORANGE raytracing functions of toroids
# Ray-torus intersection formulae from Graphics Gems II, Gem V.2 by Joseph M Cychosz, Purdue University
from collections.abc import Callable
import numpy as np
from numpy import linalg as la

'''A module for modeling Elliptic Toroid surfaces for raytracing-like applications, specifically Celeritas and ORANGE.

This module contains a class "Toroid" which represents an elliptical Toroid surface, with three functions necessary for raytracing:
 - The first intersection (if any) with the surface along a given ray
 - The sense of a point relative to the torus, whether inside, on, or outside
 - The normal vector to the surface at a given point
'''

# Suggestion: Implement Geant4 polynomial solver, the new solver that openMC uses, and the quartic equation for a simple but robust comparison
# And then also implement the bounding cylinder idea how that changes accuracy/iterations required


class Toroid:# A class representing a toroid using r, a, and b, with additional forms like p, A, B
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
        self.p = (a * a) / (b * b)
        self.A = 4 * r * r
        self.B = r * r - a * a

    # Finds the characteristic polynomial of a rays intersection with the torus,
    # and returns an np array containing that polynomial's coefficients. (First coeff always 1)
    def ray_intersection_polynomial(
        self, ray_src: np.array, ray_dir: np.array, verbose: bool = False
    ):
        '''
        Parameters
        ----------

        Returns
        -------
        A list of floats corresponding to the quartic polynomial whose roots are the distances 
        along the given ray to its intersections with the torus.
        '''
        x0 = ray_src[0]
        y0 = ray_src[1]
        z0 = ray_src[2]

        ax = ray_dir[0]
        ay = ray_dir[1]
        az = ray_dir[2]

        # Intermediate terms, from Graphics Gems
        f = 1 - az * az
        g = f + self.p * az * az
        l = 2 * (x0 * ax + y0 * ay)
        t = x0 * x0 + y0 * y0
        q = self.A / (g * g)
        m = (l + 2 * self.p * z0 * az) / g
        u = (t + self.p * z0 * z0 + self.B) / g
        if verbose:
            print("f: ", f)
            print("g: ", g)
            print("l: ", l)
            print("t: ", t)
            print("q: ", q)
            print("m: ", m)
            print("u: ", u)

        # Final polynomial coeffs
        c4 = 1
        c3 = 2 * m
        c2 = m * m + 2 * u - q * f
        c1 = 2 * m * u - q * l
        c0 = u * u - q * t
        return np.array([c4, c3, c2, c1, c0])

    """
    Solves for the intersection t values using the given quartic solver, and returns in a list alongside the locals from the solver.
    quart_solver: should take a list of floats and return a list of floats, as well as a copy of its local variables (for debugging/profiling)
    """

    def ray_intersections(
        self,
        ray_src: np.array,
        ray_dir: np.array,
        quart_solver: Callable[[list[float]], (list[float], dict)],
        verbose: bool = False,
        return_points: bool = False,
    ) -> (list[float], dict):
        poly = self.ray_intersection_polynomial(ray_src, ray_dir, verbose)
        t_vals, solver_locals = quart_solver(poly)
        intersections: list = sorted([t for t in t_vals if t > 0])
        if return_points:
            intersections = [ray_src + t * ray_dir for t in intersections]
        return intersections, solver_locals

    # Solves for the surface normal at a given position x, y, z
    def surface_normal(self, pos: np.array):
        x = pos[0]
        y = pos[1]
        z = pos[2]

        d = (x * x + y * y) ** 0.5
        f = 2 * (d - self.r) / (d * self.a * self.a)
        n = np.array([x * f, y * f, (2 * z) / (self.b * self.b)])
        length = la.norm(n)
        if length == 0:
            return None
        n /= length
        return n

    # Returns if the given point is contained inside this torus
    def point_in_volume(self, pos: np.array):
        x = pos[0]
        y = pos[1]
        z = pos[2]
        r = self.r
        a = self.a
        b = self.b
        return (x * x + y * y + z * z * (a * a) / (b * b) + (r * r - a * a)) ** 2 - (
            4 * r * r
        ) * (x * x + y * y) <= 0
