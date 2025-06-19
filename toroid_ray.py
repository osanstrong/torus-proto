# A python file/module for prototyping Celeritas/ORANGE raytracing functions of toroids
# Ray-torus intersection formulae from Graphics Gems II, Gem V.2 by Joseph M Cychosz, Purdue University
import numpy as np


#Utility function to solve a normalized quartic polynomial, returning a list of real roots
def solve_normal_quartic(b:float, c:float, d:float, e:float):
    return None

#Suggestion: Implement Geant4 polynomial solver, the new solver that openMC uses, and the quartic equation for a simple but robust comparison
#And then also implement the bounding cylinder idea how that changes accuracy/iterations required

# A class representing a toroid using r, a, and b, with additional forms like p, A, B
class Toroid:
    def __init__(self, r:float, a:float, b:float, pos:np.array = np.array([0,0,0])):
        self.r = r
        self.a = a 
        self.b = b
        #From Graphics Gems, form which is more convenient for solving ray intersection
        self.p = (a*a)/(b*b)
        self.A = 4*r*r
        self.B = r*r-a*a
        self.pos = pos

    #Finds the characteristic polynomial of a rays intersection with the torus, and returns an np array containing that polynomial's coefficients. (First coeff always 1)
    def ray_intersection_polynomial(self, ray_src:np.array, ray_dir:np.array, verbose:bool=False):
        c = self.pos 
        x0 = ray_src[0] - c[0]
        y0 = ray_src[1] - c[1]
        z0 = ray_src[2] - c[2]

        ax = ray_dir[0]
        ay = ray_dir[1]
        az = ray_dir[2]
        
        #Intermediate terms, from Graphics Gems
        f = 1 - az*az
        g = f + self.p*az*az
        l = 2 * (x0*ax + y0*ay)
        t = x0*x0 + y0*y0
        q = self.A / (g*g)
        m = (l + 2*self.p*z0*az) / g
        u = (t + self.p*z0*z0 + self.B) / g
        if verbose:
            print("f: ", f)
            print("g: ", g)
            print("l: ", l)
            print("t: ", t)
            print("q: ", q)
            print("m: ", m)
            print("u: ", u)

        #Final polynomial coeffs
        c4 = 1
        c3 = 2*m
        c2 = m*m + 2*u - q*f
        c1 = 2*m*u - q*l
        c0 = u*u - q*t
        return np.array([c4, c3, c2, c1, c0])



