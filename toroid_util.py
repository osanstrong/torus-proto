# A utility module to aide comparing different aspects of a torus raytracing implementation.
# Specifically, aids comparison of different rootfinding algorithms

import math
import numpy as np



def real_roots_numpy(coeffs, include_base_roots=False):
    all_roots = np.roots(coeffs)
    real = all_roots[np.isreal(all_roots)]
    real_list = [float(np.real(r)) for r in real]
    if include_base_roots:
        return real_list, all_roots
    else:
        return sorted(real_list)

'''
Ferrari Method Implementation
Where coeffs are organized c[0]x^4, c[1]x^3, c[2]x^2, c[3]x, c[4] and c[0]==1
Returns a list of real roots (if no real roots found, empty list), alongside local variables
#TODO: Simplify to only return positive roots?
'''
def real_roots_ferrari(coeffs: list[float]) -> (list[float], dict):
    verbose = True
    final_roots: list = []
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]
    
    #Variable change t = u-b/4
    au = -(3*b*b)/8 + c
    bu = (b**3)/8 - (b*c)/2 + d
    cu = -(3 * b**4)/256 + (b*b*c)/16 - (b*d)/4 + e
    if bu == 0: # Special case: bu = 0, resulting in a biquadratic equation
        return_stage = "biquadratic"
        z_p = (-au + math.sqrt(au*au - 4*cu)) / 2
        z_n = (-au - math.sqrt(au*au - 4*cu)) / 2
        if z_p > 0:
            zz_p = math.sqrt(z_p)
            final_roots.extend([zz_p, -zz_p])
        elif z_p == 0:
            final_roots.append(0)
        if z_n > 0:
            zz_n = math.sqrt(z_n)
            final_roots.extend([zz_n, -zz_n])
        elif z_n == 0:
            final_roots.extend(0)

        return [u - b/4 for u in final_roots], locals()

    
    
    P = -(au*au)/12 - cu
    Q = -(au**3)/108 + (au*cu)/3 - (bu*bu)/8
    W_presqrt = Q*Q/4 + (P**3)/27



    if W_presqrt < 0: 
        return_stage = "W_presqrt"
        return [u - b/4 for u in final_roots], locals()
    W = np.cbrt(-Q/2 + math.sqrt(W_presqrt))

    y = au/6 + W - P/(3*W)

    #Left and right sides of a term with 2 roots -> 4 solutions
    L2 = 2*y - au
    if L2 < 0: 
        return [u - b/4 for u in final_roots], locals()
        return_stage="L is imaginary"
    elif L2 == 0:
        raise ArithmeticError("L==0, Uhhh this means division by 0 and I'm frankly not sure what it means in the context of the intersection problem")
    L = math.sqrt(2*y - au)

    
    #R^2, where L is the negative sqrt
    R2_Ln = -2*y - au + (2*bu)/L
    if R2_Ln > 0:
        R_Ln = math.sqrt(R2_Ln)
        final_roots.append(-L+R_Ln)
        final_roots.append(-L-R_Ln)
    elif R2_Ln == 0:
        final_roots.append(-L)

    #R^2, but where L is the positive sqrt
    R2_Lp = -2*y - au - (2*bu)/L
    if R2_Lp > 0:
        R_Lp = math.sqrt(R2_Lp)
        final_roots.append(L+R_Lp)
        final_roots.append(L-R_Lp)
    elif R2_Lp == 0:
        final_roots.append(L)

    for i in range(len(final_roots)):
        final_roots[i] *= 0.5
        final_roots[i] -= b/4
    
    return [u - b/4 for u in final_roots], locals()
'''
A mod of real_roots_ferrari using numpy to solve the intermediate cubic for debug purposes
'''
def real_roots_ferrari_npdebug(coeffs):
    final_roots: list = []
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]
    
    #Variable change t = x-b/4
    au = -(3*b*b)/8 + c
    bu = (b**3)/8 - (b*c)/2 + d
    cu = -(3 * b**4)/256 + (b*b*c)/16 - (b*d)/4 + e
    if bu == 0: # Special case: biquadratic
        return_stage = "biquadratic"
        z_p = (-au + math.sqrt(au*au - 4*cu)) / 2
        z_n = (-au - math.sqrt(au*au - 4*cu)) / 2
        if z_p > 0:
            zz_p = math.sqrt(z_p)
            final_roots.extend([zz_p, -zz_p])
        elif z_p == 0:
            final_roots.append(0)
        if z_n > 0:
            zz_n = math.sqrt(z_n)
            final_roots.extend([zz_n, -zz_n])
        elif z_n == 0:
            final_roots.extend(0)

        return [u - b/4 for u in final_roots], locals()
    
    
    # P = -(au*au)/12 - cu
    # Q = -(au**3)/108 + (au*cu)/3 - (bu*bu)/8
    # W_presqrt = Q*Q/4 + (P**3)/27



    # if W_presqrt < 0: 
    #     if verbose:
    #         return sorted(final_roots), locals()
    #     else:
    #         return sorted(final_roots)
    #     if (verbose):
    #         return_stage = "W_presqrt"
    # W = np.cbrt(-Q/2 + math.sqrt(W_presqrt))

    # y = au/6 + W - P/(3*W)

    b_c = c
    c_c = (b*d - 4*e)
    d_c = 4*c*e - d*d - b*b*e
    croots = np.roots([1, b_c, c_c, d_c])
    real_croots = croots[np.isreal(croots)]
    y = float(np.real(real_croots[0]))

    #Left and right sides of a term with 2 roots -> 4 solutions
    L2 = 2*y - au
    if L2 < 0: 
        return_stage="L is imaginary"
        return sorted(final_roots), locals()
        
    elif L2 == 0:
        raise ArithmeticError("L==0, Uhhh this means division by 0 and I'm frankly not sure what it means in the context of the intersection problem")
    L = math.sqrt(2*y - au)

    
    #R^2, where L is the negative sqrt
    R2_Ln = -2*y - au + (2*bu)/L
    if R2_Ln > 0:
        R_Ln = math.sqrt(R2_Ln)
        final_roots.append(-L+R_Ln)
        final_roots.append(-L-R_Ln)
    elif R2_Ln == 0:
        final_roots.append(-L)

    #R^2, but where L is the positive sqrt
    R2_Lp = -2*y - au - (2*bu)/L
    if R2_Lp > 0:
        R_Lp = math.sqrt(R2_Lp)
        final_roots.append(L+R_Lp)
        final_roots.append(L-R_Lp)
    elif R2_Lp == 0:
        final_roots.append(L)

    for i in range(len(final_roots)):
        final_roots[i] *= 0.5
        final_roots[i] -= b/4
    
    return sorted(final_roots), locals()
'''
Returns the vertical phi angle (for any arbitrary theta) of a ray passing through the origin,
 such that it scrapes the inside of the torus on both sides.
   _\      _
 /   \\  /   \    
|     |\|     |
 \   /  \\   /
   ¯      \¯
'''