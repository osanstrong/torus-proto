# A utility module to aide comparing different aspects of a torus raytracing implementation.
# Specifically, aids comparison of different rootfinding algorithms

import math
import numpy as np



def roots_numpy(coeffs):
    return np.roots(coeffs)

'''
Ferrari Method Implementation
Where coeffs are organized c[0]x^4, c[1]x^3, c[2]x^2, c[3]x, c[4] and c[0]==1
Returns a list of real roots (if no real roots found, empty list)
#TODO: Simplify to only return positive roots?
'''
def real_roots_ferrari(coeffs):
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]
    
    #Variable change t = x-b/4
    au = -(3*b*b)/8 + c
    bu = (b**3)/8 - (b*c)/2 + d
    if bu == 0: # Special case
        print()
    cu = -(3 * b**4)/256 + (b*b*c)/16 - (c*d)
    
    P = -(au*au)/12 - cu
    Q = -(au**3)/108 + (au*cu)/3 - (bu*bu)/8
    W_presqrt = Q*Q/4 + (P**3)/27
    if W_presqrt < 0: return []
    W = np.cbrt(-Q/2 + math.sqrt(W_presqrt))

    y = au/6 + W - P/(3*W)

    #Left and right sides of a term with 2 roots -> 4 solutions
    L2 = 2*y - au
    if L2 < 0: return []
    elif L2 == 0:
        raise ArithmeticError("L==0, Uhhh this means division by 0 and I'm frankly not sure what it means in the context of the intersection problem")
    L = math.sqrt(2*y - au)

    final_roots: list = []
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
    
    return final_roots
    
    #Solve for a root of the subcubic
    # b_c = c
    # c_c = (b*d - 4*e)
    # d_c = 4*c*e - d*d - b*b*e

    # Q = (3*c_c - b_c*b_c) / 9
    # R = (9*b_c*c_c - 27*d_c - 2*b_c**3) / 54
    # # S = math.sqrt(R + math.sqrt(Q**3 + R*R))
    # # T = math.sqrt(R - math.sqrt(Q**3 + R*R))
    # y_1 = math.sqrt(2 * (R + math.sqrt(-Q)))


'''
Returns the vertical phi angle (for any arbitrary theta) of a ray passing through the origin,
 such that it scrapes the inside of the torus on both sides.
   _\      _
 /   \\  /   \    
|     |\|     |
 \   /  \\   /
   ¯      \¯
'''