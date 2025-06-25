# A utility module to aide comparing different aspects of a torus raytracing implementation.
# Specifically, aids comparison of different rootfinding algorithms

import math
import numpy as np
import cmath


"""
Solves for roots of the polynomial using numpy's eigenvalue / matrix implementation
"""


def real_roots_numpy(coeffs: list[float]) -> (list[float], dict):
    all_roots = np.roots(coeffs)
    real = all_roots[np.isreal(all_roots)]
    real_list = [float(np.real(r)) for r in real]
    return real_list, locals()


"""
Ferrari Method Implementation
Where coeffs are organized c[0]x^4, c[1]x^3, c[2]x^2, c[3]x, c[4] and c[0]==1
Returns a list of real roots (if no real roots found, empty list), alongside local variables
#TODO: Simplify to only return positive roots?
"""


def real_roots_ferrari(coeffs: list[float]) -> (list[float], dict):
    verbose = True
    final_roots: list = []
    b = float(coeffs[1])
    c = float(coeffs[2])
    d = float(coeffs[3])
    e = float(coeffs[4])

    # Variable change t = u-b/4
    au = -(3 * b * b) / 8 + c
    bu = (b**3) / 8 - (b * c) / 2 + d
    cu = -(3 * b**4) / 256 + (b * b * c) / 16 - (b * d) / 4 + e
    threshold = 1e-12
    if math.isclose(
        bu, 0, abs_tol=threshold
    ):  # Special case: bu = 0, resulting in a biquadratic equation
        return_stage = "biquadratic"
        z_p = (-au + math.sqrt(au * au - 4 * cu)) / 2
        z_n = (-au - math.sqrt(au * au - 4 * cu)) / 2
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

        return [u - b / 4 for u in final_roots], locals()

    P = -(au * au) / 12 - cu
    Q = -(au**3) / 108 + (au * cu) / 3 - (bu * bu) / 8
    W_presqrt = Q * Q / 4 + (P**3) / 27
    R = -Q / 2 + math.sqrt((Q * Q) / 4 + (P**3) / 27)

    # if W_presqrt < 0:
    #     return_stage = "W_presqrt"
    #     return [u - b/4 for u in final_roots], locals()
    # W = np.cbrt(-Q/2 + math.sqrt(W_presqrt))

    # y = au/6 + W - P/(3*W)

    U = np.cbrt(R)
    if U == 0:
        y_r = -np.cbrt(Q)
    else:
        y_r = U - P / (3 * U)
    y = float(-(5 / 6) * au + y_r)

    W = math.sqrt(au + 2 * y)

    # W_r meaning the term on the right of W, and p vs n meaning where W is added or subtracted in the overall equation
    W2_r_p = -1 * (3 * au + 2 * y + 2 * bu / W)
    W2_r_n = -1 * (3 * au + 2 * y - 2 * bu / W)

    if W2_r_p >= 0:
        W_r_p = math.sqrt(W2_r_p)
        final_roots.append(W + W_r_p)
        final_roots.append(W - W_r_p)
    if W2_r_n >= 0:
        W_r_n = math.sqrt(W2_r_n)
        final_roots.append(-W + W_r_n)
        final_roots.append(-W - W_r_n)

    return [0.5 * u - b / 4 for u in final_roots], locals()

    # #Left and right sides of a term with 2 roots -> 4 solutions
    # L2 = 2*y - au
    # if L2 < 0:
    #     return [u - b/4 for u in final_roots], locals()
    #     return_stage="L is imaginary"
    # elif L2 == 0:
    #     raise ArithmeticError("L==0, Uhhh this means division by 0 and I'm frankly not sure what it means in the context of the intersection problem")
    # L = math.sqrt(2*y - au)

    # #R^2, where L is the negative sqrt
    # R2_Ln = -2*y - au + (2*bu)/L
    # if R2_Ln > 0:
    #     R_Ln = math.sqrt(R2_Ln)
    #     final_roots.append(-L+R_Ln)
    #     final_roots.append(-L-R_Ln)
    # elif R2_Ln == 0:
    #     final_roots.append(-L)

    # #R^2, but where L is the positive sqrt
    # R2_Lp = -2*y - au - (2*bu)/L
    # if R2_Lp > 0:
    #     R_Lp = math.sqrt(R2_Lp)
    #     final_roots.append(L+R_Lp)
    #     final_roots.append(L-R_Lp)
    # elif R2_Lp == 0:
    #     final_roots.append(L)

    # return [0.5*u - b/4 for u in final_roots], locals()


J = cmath.exp(2j * cmath.pi / 3)
Jc = 1 / J


def roots2(a, b, c):
    bp = b / 2
    delta = bp * bp - a * c
    u1 = (-bp - delta**0.5) / a
    u2 = -u1 - b / a
    return u1, u2


def cardan(a, b, c, d):
    u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    u = ((-q - r) / 2) ** 0.33333333333333333333333
    v = ((-q + r) / 2) ** 0.33333333333333333333333
    w = u * v
    w0 = abs(w + p / 3)
    w1 = abs(w * J + p / 3)
    w2 = abs(w * Jc + p / 3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    return u + v - z0, u * J + v * Jc - z0, u * Jc + v * J - z0


def ferrari_stackexchange(a, b, c, d, e):
    # Taken from a stackexchange article https://stackoverflow.com/questions/35795663/fastest-way-to-find-the-smallest-positive-real-root-of-quartic-polynomial-4-degr
    # Which cites the French Wikipedia's Ferrari implementation: https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Ferrari
    "resolution of P=ax^4+bx^3+cx^2+dx+e=0"
    "CN all coeffs real."
    "First shift : x= z-b/4/a  =>  P=z^4+pz^2+qz+r"
    z0 = b / 4 / a
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d
    p = -3 * b2 / (8 * a2) + c / a
    q = b * b2 / 8 / a / a2 - 1 / 2 * b * c / a2 + d / a
    r = -3 / 256 * b2 * b2 / a2 / a2 + c * b2 / a2 / a / 16 - b * d / a2 / 4 + e / a
    "Second find X so P2=AX^3+BX^2+C^X+D=0"
    A = 8
    B = -4 * p
    C = -8 * r
    D = 4 * r * p - q * q
    y0, y1, y2 = cardan(A, B, C, D)
    if abs(y1.imag) < abs(y0.imag):
        y0 = y1
    if abs(y2.imag) < abs(y0.imag):
        y0 = y2
    a0 = (-p + 2 * y0.real) ** 0.5
    if a0 == 0:
        b0 = y0**2 - r
    else:
        b0 = -q / 2 / a0
    r0, r1 = roots2(1, a0, y0 + b0)
    r2, r3 = roots2(1, -a0, y0 - b0)
    return [r0 - z0, r1 - z0, r2 - z0, r3 - z0]


def real_roots_ferrari_SE(coeffs: list[float]) -> (list[float], dict):
    a = float(coeffs[0])
    b = float(coeffs[1])
    c = float(coeffs[2])
    d = float(coeffs[3])
    e = float(coeffs[4])

    all_roots = ferrari_stackexchange(a, b, c, d, e)
    imag_threshold = 1e-12
    return [float(r.real) for r in all_roots if abs(r.imag) < imag_threshold], locals()


"""
A mod of real_roots_ferrari using numpy to solve the intermediate cubic for debug purposes
"""


def real_roots_ferrari_npdebug(coeffs):
    final_roots: list = []
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]

    # Variable change t = x-b/4
    au = -(3 * b * b) / 8 + c
    bu = (b**3) / 8 - (b * c) / 2 + d
    cu = -(3 * b**4) / 256 + (b * b * c) / 16 - (b * d) / 4 + e
    if bu == 0:  # Special case: biquadratic
        return_stage = "biquadratic"
        z_p = (-au + math.sqrt(au * au - 4 * cu)) / 2
        z_n = (-au - math.sqrt(au * au - 4 * cu)) / 2
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

        return [u - b / 4 for u in final_roots], locals()

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
    c_c = b * d - 4 * e
    d_c = 4 * c * e - d * d - b * b * e
    croots = np.roots([1, b_c, c_c, d_c])
    real_croots = croots[np.isreal(croots)]
    y = float(np.real(real_croots[0]))

    # Left and right sides of a term with 2 roots -> 4 solutions
    L2 = 2 * y - au
    if L2 < 0:
        return_stage = "L is imaginary"
        return sorted(final_roots), locals()

    elif L2 == 0:
        raise ArithmeticError(
            "L==0, Uhhh this means division by 0 and I'm frankly not sure what it means in the context of the intersection problem"
        )
    L = math.sqrt(2 * y - au)

    # R^2, where L is the negative sqrt
    R2_Ln = -2 * y - au + (2 * bu) / L
    if R2_Ln > 0:
        R_Ln = math.sqrt(R2_Ln)
        final_roots.append(-L + R_Ln)
        final_roots.append(-L - R_Ln)
    elif R2_Ln == 0:
        final_roots.append(-L)

    # R^2, but where L is the positive sqrt
    R2_Lp = -2 * y - au - (2 * bu) / L
    if R2_Lp > 0:
        R_Lp = math.sqrt(R2_Lp)
        final_roots.append(L + R_Lp)
        final_roots.append(L - R_Lp)
    elif R2_Lp == 0:
        final_roots.append(L)

    for i in range(len(final_roots)):
        final_roots[i] *= 0.5
        final_roots[i] -= b / 4

    return sorted(final_roots), locals()


"""
Returns the vertical phi angle (for any arbitrary theta) of a ray passing through the origin,
 such that it scrapes the inside of the torus on both sides.
   _\      _
 /   \\  /   \    
|     |\|     |
 \   /  \\   /
   ¯      \¯
"""
