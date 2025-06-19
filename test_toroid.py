#A script to run test cases of toroid-ray.py

import matplotlib.pyplot as plt
import toroid_ray
from toroid_ray import Toroid
import numpy as np
import numpy.linalg as la
import math

#Returns arrays x y z representing a toroidal surface with the given precision, that matplotlib can then plot
def plot_toroid(toroid:Toroid, precision:int = 1000):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, 2*np.pi, precision)
    U, V = np.meshgrid(U, V)

    X = toroid.pos[0] + (toroid.r + toroid.a*np.cos(V))*np.cos(U)
    Y = toroid.pos[1] + (toroid.r + toroid.a*np.cos(V))*np.sin(U)
    Z = toroid.pos[2] + toroid.b*np.sin(V)
    return X, Y, Z

# Quick shorthand to check if two arrays are equivalent
def check_equal(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert math.isclose(a[i], b[i])

#Test 1: a toroidal surface is graphed as expected
def test_create():
    # tor = Toroid(5, 1, 0.3, pos=np.array([5, 5, 5]))
    tor = Toroid(5, 1, 0.3)
    pos = tor.pos
    pad = 7
    x, y, z = plot_toroid(tor)
    
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.axes.set_xlim3d(left=pos[0]-pad, right=pos[0]+pad) 
    ax.axes.set_ylim3d(bottom=pos[1]-pad, top=pos[1]+pad) 
    ax.axes.set_zlim3d(bottom=pos[2]-pad, top=pos[2]+pad)
    ax.plot_surface(x, y, z, antialiased=True, color='orange')
    plt.show()


#Test 2: compare basic generated polynomial to a known desmos test case (https://www.desmos.com/3d/3fdrpdcjjw)
def test_polynom():
    c = np.array([0.96, -0.25, 1.3])
    tor = Toroid(3.05, 1, 0.5, pos=c)
    s = np.array([1.4, 2.9, 2.6])
    u = np.array([0.63, -0.2, -1.66])
    u /= la.norm(u)

    poly = tor.ray_intersection_polynomial(s, u, verbose=True)
    desmos = np.array([1, -5.60371151562, 21.4844076830, -38.1674209108, 19.9891068153])
    
    check_equal(poly, desmos)
    #assert np.allclose(poly, desmos)
    #assert poly[0] == 1
    #assert poly[1] == -5.60371151562
    #assert poly[2] ==  21.4844076830
    #assert poly[3] == -38.1674209108
    #assert poly[4] ==  19.9891068153

#Test 3: make sure all the polynomial solvers agree with each other to a certain extent
def test_solvers():
    return None

#Ray through the center shouldn't intersect with the torus
def test_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0,0,1])
    u = np.array([0,0,-1])
    assert len(tor.ray_intersections_np(s, u)) == 0

#Ray starting inside and going out away from center should have 1 intersection
def test_inside_out():
    tor = Toroid(5, 1, 1)
    s = np.array([0,5,0])
    u = np.array([0,1,0])
    # assert len(tor.ray_intersections_np(s, u)) == 1
    check_equal(tor.ray_intersections_np(s, u), [1])

#Ray starting inside and going towards center should have 3 intersections
def test_inside_through_center():
    tor = Toroid(5, 1, 1)
    s = np.array([0,5,0])
    u = np.array([0,-1,0])
    # assert len(tor.ray_intersections_np(s, u)) == 3
    check_equal(tor.ray_intersections_np(s, u), [1, 9, 11])

    diag = np.array([0.5**0.5, 0.5**0.5, 0])
    s = 5*diag
    u = -1*diag
    check_equal(tor.ray_intersections_np(s, u), [1, 9, 11])

    diag = np.array([0.5**0.5, 0.5**0.5, 0.01])
    diag /= la.norm(diag)
    s = 5*diag
    u = -1*diag
    assert len(tor.ray_intersections_np(s, u)) == 3

