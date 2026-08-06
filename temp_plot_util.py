import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

TOR_COLORS=[
    "#FF33002D",
    "#2F00FF2B",
    "#43ff6459"
]
RAY_COLORS=[
    "#D31500",
    "#6B42FF",
    "#009747",
]

# Returns sets of 3d coordinates of the toroid in numpy arrays
def plot_toroid(tor: dict, precision: int = 1000):
    tor_rad, hor_rad, ver_rad = tor["rab"]
    U = np.linspace(0, 2 * np.pi, precision)
    V = np.linspace(0, 2 * np.pi, precision)
    U, V = np.meshgrid(U, V)

    X = (float(tor_rad) + float(hor_rad) * np.cos(V)) * np.cos(U)
    Y = (float(tor_rad) + float(hor_rad) * np.cos(V)) * np.sin(U)
    Z = float(ver_rad) * np.sin(V)
    return X, Y, Z

# Taken from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c to make drawing rays easier
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, xyz, dxyz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    x, y, z = xyz
    dx, dy, dz = dxyz
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def plot_tor_rays(ax, tors:list[dict], rays:list[dict], tors_used:list=None, len:float=10):
    
    for i, tor in enumerate(tors):
        tor_rad, hor_rad, ver_rad = tor["rab"]
        pad = float(1.5*(tor_rad+hor_rad))
        X, Y, Z = plot_toroid(tor)
        ax.axes.set_xlim3d(left=-pad, right=pad)
        ax.axes.set_ylim3d(bottom=-pad, top=pad)
        ax.axes.set_zlim3d(bottom=-pad, top=pad)
        ax.plot_surface(X, Y, Z, antialiased=True, color=TOR_COLORS[i])
    # for tor in tors[1:]:
    #     X, Y, Z = plot_toroid(tor)
    #     ax.plot_surface(X, Y, Z, antialiased=True, color="#43ff6459")
    # ax.plot_surface(X, Y, Z, antialiased=True, color="orange", zorder=0)

    for j, ray in enumerate(rays):
        # s = [float(n) for n in ray[0]]
        # d = [float(n) for n in ray[1]]
        s = ray['pos']
        d = [n*len for n in ray['dir']]
        h = [n*len*0.1 for n in ray['dir']]
        color = 'black' if tors_used is None else TOR_COLORS[tors_used[j]]
        head_color = 'black' if tors_used is None else RAY_COLORS[tors_used[j]]
        ax.arrow3D(s, d, color=color)
        ax.arrow3D(s, h, color=head_color)

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")

def plot_tor_rays_standalone(tor:dict, rays:list[dict], tors_used:list=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_tor_rays(ax, tor, rays, tors_used=tors_used)
    fig.show()
    plt.show()