import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
import matplotlib.pyplot as plt
from Projection import *


class Arrow3D(FancyArrowPatch):
    """
    Class to draw a 3D arrow.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def xyztohkl(X, Y, Z, Lambda):
    """
    Projects detector points onto an Ewald sphere.
    Lambda is the wavelength and X, Y and Z are the pixelcoordinates.
    The wavelength and coordinates do not need to be in the same units.
    X, Y and Z are all M x N matrices, where M x N is the detector size.
    It is assumed that the detector is perpendicular to the Z-axis (meaning
    that all values in the matrix Z should be equal)
    """
    D = np.sqrt(X**2 + Y**2 + Z**2)

    H = 1./Lambda * (X/D)          # x/d = sin(phiX).
    K = 1./Lambda * (Y/D)          # y/d = sin(phiY).
    L = 1./Lambda * (Z/D - 1)      # z/d = cos(2*theta).

    return np.array([H, K, L])


def experiment():
    """
    Planar surface: detector with diffraction image
    Sphere: map the detector data in Fourier space (also called Ewald sphere)
    Arrow: wave vector of the incident radiation
    """
    # Distance between samples.
    step = 1. / 20
    cutoff = 1.
    # Sample points on positive x-axis
    x = np.arange(0, cutoff, step) + step
    # Add sample points on negative axis and at the origin
    x = np.concatenate((x[::-1], np.zeros(1), -x), axis=0)
    # Use the same points for the y-axis
    y = x
    # Construct a two-dimensional grid
    [X, Y] = np.meshgrid(x, y)

    detectorDistance = 1.
    Z = np.ones(X.shape) * detectorDistance

    [H, K, L] = xyztohkl(X, Y, Z, 1.)

    caf = Molecule('caffeine.pdb')
    F = moltrans(caf, H, K, L)

    int = np.abs(F) ** 2
    Color = np.log(int)  # use log value of the Fourier component as the color of the surface
    Color = Color / Color.max() # normalize

    ax = plt.figure().gca(projection='3d')
    ax.plot_surface(X, Z, Y, rstride=1, cstride=1, linewidth=0, facecolors=cm.jet(Color), antialiased=True)
    ax.plot_surface(H, L, K, rstride=1, cstride=1, linewidth=0, facecolors=cm.jet(Color), antialiased=True)
    arrow = Arrow3D([0, 0], [0, 0.5], [0, 0], mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
    ax.view_init(40, 30)
    ax.add_artist(arrow)
    ax.set_axis_off()
    plt.show()


def spheresec(num=20, rad=1., theta=[0, np.pi], phi=[0, 2 * np.pi]):
    """
    Generates a spherical surface object with radius rad (which defaults to one).
    The object can be a section of a sphere, in which case theta is a vector [THMIN, THMAX]
    specifying the smallest and largest polar angle (colatitude, the angle from the north pole),
    and phi is a vector [PHMIN,PHMAX] specifying the smallest and largest azimuth (longitude).

    Output:
    An array contains three square matrix with dimension (n+1)*(n+1)
    """
    n = num
    r = rad
    theta = np.linspace(theta[0], theta[-1], n + 1)  # Polar angle
    phi = np.linspace(phi[0], phi[-1], n + 1)  # Azimuth

    x = r * np.expand_dims(np.sin(theta), axis=1) * np.expand_dims(np.cos(phi), axis=0)  # (n+1)*(n+1) matrix

    y = r * np.expand_dims(np.sin(theta), axis=1) * np.expand_dims(np.sin(phi), axis=0)

    z = r * np.expand_dims(np.cos(theta), axis=1) * np.expand_dims(np.ones(n + 1), axis=0)

    return x, y, z


def sph2rotmat(zenith, azimuth, inplane):
    # Input should be either all lists or all numbers.
    # Output is an array containing the rotation matrix for each angle combination.

    if type(zenith) is not list:
        R = np.dot(np.dot(rotmatz(-azimuth), rotmaty(-zenith)), rotmatz(azimuth - inplane))
    else:
        R = np.zeros((3, 3, len(zenith) * len(azimuth) * len(inplane)))
        idx = -1

        for izen in range(len(zenith)):
            for iaz in range(len(azimuth)):
                for iin in range(len(inplane)):
                    idx += 1
                    R[:, :, idx] = np.dot(np.dot(rotmatz(-azimuth[iaz]),
                                                 rotmaty(-zenith[izen])),
                                          rotmatz(azimuth[iaz] - inplane[iin]))
    return R


def ewaldsurf(lam=1., res=None, N=50, theta=0, phi=0, psi=0, plot=False):
    """
    lam: wavelength of the incident light
    res: resolution setting
    phi rotates object clockwise around z-axis
    theta rotates object clockwise around x-axis
    psi rotates object clockwise around z-axis
    (phi is thus used to orient the object around the z-axis before
    the rotation around the z-axis)

    Default value:
    lam=1., res=0.5, N=50, theta=0, phi=0, psi=0
    """
    if res is None:
        res = 0.5 * lam
    if res < 0.5 * lam:
        print "Resolution unachievable: resolution must be larger than lambda/2!"
        return

    thetaMax = np.arcsin(lam / (2. * res))
    pol = [0, 2 * thetaMax]
    az = [0, 2 * np.pi]
    X, Y, Z = spheresec(N, 1. / lam, pol, az)  # X ,Y ,Z are square matrix

    H = np.reshape(X, X.shape[0] * X.shape[1])
    K = np.reshape(Y, Y.shape[0] * Y.shape[1])
    L = np.reshape(Z, Z.shape[0] * Z.shape[1]) - 1. / lam

    R = sph2rotmat(theta, phi, psi)

    rotHKL = np.dot(np.transpose(np.array([H, K, L])), np.transpose(R))

    rotH = np.reshape(rotHKL[:, 0], X.shape)
    rotK = np.reshape(rotHKL[:, 1], X.shape)
    rotL = np.reshape(rotHKL[:, 2], X.shape)

    if plot:
        ax = plt.figure().gca(projection='3d')
        C = np.sqrt(rotH ** 2 + rotK ** 2 + rotL ** 2)
        C -= np.amax(C)
        C = np.abs(C)
        ax.plot_surface(rotH, rotK, rotL, rstride=1, cstride=1, linewidth=0, facecolors=cm.jet(C), antialiased=True)

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([rotH.max() - rotH.min(), rotK.max() - rotK.min(), rotL.max() - rotL.min()]).max() / 2.0
        mid_x = (rotH.max() + rotH.min()) * 0.5
        mid_y = (rotK.max() + rotK.min()) * 0.5
        mid_z = (rotL.max() + rotL.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()
    else:
        return rotH, rotK, rotL


def ewaldmult(lam=1., res=None, N=50, phi=0, theta=0, psi=0):
    """
    Plots multiple Ewald surfaces. Calls ewaldsurf.

    phi, theta, psi: must be all numbers or lists or numpy array
    Output: plot all surfaces specified by the angles.
    """
    if res is None:
        res = 0.5 * lam

    if not isinstance(phi, (list, tuple, np.ndarray)):  # if phi, theta, psi are all numbers
        phi = np.array([phi])
        theta = np.array([theta])
        psi = np.array([psi])

    ax = plt.figure().gca(projection='3d')

    for i in range(theta.shape[0]):
        X, Y, Z = ewaldsurf(lam, res, N, phi[i], theta[i], psi[i])
        C = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        C -= np.amax(C)
        C = np.abs(C)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, facecolors=cm.jet(C), antialiased=True)

    plt.show()

