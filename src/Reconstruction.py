from Diffraction import *
from mayavi import mlab
from numpy import fft


def xyzgrid(cutoff, step):
    """
    Generates eqiuspaced data points within a cube with side 2*cutoff.
    The cube is centered on the origin, and the sample spacing is given by step.
    Returns three-dimensional matrices as produced by meshgrid.
    """
    # Sample points on positive x-axis
    x = np.arange(0, cutoff, step) + step
    # Add sample points on negative axis and at the origin
    x = np.concatenate((x[::-1], np.zeros(1), -x), axis=0)

    # Use the same points for y and z axis
    y = x
    z = x
    # Construct a three-dimensional grid
    X, Y, Z = np.meshgrid(x, y, z)
    return X, Y, Z


def grid_nearest(grid,value):
    """
    Input: 
    grid: a grid array generated using meshgrid
    value: the value to find the nearest index in grid
    """
    tmp = np.sort(np.unique(np.ravel(grid)))
    return np.abs(tmp-value).argmin()


def slice(H, K, L, F, x, y, z):
    """
    Input:
    H,K,L: 3D meshgrid
    F: 3D scalar field on the lattice H,K,L
    x,y,z: can be either number or list or array, indicate the slices to be ploted. For example:
        x = [0,1] means that plot the planes x=0 and x=1.
    """
    # convert x,y,z to list if they are numbers
    if not isinstance(x, (list, tuple, np.ndarray)):
        x = [x]
    if not isinstance(y, (list, tuple, np.ndarray)):
        y = [y]
    if not isinstance(z, (list, tuple, np.ndarray)):
        z = [z]
    
    # create a scalar field object using F
    s = mlab.pipeline.scalar_field(F)
    
    mlab.figure()
    # calculate the slice_index of (x,y,z) for plot, based on the values of H, K, L
    for i in x:
        sl = grid_nearest(H, i)
        fig_obj = mlab.pipeline.image_plane_widget(s, plane_orientation='x_axes', slice_index=sl)
    for j in y:
        sl = grid_nearest(K, j)
        fig_obj = mlab.pipeline.image_plane_widget(s, plane_orientation='y_axes', slice_index=sl)
    for k in z:
        sl = grid_nearest(L, k)
        fig_obj = mlab.pipeline.image_plane_widget(s, plane_orientation='z_axes', slice_index=sl)
    mlab.outline()
    return fig_obj


def isosurface(F, values):
    """
    Plot the isosurface with values for scalar field F.
    values: numbers or 1D iterable objects (list, array, tuple)
    """
    if not isinstance(values, (list, tuple, np.ndarray)):
        iso_values = [values]
    else:
        iso_values = [i for i in values]
    
    mlab.figure()
    return mlab.contour3d(F, contours=iso_values, transparent=True)


def rect(T):
    """create a centered rectangular pulse of width T"""
    return lambda t: (-T/2 <= t) & (t < T/2)


def wincomp():
    """
    One-dimensional comparison of the two windows and their transforms
    """
    cutoff = 2
    k = np.linspace(-cutoff-1, cutoff+1, (cutoff*2+2)/0.005+1)
    x = k
    sigma = 1
    f, ax = plt.subplots(2)
    r = rect(cutoff*2)
    ax[0].plot(k, r(k), label = 'Rectangular window')
    ax[0].plot(k, np.exp(-k**2/(2*sigma**2)), 'r', label = 'Gaussian, ' + r'$\sigma$ =' + str(sigma) )
    ax[0].set_title('Transform space')
    ax[0].set_ylim([-1, 4])
    ax[0].legend()
    ax[1].plot(x, 2*cutoff*np.sinc(2*cutoff*x), label = 'Sinc function')
    ax[1].plot(x, sigma*np.sqrt(2*np.pi)*np.exp(-2 * np.pi**2 * x**2 * sigma**2), 'r', label = 'Gaussian')
    ax[1].set_title('Real space')
    ax[1].set_ylim([-1, 4])
    ax[1].legend()
    plt.show()

