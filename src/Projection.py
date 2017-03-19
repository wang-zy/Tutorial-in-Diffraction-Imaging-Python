import numpy as np
from numpy.matlib import repmat
from numpy import fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import os


class Molecule(object):
    def __init__(self, *fname):
        # Read the pdb file if exist and construct the Molecule object.
        # If file name not exist, initialize the object.
        self.name = []
        self.x = []
        self.y = []
        self.z = []
        self.tempFactor = []
        self.element = []
        self.charge = []
        self.crd = [self.x, self.y, self.z] # coordinates of the atoms in the molecule.
        self.IDP = [1] * len(self.element)
        for f in fname:
            self.readpdb(f)

    def readpdb(self, fname):
        # COLUMNS        DATA TYPE       FIELD         DEFINITION
        # ---------------------------------------------------------------------------------
        #  1 -  6        Record name     "ATOM  "
        #  7 - 11        Integer         serial        Atom serial number.
        # 13 - 16        Atom            name          Atom name.
        # 17             Character       altLoc        Alternate location indicator.
        # 18 - 20        Residue name    resName       Residue name.
        # 22             Character       chainID       Chain identifier.
        # 23 - 26        Integer         resSeq        Residue sequence number.
        # 27             AChar           iCode         Code for insertion of residues.
        # 31 - 38        Real(8.3)       x             Orthogonal coordinates for X in
        #                                              Angstroms.
        # 39 - 46        Real(8.3)       y             Orthogonal coordinates for Y in
        #                                              Angstroms.
        # 47 - 54        Real(8.3)       z             Orthogonal coordinates for Z in
        #                                              Angstroms.
        # 55 - 60        Real(6.2)       occupancy     Occupancy.
        # 61 - 66        Real(6.2)       tempFactor    Temperature factor.
        # 73 - 76        LString(4)      segID         Segment identifier, left-justified.
        # 77 - 78        LString(2)      element       Element symbol*, right-justified.
        # 79 - 80        LString(2)      charge        Charge on the atom.
        #
        # Details for the atom name:
        # 13 - 14    Chemical symbol* - right justified, except for hydrogen atoms
        # 15         Remoteness indicator (alphabetic)
        # 16         Branch designator (numeric)
        #
        # Element and chemical symbol both refer to the corresponding entry in the
        # periodic table.
        fname = os.path.join(os.path.dirname(__file__), '..', 'data', fname)
        f = open(fname)
        content = f.readlines()
        f.close()
        for i, val in enumerate(content):
            if val[:4] == 'ATOM':
                self.name.append(val[11:16].strip())
                self.x.append(float(val[30:38]))
                self.y.append(float(val[38:46]))
                self.z.append(float(val[46:54]))
                self.tempFactor.append(float(val[60:66]))
                self.element.append(val[76:78].strip())
                self.charge.append(val[78:80].strip())
        self.crd = [self.x, self.y, self.z]
        self.IDP = [1] * len(self.element)


def sphere(n):
    u = np.linspace(-np.pi, np.pi, n+1)
    v = np.linspace(0, np.pi, n+1)

    x =  np.outer(np.cos(u), np.sin(v))
    y =  np.outer(np.sin(u), np.sin(v))
    z =  np.outer(np.ones(np.size(u)), np.cos(v))
    
    return x,y,z


def drawmol(mol, elevation, azimuth):
    """
    Colors:
    White: hydrogen
    Black: carbon
    Red: oxygen
    Blue: nitrogen
    Yellow: sulphur
    Grey: other
    """
    if len(mol.element) > 50:
        rad = 10
    else:
        rad = 20
    
    X, Y, Z = sphere(rad)
    ax = plt.figure().gca(projection='3d')
    
    for i in range(len(mol.element)):  
        if mol.element[i] == 'H':
            col = 'w'; r = 0.5    # White
        elif mol.element[i] == 'C':
            col = 'k'; r = 0.85   # Black
        elif mol.element[i] == 'O':
            col = 'r'; r = 0.95   # Red
        elif mol.element[i] == 'N':
            col = 'b'; r = 0.9    # Blue
        elif mol.element[i] == 'S':
            col = 'y'; r = 1.0    # Yellow
        else:
            col = 'g'; r = 0.9    # others
        
        ax.plot_surface(mol.x[i]+r*X, mol.y[i] + r*Y, mol.z[i] + r*Z, rstride=1, cstride=1, linewidth=0, color=col)
    
    # Create cubic bounding box to simulate equal aspect ratio
    x_max = max(mol.x) + 1; x_min = min(mol.x) - 1
    y_max = max(mol.y) + 1; y_min = min(mol.y) - 1
    z_max = max(mol.z) + 1; z_min = min(mol.z) - 1
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # plt.axis('off')
    ax.view_init(elevation, azimuth)
    plt.show()


class ScatterFactor(object):
    def __init__(self, sffile):
        """
        Read the file containing coefficients for analytical approximation
        to scattering factors for different atoms into a object.

        Definition:
        f: scattering factor
        f(\sin(\theta)/\lambda) = \sum_{i=1}^4 a_i \exp(- b_i * (\sin(\theta)/\lambda)^2 ) + c

        Structure of the file:
        Element_name a1 b1 a2 b2 a3 b3 a4 b4 c
        """
        self.label = []
        self.a = []
        self.b = []
        self.c = []

        sffile = os.path.join(os.path.dirname(__file__), '..', 'data', sffile)
        data = np.loadtxt(sffile, delimiter=None, comments='#', dtype="|S15,f8,f8,f8,f8,f8,f8,f8,f8,f8")
        for n in range(len(data)):
            self.label.append(data[n][0])
            self.a.append([data[n][l] for l in [1, 3, 5, 7]])
            self.b.append([data[n][l] for l in [2, 4, 6, 8]])
            self.c.append(data[n][9])


def scattering_factor(elements, HKL, sf):
    """
    Input:
    elements: all the elements in a molecule -> mol.element (1*n) list
    HKL: (3*N) array contains all points that we are calculating in 3D Fourier space
    sf: object of class ScatterFactor, which contains information of sf for each atom

    Output:
    (n*N) array
    """
    atomTypes = np.unique(elements)

    f = np.zeros((len(elements), HKL.shape[1]))

    stols = np.array([0.25 * np.sum(np.square(HKL), axis=0)])

    for iType in range(atomTypes.shape[0]):

        try:
            sfnr = sf.label.index(atomTypes[iType])
        except ValueError:
            print "Cannot find " + atomTypes[iType] + " in atomsf.lib"

        idx = [i for i, x in enumerate(elements) if x == atomTypes[iType]]

        a = np.array(sf.a[sfnr])  # 1*4, based on the structure of the atomsf.lib file
        b = np.array(sf.b[sfnr])
        b.shape = len(b), 1  # 4*1
        c = sf.c[sfnr]

        fType = c + np.dot(a, np.exp(-b * stols))  # b * stols -> 4*N, fType-> 1*N

        f[idx, :] = repmat(fType, len(idx), 1)

    smallerthanzero = np.where(f < 0)
    f[smallerthanzero] = np.finfo(float).eps

    return f


def debye_waller_factor(IDP, HKL):
    """
    Input:
    IDP: 1*n list
    HKL: 3*N array

    Output:
    (n*N) array
    """
    mol_IDP = np.array(IDP)  # convert list to array
    mol_IDP.shape = len(IDP), 1  # convert dimension to n*1
    stols = np.array([0.25 * np.sum(np.square(HKL), axis=0)])  # 1*N
    T = np.exp(-mol_IDP * stols)
    return T


def structure_factor(f, R, HKL):
    """
    Input:
    f: pre-calculated factor including scatting factor as well as the debye_waller_factor. -> (n*N)
    R: coordinates of elements in a molecule. -> 3*n list
    HKL: 3*N array

    Output:
    1*N array
    """
    phase = -2*np.pi * np.dot(np.transpose(HKL), np.array(R))  # N*n
    F = np.sum(np.multiply(np.transpose(np.exp(1j*phase)), f), axis=0)
    return F


def moltrans(mol, H, K, L):
    """
    Input:
    mol: Molecule class object
    H,K,L: meshgrid contains all the points in Fourier space to be calculated (must be matrix of the same size)

    Output:
    Molecule transform in Fourier space, same size as H, K or L
    """
    sizeH = H.shape
    nrCrds = np.prod(sizeH)

    H = np.reshape(H, nrCrds)
    K = np.reshape(K, nrCrds)
    L = np.reshape(L, nrCrds)

    HKL = np.array([H, K, L])

    sf = ScatterFactor('atomsf.lib')
    f = scattering_factor(mol.element, HKL, sf)
    f = np.multiply(f, debye_waller_factor(mol.IDP, HKL))
    F = structure_factor(f, mol.crd, HKL)

    return np.reshape(F.conj(), sizeH)


def TwoD_grid(step, cutoff):
    # Sample points on positive h-axis
    h = np.arange(0, cutoff, step) + step
    # Add sample points on negative axis and at the origin
    h = np.concatenate((h[::-1], np.zeros(1), -h),axis=0)

    # Use the same points for the k-axis
    k = h

    # Construct a two-dimensional grid
    [H,K] = np.meshgrid(h,k)
    L = np.zeros_like(K)
    return H, K, L


def rotmatx(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), np.sin(theta)],
                  [0, -np.sin(theta), np.cos(theta)]])
    return R


def rotmaty(theta):
    R = np.array([[np.cos(theta), 0, -np.sin(theta)],
                  [0, 1, 0],
                  [np.sin(theta), 0, np.cos(theta)]])
    return R


def rotmatz(theta):
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


def rotationmatrix(phi, n): 
    """
    n is vector.
    Return R is a 3D rotation matrix
    """
    if not isinstance(n, (list, tuple, np.ndarray)):
        raise ValueError('n must be a vector!')
    n = np.asarray(n, dtype=float)
    n = n / np.sqrt(np.sum(n**2))
    # Calculate the assymetric matrix (for cross products).
    N = np.array([[0, -n[2], n[1]], [n[2], 0 , -n[0]], [-n[1], n[0], 0]])
    # Calculate rotation matrix
    R = expm(N * phi)
    
    return R


def squarewin2(innersz, outersz):
    """
    Creat a 2-D grid W with the siez of outersz and the values of W are all 0
    except in its center range with the size of innersz, where the values are 1.

    Input:
    innersz: size of the inner range with value 1. -> tuple, list, or array
    outersz: size of the outer range with value 0. -> tuple, list, or array
    Output: W
    """
    innersz = np.asarray(innersz)
    outersz = np.asarray(outersz)
    ic = np.ceil((innersz+1)/2.).astype(int)
    oc = np.ceil((outersz+1) / 2.).astype(int)

    W = np.zeros(outersz)
    x1 = oc[0] - ic[0]
    x2 = x1 + innersz[0]
    y1 = oc[1] - ic[1]
    y2 = y1 + innersz[1]
    W[x1:x2, y1:y2] = np.ones(innersz)
    return W


def getviolations(g, support):
    """
    Return index for violations of the support.
    """
    return np.where(g*support <=0)

