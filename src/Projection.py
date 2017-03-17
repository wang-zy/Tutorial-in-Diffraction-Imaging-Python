import numpy as np
from numpy.matlib import repmat
from numpy import fft
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

