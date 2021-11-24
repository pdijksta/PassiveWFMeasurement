from collections import OrderedDict
import os
import itertools
import scipy
import numpy as np
from scipy.constants import c

from . import h5_storage

def transferMatrixDrift66(Ld):
    Md1 = [[1, Ld, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 1, Ld, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1]]

    return np.array(Md1, dtype=float)

def transferMatrixQuad66(Lq, kini):
    # Using complex numbers, this method is valid for positive as well as negative k values
    sin, cos, sqrt = scipy.sin, scipy.cos, scipy.sqrt # numpy trigonometric functions do not work

    kinix = kini
    kiniy = -kini
    phi10x = Lq * sqrt(kinix)
    phi10y = Lq * sqrt(kiniy)
    if kinix == 0:
        Mq10 = transferMatrixDrift66(Lq)
    else:
        Mq10 = [[cos(phi10x), sin(phi10x) / sqrt(kinix), 0, 0, 0, 0],
            [-sqrt(kinix) * sin(phi10x), cos(phi10x), 0, 0, 0, 0],
            [0, 0, cos(phi10y), sin(phi10y) / sqrt(kiniy), 0, 0],
            [0, 0, -sin(phi10y) * sqrt(kiniy), cos(phi10y), 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]]

    Mq10 = np.array(Mq10, dtype=complex)
    #assert np.all(np.imag(Mq10) == 0)
    return np.real(Mq10)

class Lattice:
    """
    Reads elegant .mat file that has been converted from sdds to h5 format using the sdds2hdf command.
    The convention is changed from s to t for the 5th coordinate.
    """
    dim_index = OrderedDict((
            ('x',0),
            ('xp', 1),
            ('y', 2),
            ('yp', 3),
            ('t', 4),
            ('delta', 5),
            ))

    def __init__(self, math5_file):
        mat = h5_storage.loadH5Recursive(math5_file)['page1']
        self.columns = mat['columns']
        self.types = np.array([x.decode() for x in self.columns['ElementType']])
        self.names = np.array([x.decode() for x in self.columns['ElementName']])
        self.quad_names = self.names[self.types == 'QUAD'].copy()

    def generate(self, quad_k1l_dict):
        names, types, columns = self.names, self.types, self.columns
        matrix = np.identity(6)
        ele_matrix = np.zeros_like(matrix)
        all_matrices = []
        for n_element, (type_, name) in enumerate(zip(types, names)):
            if type_ == 'QUAD':
                assert columns['R21'][n_element] == 0 # Quadrupoles in simulations must be 0
                length = columns['R12'][n_element]
                if name.endswith('.Q1') or name.endswith('.Q2'):
                    name2 = name[:-3]
                    length2 = length*2
                else:
                    name2 = name
                    length2 = length
                k1 = quad_k1l_dict[name2.replace('-','.')]/length2
                ele_matrix = transferMatrixQuad66(length, k1)
            else:
                for n_col, n_row in itertools.product(list(range(1,7)), repeat=2):
                    # elegant uses s for the 5th coordinate (S5). Here we transform the matrix to time.
                    factor = 1
                    if n_row == 5:
                        factor /= c
                    if n_col == 5:
                        factor *= c
                    ele_matrix[n_row-1,n_col-1] = columns['R%i%i' % (n_row, n_col)][n_element] * factor
            matrix = ele_matrix @ matrix
            all_matrices.append(matrix)

        self.all_matrices = np.array(all_matrices)
        self.element_names = names
        self.matrix_dict = {name: matrix for name, matrix in zip(names, all_matrices)}

    def get_matrix(self, from_, to):
        r1 = self.matrix_dict[from_]
        r_tot = self.matrix_dict[to]
        return r_tot @ np.linalg.inv(r1)

def generated_lattice(h5_file, quad_k1l_dict):
    lat = Lattice(os.path.join(os.path.dirname(__file__), h5_file))
    lat.generate(quad_k1l_dict)
    return lat

def aramis_lattice(quad_k1l_dict):
    return generated_lattice('./elegant/Aramis.mat.h5', quad_k1l_dict)

def athos_lattice(quad_k1l_dict):
    return generated_lattice('./elegant/Athos.mat.h5', quad_k1l_dict)

def get_beamline_lattice(beamline, quad_k1l_dict):
    if beamline == 'Aramis':
        return aramis_lattice(quad_k1l_dict)
    elif beamline == 'Athos':
        return athos_lattice(quad_k1l_dict)
    elif beamline == 'Athos Pre-Undulator':
        raise NotImplementedError

