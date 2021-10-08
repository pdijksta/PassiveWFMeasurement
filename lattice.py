import itertools
import scipy
import numpy as np

import h5_storage

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
    assert np.all(np.imag(Mq10) == 0)
    return np.real(Mq10)

class Lattice:
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
                length = columns['R12'][n_element]
                k1 = quad_k1l_dict[name]
                ele_matrix = transferMatrixQuad66(length, k1)
            else:
                for n_col, n_row in itertools.product(list(range(1,7)), repeat=2):
                    ele_matrix[n_col-1,n_row-1] = columns['R%i%i' % (n_col, n_row)][n_element]
            matrix = matrix @ ele_matrix
            all_matrices.append(matrix)

        self.all_matrices = np.array(all_matrices)
        self.element_names = names
        self.matrix_dict = {name: matrix for name, matrix in zip(names, all_matrices)}

    def get_matrix(self, from_, to):
        r1 = self.matrix_dict[from_]
        r_tot = self.matrix_dict[to]
        return r_tot @ np.linalg.inv(r1)


#if __name__ == '__main__':
#    test_mat = './elegant/Aramis.mat.h5'
#    lat = Lattice(test_mat)
#    quad_k1l_dict = {x:1 for x in lat.quad_names}
#    lat.generate(quad_k1l_dict)

