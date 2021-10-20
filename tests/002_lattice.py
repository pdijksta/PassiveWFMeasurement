import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)
import WakefieldAnalysis.elegant_matrix as elegant_matrix
import WakefieldAnalysis.h5_storage as h5_storage
import PassiveWFMeasurement.lattice as lattice
elegant_matrix.set_tmp_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tmp_elegant'))

file_ = './data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5'
dict_ = h5_storage.loadH5Recursive(file_)
meta_data = dict_['meta_data_begin']



simulator = elegant_matrix.get_simulator(meta_data)
mat_elegant = simulator.get_streaker_matrices(None, 'Aramis')['s2_to_screen']

lat = lattice.aramis_lattice(meta_data)
mat_python = lat.get_matrix('MIDDLE_STREAKER_%i' % 2, 'SARBD02.DSCR050')


file_ = './data/2021_09_23-16_28_16_Screen_data_SATBD02-DSCR050.h5'
dict_ = h5_storage.loadH5Recursive(file_)
meta_data = dict_['meta_data_begin']



simulator = elegant_matrix.get_simulator(meta_data)
mat_elegant_athos = simulator.get_streaker_matrices(None, 'Athos')['s1_to_screen']

lat = lattice.athos_lattice(meta_data)
mat_python_athos = lat.get_matrix('MIDDLE_STREAKER_%i' % 1, 'SATBD02.DSCR050')

