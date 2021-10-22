import copy
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)
import WakefieldAnalysis.elegant_matrix as elegant_matrix
import WakefieldAnalysis.h5_storage as h5_storage
import PassiveWFMeasurement.lattice as lattice
import PassiveWFMeasurement.beam as beam
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()


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

lat_athos = lattice.athos_lattice(meta_data)
mat_python_athos = lat_athos.get_matrix('MIDDLE_STREAKER_%i' % 1, 'SATBD02.DSCR050')


dim1 = ('x', 't')
dim2 = ('x', 'y', 't')



for n_d, dimensions in enumerate([dim1, dim2]):
    spec = config.get_default_beam_spec('Athos')
    n_particles = config.default_n_particles
    sig_t = 20e-15
    tt_range = 200e-15
    tt_points = 1000
    total_charge = 200e-12
    energy_eV = 6e9
    beamProfile = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, energy_eV)

    beam_obj = beam.beam_from_spec(dimensions, spec, n_particles, beamProfile, total_charge, energy_eV)
    beam0 = copy.deepcopy(beam_obj)

    mat6 = lat_athos.get_matrix('MIDDLE_STREAKER_1', 'SATBD02.DSCR050')
    beam_obj.linear_propagate(mat6)

    ms.figure(dimensions)
    subplot = ms.subplot_factory(2,5)
    sp_ctr = 1

    for n_b, b in enumerate([beam0, beam_obj]):
        for dim in beam_obj.dim_index.keys():
            sp = subplot(sp_ctr, title='%i %s' % (n_b, dim), scix=True)
            sp_ctr += 1
            sp.hist(b[dim], bins=100)
            print('Case', n_d, 'Beam', n_b, dim, b[dim].std())







ms.show()




