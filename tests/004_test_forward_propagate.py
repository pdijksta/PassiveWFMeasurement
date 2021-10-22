import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.beam as beam
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()



beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
file_ = './data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5'
dict_ = h5_storage.loadH5Recursive(file_)
meta_data = dict_['meta_data_begin']
forward_options = config.get_default_forward_options()
backward_options = None

delta_gap = -70e-6
structure_center = 360e-6
screen_center = 1044e-6

tracker = tracking.Tracker('Aramis', screen_name, structure_name, meta_data, delta_gap, structure_center, screen_center, forward_options, backward_options)

beam_spec = config.get_default_beam_spec('Aramis')
n_particles = int(1e5)
sig_t = 20e-15
tt_range = 10*sig_t
tt_points = int(1e4)
total_charge = 200e-12
bp = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, tracker.energy_eV)

beam_obj = beam.beam_from_spec(['x', 't'], beam_spec, n_particles, bp, total_charge, tracker.energy_eV)

forward_dict = tracker.forward_propagate(beam_obj)

screen = forward_dict['screen']

ms.figure('Test forward')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1

screen.plot_standard(sp_screen)




ms.show()



