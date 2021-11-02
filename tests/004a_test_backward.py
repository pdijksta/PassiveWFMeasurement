import sys
import os
import numpy as np
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()



beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
file_ = './data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5'
dict_ = h5_storage.loadH5Recursive(file_)
meta_data = dict_['meta_data_begin']
forward_options = config.get_default_forward_options()
backward_options = config.get_default_backward_options()
reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
beam_options = config.get_default_beam_spec()
beam_optics = config.default_optics[beamline]

delta_gap = -70e-6
structure_center = 360e-6
screen_center = -600e-6
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_center)

tracker = tracking.Tracker('Aramis', screen_name, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_options, beam_optics)

sig_t = 20e-15
tt_range = 10*sig_t
tt_points = int(1e4)
total_charge = tracker.total_charge
bp = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, tracker.energy_eV)

image = dict_['pyscan_result']['image'][10].astype(np.float64)
projx = image.sum(axis=0)[::-1]
x_axis = dict_['pyscan_result']['x_axis_m'][::-1]

screen_raw = beam_profile.ScreenDistribution(x_axis-screen_center, projx, total_charge=total_charge)

screen = tracker.prepare_screen(screen_raw)

backward_dict = tracker.backward_propagate(screen, bp, plot_details=True)

bp_new = backward_dict['profile']

ms.figure('Test backward')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

bp.plot_standard(sp_profile)
bp_new.plot_standard(sp_profile)

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1

screen.plot_standard(sp_screen)




ms.show()



