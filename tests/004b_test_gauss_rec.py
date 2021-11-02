import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
#import PassiveWFMeasurement.beam as beam
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.myplotstyle as ms
import WakefieldAnalysis.tracking as tracking_old
import WakefieldAnalysis.config as config_old
import WakefieldAnalysis.elegant_matrix as elegant_matrix

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

#import WakefieldAnalysis.streaker_calibration as sc

ms.closeall()

plot_details = True


beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
total_charge = 180e-12
file_ = './data/2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
dict_ = h5_storage.loadH5Recursive(file_)['raw_data']

#outp = sc.analyze_streaker_calibration(dict_, do_plot=False)
#screen_center = outp['meta_data']['fit_dict_centroid']['screen_x0']
#structure_center = outp['meta_data']['fit_dict_centroid']['streaker_offset']
screen_center = 898e-6
structure_center = 364e-6
delta_gap = -70e-6

offset_index = 0

meta_data = dict_['meta_data_begin']
structure_position = dict_['streaker_offsets'][offset_index]*1e3
meta_data[structure_name+':CENTER'] = structure_position
forward_options = config.get_default_forward_options()
backward_options = config.get_default_backward_options()
gauss_options = config.get_default_reconstruct_gauss_options()
beam_options = config.get_default_beam_spec()
beam_optics = config.default_optics['Aramis']

calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_center)

tracker = tracking.Tracker('Aramis', screen_name, structure_name, meta_data, calib, forward_options, backward_options, gauss_options, beam_options, beam_optics, force_charge=total_charge)

image = dict_['pyscan_result']['image'][offset_index][10].astype(np.float64)
projx = image.sum(axis=0)[::-1]
x_axis = dict_['pyscan_result']['x_axis_m'][::-1]

tracker_old = tracking_old.Tracker(**config_old.get_default_tracker_settings())
tracker_old.set_simulator(meta_data)

screen_raw = beam_profile.ScreenDistribution(x_axis-screen_center, projx, total_charge=total_charge)
meas_screen = tracker.prepare_screen(screen_raw)

gauss_dict = tracker.reconstruct_profile_Gauss(meas_screen, output_details=True, plot_details=plot_details, centroid_meas=screen_raw.mean())

gauss_kwargs = config_old.get_default_gauss_recon_settings()
gauss_kwargs['meas_screen'] = screen_raw
gauss_kwargs['gaps'] = [0., tracker.structure_gap]
gauss_kwargs['beam_offsets'] = [0., tracker.beam_position]
gauss_kwargs['n_streaker'] = 1
gauss_kwargs['charge'] = tracker.total_charge
gauss_dict_old = tracker_old.find_best_gauss2(**gauss_kwargs, plot_details=plot_details)


ms.figure('Test Gauss')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Transverse charge distribution', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1
screen_raw.plot_standard(sp_screen, label='raw')

meas_screen.plot_standard(sp_screen, label='prepared')

sp_screen.legend()

sp_profile = subplot(sp_ctr, title='Current profiles', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

gauss_dict['reconstructed_profile'].plot_standard(sp_profile, label='New', center='Mean')
gauss_dict_old['reconstructed_profile'].plot_standard(sp_profile, label='Old', center='Mean')
sp_profile.legend()

index = gauss_dict['best_index']
wake_time = gauss_dict['opt_func_wake_time'][index]
wake_x = gauss_dict['opt_func_wake_x'][index]

index_old = gauss_dict_old['best_index']
wake_time_old = gauss_dict_old['opt_func_wakes'][index_old]




sp_wake = subplot(sp_ctr, title='Wake effect', xlabel='t (fs)', ylabel='x (mm)')
sp_ctr += 1

sp_wake.plot(wake_time, wake_x, label='New')


sp_wake.legend()


ms.show()

