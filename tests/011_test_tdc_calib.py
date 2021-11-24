import socket
import numpy as np; np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.data_loader as data_loader
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

beamline = 'Aramis'
structure_name = 'SARUN18-UDCP020'
screen_name = 'SARBD02-DSCR050'
screen_center = 898.02e-6
structure_position0 = 364e-6
delta_gap = 0

hostname = socket.gethostname()
if hostname == 'desktop':
    default_dir = '/storage/data_2021-05-18/'
elif hostname == 'pubuntu':
    default_dir = '/home/work/data_2021-05-18/'
elif 'psi' in hostname or 'lc6a' in hostname or 'lc7a' in hostname or True:
    default_dir = '/sf/data/measurements/2021/05/18/'


blmeas_file = default_dir + '119325494_bunch_length_meas.h5'
example_file = default_dir + '2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'

#blmeas_dict = blmeas.load_avg_blmeas(bunch_length_meas_file)[1]

data = h5_storage.loadH5Recursive(example_file)
meta_data = data['meta_data_begin']
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_position0)
tracker = tracking.get_default_tracker(beamline, structure_name, meta_data, calib, screen_name)
blmeas_profile = beam_profile.profile_from_blmeas(blmeas_file, 400e-15, tracker.total_charge, tracker.energy_eV, 0.02)
print(blmeas_profile.time.min(), blmeas_profile.time.max())
#beam_profile.BeamProfile(blmeas_dict['time'], blmeas_dict['current_reduced'], energy_eV=tracker.energy_eV, total_charge=tracker.total_charge)

x_axis, proj = data_loader.screen_data_to_median(data['pyscan_result'])
raw_screen = beam_profile.ScreenDistribution(x_axis-calib.screen_center, proj, subtract_min=True, total_charge=tracker.total_charge)
raw_screen.aggressive_cutoff(0.02)
raw_screen.crop()

tdc_dict = calibration.tdc_calibration(tracker, blmeas_profile, raw_screen, delta_position=150e-6)


ms.figure('Test TDC calibration')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Profile')
sp_ctr += 1

sp_screen = subplot(sp_ctr, title='Screen')
sp_ctr += 1

sp_wake = subplot(sp_ctr, title='Wakes')
sp_ctr += 1

blmeas_profile.plot_standard(sp_profile, label='Measured')
raw_screen.plot_standard(sp_screen, label='Measured')
tdc_dict['forward_screen'].plot_standard(sp_screen, label='Reconstructed')
tdc_dict['backward_dict']['profile'].plot_standard(sp_profile, label='Backward propagated')
tdc_dict['backward_dict']['screen'].plot_standard(sp_screen, label='Backward propagation')
wake_t = tdc_dict['backward_dict']['wake_time']
wake_x = tdc_dict['backward_dict']['wake_x']

sp_wake.plot(wake_x*1e3, wake_t*1e15)

sp_profile.legend()
sp_screen.legend()




ms.show()
