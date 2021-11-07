import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.myplotstyle as ms
import PassiveWFMeasurement.plot_results as plot_results

ms.closeall()

beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
file_ = './data/2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
dict_ = h5_storage.loadH5Recursive(file_)['raw_data']
meta_data = dict_['meta_data_begin']
forward_options = config.get_default_forward_options()
backward_options = config.get_default_backward_options()
reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
beam_options = config.get_default_beam_spec()
beam_optics = config.default_optics[beamline]
structure_calib_options = config.get_default_structure_calibrator_options()

delta_gap = -70e-6
structure_position0 = 360e-6
screen_center = -600e-6
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_position0)

tracker = tracking.Tracker(beamline, screen_name, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_options, beam_optics, force_charge=180e-12)

calibrator = calibration.StructureCalibrator(tracker, structure_calib_options, dict_)
calibrator.fit()
plot_results.plot_structure_position0_fit(calibrator.fit_dicts)
new_calib = calibrator.fit_dicts['centroid']['calibration']
tracker.calib = new_calib

meas_screens = calibrator.get_meas_screens()
meas_screens.plot()

gap_recon_dict = calibrator.reconstruct_gap()
plot_results.plot_gap_reconstruction(gap_recon_dict)

max_distance = 350e-6
n_positions = calibrator.get_n_positions(gap_recon_dict['gap'], max_distance)

gap_recon_dict2 = calibrator.reconstruct_gap(use_n_positions=n_positions)
plot_results.plot_gap_reconstruction(gap_recon_dict2)

gauss_dicts = gap_recon_dict2['final_gauss_dicts']

plot_results.plot_reconstruction(gauss_dicts)

ms.show()

