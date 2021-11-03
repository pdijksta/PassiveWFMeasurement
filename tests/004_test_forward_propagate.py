import numpy as np
import copy
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
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.blmeas as blmeas
import PassiveWFMeasurement.myplotstyle as ms
import WakefieldAnalysis.tracking as tracking_old
import WakefieldAnalysis.config as config_old
import WakefieldAnalysis.elegant_matrix as elegant_matrix
import WakefieldAnalysis.image_and_profile as iap

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

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
structure_position0 = 360e-6
screen_center = 1044e-6
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_position0)

tracker = tracking.Tracker('Aramis', screen_name, structure_name, meta_data, calib, forward_options, backward_options, None, None, None)

beam_spec = config.get_default_beam_spec()
beam_spec.update(config.default_optics['Aramis'])
n_particles = int(1e5)
sig_t = 20e-15
tt_range = 10*sig_t
tt_points = int(1e4)
total_charge = 200e-12
bp = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, tracker.energy_eV)

beam_obj = beam.beam_from_spec(['x', 't'], beam_spec, n_particles, bp, total_charge, tracker.energy_eV)

beam_obj0 = copy.deepcopy(beam_obj)
beam_obj0.linear_propagate(tracker.lat.get_matrix(tracker.lat.element_names[0].replace('-', '.'), tracker.structure_name.replace('-', '.')))
betax = beam_obj.get_beta_from_beam('x')
alphax = beam_obj.get_alpha_from_beam('x')
print('Betax, alphax:', betax, alphax)

forward_dict = tracker.forward_propagate(beam_obj, plot_details=True, output_details=True)
screen = forward_dict['screen']

ms.figure('Test forward')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

bp.plot_standard(sp_profile)

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1

sp_wake = subplot(sp_ctr, title='Wake', xlabel='t (fs)', ylabel='$\Delta$ x\'')
sp_ctr += 1

wake_time = forward_dict['wake_dict_dipole']['wake_time']
wake_xp = forward_dict['wake_dict_dipole']['wake_potential']/tracker.energy_eV
sp_wake.plot(wake_time*1e15, wake_xp, label='New')

screen.plot_standard(sp_screen, label='New %i %i' % (screen.mean()*1e6, screen.rms()*1e6))

tracker_old = tracking_old.Tracker(**config_old.get_default_tracker_settings())
tracker_old.set_simulator(meta_data)
bp_old = iap.BeamProfile(bp.time, bp.charge_dist, bp.energy_eV, bp.total_charge)
forward_dict_old = tracker_old.matrix_forward(bp_old, [0., tracker.structure_gap], [0., tracker.beam_position])
screen_old = forward_dict_old['screen']

screen_old.plot_standard(sp_screen, label='Old %i %i' % (screen_old.mean()*1e6, screen_old.rms()*1e6))

# Test optics
beta0, alpha0, gamma0 = beam_obj0.get_beta_from_beam('x'), beam_obj0.get_alpha_from_beam('x'), beam_obj0.get_gamma_from_beam('x')
optics0 = np.array([[beta0, -alpha0], [-alpha0, gamma0]])
mat = tracker.matrix[:2,:2]
optics = mat @ optics0 @ mat.T


beam_obj2 = copy.deepcopy(beam_obj0)
beam_obj2.linear_propagate(tracker.matrix)

beta, alpha, gamma = beam_obj.get_beta_from_beam('x'), beam_obj.get_alpha_from_beam('x'), beam_obj.get_gamma_from_beam('x')
# --> Works

beam_start = forward_dict_old['beam_start']
del forward_dict_old['streaker_matrices']['mat_dict']
streaker_matrices2 = {x: y[:4,:4] for x, y in forward_dict_old['streaker_matrices'].items()}
s1 = streaker_matrices2['start_to_s1']
beam_after_s1 = np.matmul(s1, beam_start)
beam_before_s2 = np.matmul(streaker_matrices2['s1_to_s2'], beam_after_s1)
beam_after_s2 = forward_dict_old['beam_after_s2']
beam_at_screen = forward_dict_old['beam_at_screen']

for beam_old, key_new in [
        (beam_start, 'beam_init'),
        (beam_before_s2, 'beam_before_streaker'),
        (beam_after_s2, 'beam_after_streaker'),
        (beam_at_screen, 'beam_at_screen'),
        ]:
    beam_new = forward_dict[key_new]
    print(key_new, beam_old[0,:].std(), beam_new['x'].std())
    print(beam_new.get_beta_from_beam('x'))
# -> Works now

wake_dict_old = forward_dict_old['wake_dict']
wake_time_old = wake_dict_old[1]['wake_t']
wake_xp_old = wake_dict_old[1]['wake']/tracker.energy_eV

sp_wake.plot(wake_time_old*1e15, wake_xp_old, label='Old')

sp_screen.legend()
sp_wake.legend()

# Test quadrupole

blmeas_file = './data/119325494_bunch_length_meas.h5'
blmeas_dict = blmeas.load_avg_blmeas(blmeas_file)

xx = blmeas_dict[1]['time']
yy = blmeas_dict[1]['current']
yy = yy - yy.min()
blmeas_profile = beam_profile.BeamProfile(xx, yy, tracker.energy_eV, tracker.total_charge)

blmeas_beam = beam.beam_from_spec(['x', 't'], beam_spec, n_particles, blmeas_profile, total_charge, tracker.energy_eV)



ms.figure('Test quadrupole')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1

for use_quad, label in zip([False, True], ['Dipole', '+Quadrupole']):
    tracker.forward_options['quad_wake'] = use_quad
    forward_dict = tracker.forward_propagate(blmeas_beam, plot_details=True, output_details=True)
    screen = forward_dict['screen']
    screen.plot_standard(sp_screen, label=label, show_mean=True)

sp_screen.legend()

ms.show()

