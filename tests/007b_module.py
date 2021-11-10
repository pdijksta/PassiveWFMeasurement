import numpy as np; np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.myplotstyle as ms
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.plot_results as plot_results

ms.closeall()

beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
file_ = './data/2021_10_24-19_34_36_Calibration_SARUN18-UDCP020.h5'
dict_ = h5_storage.loadH5Recursive(file_)['raw_data']
meta_data = dict_['meta_data_begin']
forward_options = config.get_default_forward_options()
backward_options = config.get_default_backward_options()
reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
beam_options = config.get_default_beam_spec()
beam_optics = config.default_optics[beamline]
structure_calib_options = config.get_default_structure_calibrator_options()

delta_gap = 0
structure_position0 = 360e-6
screen_center = 0
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_position0)

tracker = tracking.Tracker(beamline, screen_name, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_options, beam_optics, force_charge=180e-12)

calibrator = calibration.StructureCalibrator(tracker, structure_calib_options, dict_)
calibrator.fit()
plot_results.plot_structure_position0_fit(calibrator.fit_dicts)

calib_dict = calibrator.calibrate_gap_and_struct_position()
delta_gap_range = calib_dict['delta_gap_range']
delta_streaker0_range = calib_dict['delta_streaker0_range']
fit_coefficients2 = calib_dict['fit_coefficients2']
mean_rms_arr = calib_dict['mean_rms']
diff_sides = calib_dict['diff_sides']
argmin = calib_dict['best_index']
distance_rms_arr = calib_dict['distance_rms_arr']
beam_positions = calib_dict['beam_positions']



#gap_recon_dict = h5_storage.loadH5Recursive('./gap_recon_dict.h5')

fig = ms.figure('Gap reconstruction improvements')
subplot = ms.subplot_factory(3,3)
sp_ctr = 1

sp_raw = subplot(sp_ctr, title='Raw data', xlabel='$\Delta$ g ($\mu$ m)', ylabel='rms duration (fs)')
sp_ctr += 1


for n in range(len(distance_rms_arr)):
    distance_arr = distance_rms_arr[n,:,0]
    rms_arr = distance_rms_arr[n,:,1]
    distance_plot = distance_arr - distance_arr.min()
    sort = np.argsort(distance_plot)
    label = '%i $\mu$m' % (beam_positions[n]*1e6)
    sp_raw.plot(distance_plot[sort]*1e6, rms_arr[sort]*1e15, label=label, marker='.')

sp_heat = subplot(sp_ctr, title='Fit', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1

x_axis = delta_gap_range
y_axis = delta_streaker0_range
x_factor = y_factor = 1e6
extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[-1]*y_factor, y_axis[0]*y_factor]
#extent = None
plot = sp_heat.imshow(fit_coefficients2, extent=extent, aspect='auto')
fig.colorbar(plot, label='Fit coefficint (arb. units)', ax=sp_heat)

sp_heat_rms = subplot(sp_ctr, title='Rms duration', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1


plot = sp_heat_rms.imshow(mean_rms_arr*1e15, cmap='hot', extent=extent, aspect='auto')
ms.plt.colorbar(plot)


sp_heat_diff = subplot(sp_ctr, title='Rms duration difference', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1

plot = sp_heat_diff.imshow(diff_sides*1e15, cmap='hot', extent=extent, aspect='auto')
ms.plt.colorbar(plot)

combined_target = calib_dict['combined_target']
sp_comb = subplot(sp_ctr, title='Target', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1

plot = sp_comb.imshow(np.sqrt(combined_target), cmap='hot', extent=extent, aspect='auto')
ms.plt.colorbar(plot)





print('Delta gap %i um' % round(delta_gap_range[argmin[1]]*1e6))
print('Delta struct center %i um' % round(delta_streaker0_range[argmin[0]]*1e6))


sp_final = subplot(sp_ctr, title='New fits', xlabel='distances ($\mu$m)', ylabel='rms duration (fs')
sp_ctr += 1


n12_pairs = []

for n1 in [0, len(delta_streaker0_range)-1]:
    for n2 in [0, len(delta_gap_range)-1]:
        n12_pairs.append([n1, n2])
n12_pairs.append(argmin)

mask_pos, mask_neg = beam_positions > 0, beam_positions < 0

for n1, n2 in n12_pairs:
    delta_streaker0 = delta_streaker0_range[n1]
    delta_gap = delta_gap_range[n2]
    fit_dict = calib_dict['all_fit_dicts'][n1][n2]
    new_distances = fit_dict['new_distances']
    new_rms = fit_dict['new_rms']
    label = '%i / %i / %i' % (round(delta_streaker0*1e6), delta_gap*1e6, new_rms.mean()*1e15)
    color = sp_final.plot(new_distances[mask_pos]*1e6, new_rms[mask_pos]*1e15, label=label, ls='None', marker='.')[0].get_color()
    sp_final.plot(new_distances[mask_neg]*1e6, new_rms[mask_neg]*1e15, color=color, ls='None', marker='o')
    xx_fit = np.array(new_distances)
    yy_fit = fit_dict['fit'](xx_fit)
    sp_final.plot(xx_fit*1e6, yy_fit*1e15, color=color, ls='dotted')


sp_raw.legend()
sp_final.legend(title='$\Delta p_0$ ($\mu$m) / $\Delta$g ($\mu$m) / rms (fs)')

ms.show()

