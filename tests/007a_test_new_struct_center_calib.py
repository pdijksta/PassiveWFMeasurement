from scipy.interpolate import interp1d
import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

gap_recon_dict = h5_storage.loadH5Recursive('./gap_recon_dict.h5')

ms.figure('Gap reconstruction improvements')
subplot = ms.subplot_factory(3,3)
sp_ctr = 1

sp_raw = subplot(sp_ctr, title='Raw data', xlabel='$\Delta$ g ($\mu$ m)', ylabel='rms duration (fs)')
sp_ctr += 1

sp_fit = subplot(sp_ctr, title='Fit 0', xlabel='distances ($\mu$m)', ylabel='rms duration (fs')
sp_ctr += 1

sp_heat = subplot(sp_ctr, title='Fit', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1


distance_rms_arr = gap_recon_dict['distance_rms_arr']

distance_rms_dict = {}

for n in range(len(distance_rms_arr)):
    distance_arr = distance_rms_arr[n,:,0]
    rms_arr = distance_rms_arr[n,:,1]
    distance_plot = distance_arr - distance_arr.min()
    sort = np.argsort(distance_plot)
    distance_rms_dict[n] = [distance_arr[sort], rms_arr[sort]]
    label = '%i $\mu$m' % (gap_recon_dict['final_beam_positions'][n]*1e6)
    sp_raw.plot(distance_plot[sort]*1e6, rms_arr[sort]*1e15, label=label, marker='.')



beam_positions = gap_recon_dict['final_beam_positions']
distances = gap_recon_dict['gap']/2. - np.abs(beam_positions)

mask_pos = beam_positions > 0
mask_neg = beam_positions < 0

index0 = gap_recon_dict['best_index']
rms_arr = gap_recon_dict['all_rms'][index0]
sp_fit.plot(distances[mask_pos]*1e6, rms_arr[mask_pos]*1e15, label='Pos')
sp_fit.plot(distances[mask_neg]*1e6, rms_arr[mask_neg]*1e15, label='Neg')


def get_fit_param(delta_streaker0, delta_gap):
    distances = gap_recon_dict['gap']/2. - np.abs(beam_positions)
    new_distances = np.zeros_like(distances)
    new_distances[mask_pos] = distances[mask_pos] - delta_streaker0 + delta_gap/2.
    new_distances[mask_neg] = distances[mask_neg] + delta_streaker0 + delta_gap/2.
    new_rms_list = []
    for n, new_distance in enumerate(new_distances):
        distance_arr = distance_rms_dict[n][0]
        rms_arr = distance_rms_dict[n][1]
        if distance_arr[1] < distance_arr[0]:
            distance_arr = distance_arr[::-1]
            rms_arr = rms_arr[::-1]

        #new_rms = np.interp(new_distance, distance_arr, rms_arr, left=np.nan, right=np.nan)
        new_rms = interp1d(distance_arr, rms_arr, fill_value='extrapolate')(new_distance)
        new_rms_list.append(new_rms)
    new_rms_arr = np.array(new_rms_list)
    #not_nan = ~np.isnan(new_rms_arr)
    #if np.sum(not_nan) > 4:
    #    fit = np.poly1d(np.polyfit(new_distances[not_nan], new_rms_arr[not_nan], 1))
    #else:
    #    fit = [np.nan, np.nan]
    #    print('.')
    fit = np.poly1d(np.polyfit(new_distances, new_rms_arr, 1))
    outp = {
            'fit': fit,
            'new_distances': new_distances,
            'new_rms': new_rms_arr,
            }
    #import pdb; pdb.set_trace()
    return outp

delta_streaker0_range = np.linspace(0e-6, 5e-6, 21)
delta_gap_range = np.linspace(-25e-6, 25e-6, 51)

fit_arr = np.zeros([len(delta_streaker0_range), len(delta_gap_range)])
mean_rms_arr = fit_arr.copy()
mean_rms_pos = fit_arr.copy()
mean_rms_neg = fit_arr.copy()

for n1, delta_streaker0 in enumerate(delta_streaker0_range):
    for n2, delta_gap in enumerate(delta_gap_range):
        fit_dict = get_fit_param(delta_streaker0, delta_gap)
        #fit_arr[n1, n2] = fit_dict['fit'][1] / np.mean(fit_dict['new_rms'])
        fit_arr[n1, n2] = fit_dict['fit'][1]
        mean_rms_arr[n1, n2] = np.mean(fit_dict['new_rms'])
        mean_rms_pos[n1, n2] = np.mean(fit_dict['new_rms'][mask_pos])
        mean_rms_neg[n1, n2] = np.mean(fit_dict['new_rms'][mask_neg])

#fit_arr2 = fit_arr**2
fit_arr2 = np.abs(fit_arr)
#fit_arr2 /= np.mean(fit_arr2)

diff_sides = np.abs(mean_rms_pos - mean_rms_neg)



x_axis = delta_gap_range
y_axis = delta_streaker0_range
x_factor = y_factor = 1e6
extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[-1]*y_factor, y_axis[0]*y_factor]
#extent = None
plot = sp_heat.imshow(fit_arr2, extent=extent, aspect='auto')
ms.plt.colorbar(plot)

sp_heat_rms = subplot(sp_ctr, title='Rms duration', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1


plot = sp_heat_rms.imshow(mean_rms_arr*1e15, cmap='hot', extent=extent, aspect='auto')
ms.plt.colorbar(plot)


sp_heat_diff = subplot(sp_ctr, title='Rms duration difference', xlabel='$\Delta$ g ($\mu$m)', ylabel='$\Delta$ center ($\mu$m)', grid=False)
sp_ctr += 1

plot = sp_heat_diff.imshow(diff_sides*1e15, cmap='hot', extent=extent, aspect='auto')
ms.plt.colorbar(plot)



argmin = np.argwhere(fit_arr2 == np.nanmin(fit_arr2)).squeeze()
fit_dict = get_fit_param(delta_streaker0_range[argmin[0]], delta_gap_range[argmin[1]])
new_distances = fit_dict['new_distances']
new_rms = fit_dict['new_rms']
sp_fit.plot(new_distances[mask_pos]*1e6, new_rms[mask_pos]*1e15, label='Pos new')
sp_fit.plot(new_distances[mask_neg]*1e6, new_rms[mask_neg]*1e15, label='Neg new')




print('Delta gap %i um' % round(delta_gap_range[argmin[1]]*1e6))
print('Delta struct center %i um' % round(delta_streaker0_range[argmin[0]]*1e6))


#delta_streaker0 = delta_streaker0_range[argmin[0]]
#delta_gap = delta_gap_range[argmin[1]]
#fit_dict = get_fit_param(delta_streaker0, delta_gap)
#new_distances = fit_dict['new_distances']
#new_rms = fit_dict['new_rms']

#color = sp_final.plot(new_distances[mask_pos]*1e6, new_rms[mask_pos]*1e15, label='Best')[0].get_color()
#sp_final.plot(new_distances[mask_neg]*1e6, new_rms[mask_neg]*1e15, color=color, ls='--')

#n12_pairs = [list(argmin)]


sp_final = subplot(sp_ctr, title='New fits', xlabel='distances ($\mu$m)', ylabel='rms duration (fs')
sp_ctr += 1


n12_pairs = []

for n1 in [0, len(delta_streaker0_range)-1]:
    for n2 in [0, len(delta_gap_range)-1]:
        n12_pairs.append([n1, n2])
n12_pairs.append(argmin)

for n1, n2 in n12_pairs:
    delta_streaker0 = delta_streaker0_range[n1]
    delta_gap = delta_gap_range[n2]
    fit_dict = get_fit_param(delta_streaker0, delta_gap)
    new_distances = fit_dict['new_distances']
    new_rms = fit_dict['new_rms']
    label = '%i / %i / %i' % (round(delta_streaker0*1e6), delta_gap*1e6, new_rms.mean()*1e15)
    color = sp_final.plot(new_distances[mask_pos]*1e6, new_rms[mask_pos]*1e15, label=label, ls='None', marker='.')[0].get_color()
    sp_final.plot(new_distances[mask_neg]*1e6, new_rms[mask_neg]*1e15, color=color, ls='None', marker='o')
    xx_fit = np.array(new_distances)
    yy_fit = fit_dict['fit'](xx_fit)
    sp_final.plot(xx_fit*1e6, yy_fit*1e15, color=color, ls='dotted')





sp_raw.legend()
sp_fit.legend()
sp_final.legend(title='$\Delta p_0$ ($\mu$m) / $\Delta$g ($\mu$m) / rms (fs)')

ms.show()

