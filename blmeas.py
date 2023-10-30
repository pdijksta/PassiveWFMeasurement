import itertools
import os
import copy
import numpy as np

from . import beam_profile
from . import data_loader
from . import h5_storage
from . import myplotstyle as ms

def get_mean_profile(profile_list0, outp='profile', size=5000, cutoff=0.02):
    """
    outp can be profile or index
    """
    profile_list = copy.deepcopy(profile_list0)
    for bp in profile_list:
        bp._yy = bp._yy - bp._yy.min()
        bp.reshape(size)
        bp.aggressive_cutoff(cutoff)
        bp.crop()
        bp.reshape(size)
        bp.center('Mean')

    squares_mat = np.zeros([len(profile_list)]*2, float)

    for n_row in range(len(squares_mat)):
        for n_col in range(n_row):
            bp1 = profile_list[n_row]
            bp2 = profile_list[n_col]
            minus = bp1.charge_dist - bp2.charge_dist
            squares_mat[n_row,n_col] = squares_mat[n_col,n_row] = np.sum(minus**2)

    squares = squares_mat.sum(axis=1)
    n_best = np.argmin(squares)
    if outp == 'profile':
        return profile_list0[n_best]
    elif outp == 'index':
        return n_best

def get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases, cutoff=5e-2, size=int(1e3)):
    time_arr = y_axis / calibration
    current0 = images.sum(axis=-1, dtype=np.float64)
    current = current0.reshape([current0.size//current0.shape[-1], current0.shape[-1]])
    reverse = time_arr[1] < time_arr[0]
    if reverse:
        time_arr = time_arr[::-1]
    current_profiles0 = []

    # Find out where is the head and the tail.
    # In our conventios, the head is at negative time
    if hasattr(phases, '__len__') and len(phases) > 1:
        dy_dphase = np.polyfit(phases, centroids, 1)[0]
        sign_dy_dt = np.sign(dy_dphase)
    else:
        print('Warning! Time orientation of bunch length measurement cannot be determined!')
        sign_dy_dt = -1

    reverse_current = sign_dy_dt == -1

    if reverse != reverse_current:
        current = current[:,::-1]

    for curr in current:
        bp = beam_profile.BeamProfile(time_arr, curr, 1, 1)
        current_profiles0.append(bp)

    n_best = get_mean_profile(current_profiles0, outp='index', size=size, cutoff=cutoff)

    #import myplotstyle as ms
    #fignum = ms.plt.gcf().number
    #ms.figure('Debug')
    #sp = ms.plt.subplot(1,1,1)
    #for curr in current_profiles0:
    #    curr.plot_standard(sp)
    #current_profiles0[n_best].plot_standard(sp, color='black', lw=3)
    #ms.show()
    #import pdb; pdb.set_trace()
    #ms.plt.figure(fignum)

    return time_arr, current[n_best], current

def load_avg_blmeas_new(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)

    outp = {}
    calibration = blmeas_dict['Processed data']['Calibration'] * 1e-6/1e-15

    zc_strings = ['']
    if 'Beam images 2' in blmeas_dict['Processed data']:
        zc_strings.append(' 2')
    for n_zero_crossing, zc_string in enumerate(zc_strings, 1):
        images = blmeas_dict['Processed data']['Beam images'+zc_string]
        x_axis = blmeas_dict['Processed data']['x axis'+zc_string]*1e-6
        y_axis = blmeas_dict['Processed data']['y axis'+zc_string]*1e-6
        phases = blmeas_dict['Processed data']['Phase'+zc_string]
        centroids = blmeas_dict['Processed data']['Beam centroids'+zc_string]

        time_arr, curr_best, all_current = get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases)

        outp[n_zero_crossing] = {
                'time': time_arr,
                'current': curr_best,
                'current_reduced': curr_best - curr_best.min(),
                'all_current': all_current,
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis]
                }
    return outp

def load_avg_blmeas_old(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)
    outp = {}
    calibration = blmeas_dict['Meta_data']['Calibration factor'] * 1e-6/1e-15
    zc_strings = ['']
    if 'Beam images 2' in blmeas_dict['Raw_data']:
        zc_strings.append(' 2')

    for n_zero_crossing, zc_string in enumerate(zc_strings, 1):
        images = blmeas_dict['Raw_data']['Beam images'+zc_string]
        x_axis = blmeas_dict['Raw_data']['xAxis'+zc_string.replace(' ','')]*1e-6
        y_axis = blmeas_dict['Raw_data']['yAxis'+zc_string.replace(' ','')]*1e-6
        centroids = blmeas_dict['Raw_data']['Beam centroids'+zc_string].mean(axis=1)
        phases = np.arange(len(centroids))

        time_arr, curr_best, all_current = get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases)

        outp[n_zero_crossing] = {
                'time': time_arr,
                'current': curr_best,
                'current_reduced': curr_best - curr_best.min(),
                'all_current': all_current,
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis]
                }
    return outp

def load_avg_blmeas(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)

    if 'Raw_data' in blmeas_dict:
        return load_avg_blmeas_old(blmeas_dict)
    elif 'data' in blmeas_dict:
        return load_avg_blmeas_new(blmeas_dict['data'])
    else:
        return load_avg_blmeas_new(blmeas_dict)

def analyze_blmeas(file_or_dict, charge, voltage, force_cal=None, title=None, plot_all_images=False, error_of_the_average=True, separate_calibrations=True, profile_center_plot='Mean', current_cutoff=0.1e3, data_loader_options=None):
    if type(file_or_dict) is dict:
        all_data = file_or_dict
    else:
        all_data = h5_storage.loadH5Recursive(file_or_dict)
        if title is None:
            title = os.path.basename(file_or_dict)
    data = all_data['data']
    processed_data = data['Processed data']

    if data_loader_options is None:
        data_loader_options = {
                'subtract_quantile': 0.5,
                'subtract_absolute': None,
                'void_cutoff': [None, None],
                'cutX': None,
                'cutY': None,
                }

    textbbox = {'boxstyle': 'square', 'alpha': 0.75, 'facecolor': 'white', 'edgecolor': 'gray'}

    #voltage = abs(processed_data['Voltage axis'][0])
    energy_eV = data['Input data']['beamEnergy']*1e6
    #_ii = processed_data['Fit current profile_image_0']
    #_tt = processed_data['Good region time axis_image_0']
    #charge = np.trapz(_ii, _tt)*1e-12
    #print(charge)

    tds_freq_dict = {
            'SINDI02-DSCR075': 2.9988e9,
            'SINDI02-DLAC055': 2.9988e9,
            'S10DI02-DSCR020': 2.9988e9,
            'S10BD01-DSCR030': 2.9988e9,
            'SARCL01-DSCR170': 5.712e9,
            'SARCL02-DSCR280': 5.712e9,
            'SARBD01-DSCR050': 5.712e9,
            'SARBD02-DSCR050': 5.712e9,
            'SATBD01-DSCR120': 11.9952e9,
            'SATBD02-DSCR050': 11.9952e9,
            }
    streaking_dict = {
            'SINDI02-DSCR075': 'Y',
            'SINDI02-DLAC055': 'Y',
            'S10DI02-DSCR020': 'Y',
            'S10BD01-DSCR030': 'Y',
            'SARCL01-DSCR170': 'Y',
            'SARCL02-DSCR280': 'Y',
            'SARBD01-DSCR050': 'Y',
            'SARBD02-DSCR050': 'Y',
            'SATBD01-DSCR120': 'X',
            'SATBD02-DSCR050': 'X',
            }


    tds_freq = tds_freq_dict[data['Input data']['profileMonitor']]
    streaking_direction = streaking_dict[data['Input data']['profileMonitor']]

    zero_crossings = [1,]
    if 'Beam images 2' in processed_data:
        zero_crossings.append(2)

    fig_main = ms.figure('Main result %s' % title, figsize=(16,10))
    fig_main.subplots_adjust(wspace=0.35, hspace=0.3)
    subplot_main = ms.subplot_factory(2, 4)
    sp_ctr_main = 1

    sp_calib = subplot_main(sp_ctr_main, grid=True, title='Calibration phase scan', xlabel='$\Delta$Phase (deg)', ylabel='Screen centroid ($\mu$m)')
    sp_ctr_main += 1

    sp_residual = subplot_main(sp_ctr_main, grid=True, title='Calibration fit residuals', xlabel='$\Delta$Phase (deg)', ylabel='Fit residuals ($\mu$m)')
    sp_ctr_main += 1

    sp_parabola = subplot_main(sp_ctr_main, grid=True, title='Rms beam size parabola', xlabel='Voltage (MV)', ylabel='Rms beam size ($\mu$m)')
    sp_ctr_main += 1

    sp_bunch_duration = subplot_main(sp_ctr_main, grid=True, title='Bunch durations', xlabel='$\Delta$Phase (deg)', ylabel='Bunch duration (fs)')
    sp_ctr_main += 1

    sp_average_profile = subplot_main(sp_ctr_main, grid=True, title='Representative profiles', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr_main += 1

    sp_example_image = subplot_main(sp_ctr_main, grid=False, xlabel='x (mm)', ylabel='y (mm)')
    sp_ctr_main += 1

    calibrations = []
    calibrations_err = []
    all_projections = []
    all_streaked_axes = []
    all_phases_plot = []
    all_phases_rad = []
    all_fits = []

    for zero_crossing in zero_crossings:
        if zero_crossing == 1:
            zc_str = ''
        elif zero_crossing == 2:
            zc_str = ' 2'
        phases_deg = processed_data['Phase'+zc_str].astype(float)
        #phases_deg0 = phases_deg.copy()
        phase_old = phases_deg[0]
        for n_phase, phase in enumerate(phases_deg[1:], 1):
            if abs(phase - phase_old) > 180:
                phases_deg[n_phase] += 360
        #print(phases_deg0)
        #print(phases_deg)

        images = processed_data['Beam images'+zc_str]
        x_axis = processed_data['x axis'+zc_str].astype(float)*1e-6
        y_axis = processed_data['y axis'+zc_str].astype(float)*1e-6

        #zc_data = analyze(images, x_axis, y_axis, phases_deg, charge, energy_eV, streaking_direction)
        if x_axis[0] > x_axis[1]:
            x_axis = x_axis[::-1]
            images = images[:,:,:,::-1]
        if y_axis[0] > y_axis[1]:
            y_axis = y_axis[::-1]
            images = images[:,:,::-1]

        n_phases, n_images, leny, lenx = images.shape
        all_centroids = np.zeros([n_phases, n_images], float)
        data_objs = []

        if streaking_direction == 'Y':
            axis = y_axis
        elif streaking_direction == 'X':
            axis = x_axis

        projections = np.zeros([n_phases, n_images, len(axis)])
        all_projections.append(projections)
        all_streaked_axes.append(axis)

        for n_phase, phase_deg in enumerate(phases_deg):
            charge2 = np.ones(n_images)*charge
            energy_eV2 = np.ones(n_images)*energy_eV
            single_phase_data = data_loader.DataLoaderSimple(images[n_phase], x_axis, y_axis, charge2, energy_eV2, data_loader_options)
            data_objs.append(single_phase_data)
            single_phase_data.prepare_data()
            single_phase_data.init_images()
            single_phase_data.init_screen_distributions(streaking_direction)
            all_centroids[n_phase] = single_phase_data.sd_dict[streaking_direction]['mean']
            if streaking_direction == 'Y':
                projections[n_phase] = single_phase_data.image_data.sum(axis=2)
            elif streaking_direction == 'X':
                projections[n_phase] = single_phase_data.image_data.sum(axis=1)

            if zero_crossing == 1 and n_phase == n_phases//2:
                example_image = single_phase_data.images[n_images//2]
                gf_dict = example_image.plot_img_and_proj(sp_example_image, sqrt=True)
                sp_example_image.set_title('Phase %.2f deg, image %i' % (phase_deg, n_images//2), fontsize=None)
                textstr = 'Dim x/y: %i/%i\n' % (len(x_axis), len(y_axis))
                textstr += '$\sigma_x$: %.0f $\mu$m\n' % (gf_dict['gf_x'].sigma*1e6)
                textstr += '$\sigma_y$: %.0f $\mu$m' % (gf_dict['gf_y'].sigma*1e6)
                sp_example_image.text(0.05, 0.05, textstr, transform=sp_example_image.transAxes, verticalalignment='bottom', bbox=textbbox)

        centroids = np.mean(all_centroids, axis=1)
        centroids_err = np.std(all_centroids, axis=1)
        if error_of_the_average and n_images > 1:
            centroids_err /= np.sqrt(n_images-1)

        phases_rad = phases_deg * np.pi / 180
        all_phases_rad.append(phases_rad)

        if zero_crossing == 1:
            phases_rad_fit = phases_rad
        if zero_crossing == 2:
            phases_rad_fit = phases_rad - np.pi
        phases_rad_fit= phases_rad_fit - np.mean(phases_rad_fit)

        p, cov = np.polyfit(phases_rad_fit, centroids, 1, w=1/centroids_err, cov='unscaled')
        poly = np.poly1d(p)
        all_fits.append(poly)
        centroids_fit = poly(phases_rad_fit)
        residuals = centroids - centroids_fit
        chi_square = np.sum(residuals**2/centroids_err**2)
        nu = len(phases_deg) - 2
        chi_square_red = chi_square/nu

        calibration = p[0] * 2*np.pi*tds_freq
        calibration_error = np.sqrt(cov[0,0]) * 2*np.pi*tds_freq
        label = '%i: %.1f $\pm$ %.2f' % (zero_crossing, calibration*1e-9, calibration_error*1e-9)

        calibrations.append(calibration)
        calibrations_err.append(calibration_error)

        if zero_crossing == 2:
            phases_plot = phases_deg - 180
        else:
            phases_plot = phases_deg
        phases_plot -= phases_plot.mean()
        all_phases_plot.append(phases_plot)

        color = sp_calib.errorbar(phases_plot, centroids*1e6, yerr=centroids_err*1e6, ls='--')[0].get_color()
        sp_calib.plot(phases_plot, centroids_fit*1e6, color=color, label=label)

        sp_residual.errorbar(phases_plot, np.zeros_like(phases_plot), yerr=centroids_err*1e6, color=color, ls='None', capsize=5)
        sp_residual.scatter(phases_plot, residuals*1e6, marker='x', label='%i: %.2f' % (zero_crossing, chi_square_red))

        for n_phase, phase_deg in enumerate(phases_deg):
            if plot_all_images:
                figs, all_sps = data_objs[n_phase].plot_all(2, 3, title='Phase %.3f' % phase_deg, plot_kwargs={'sqrt': True}, subplots_adjust_kwargs={'wspace': 0.35})
                all_sps2 = []
                for sps in all_sps:
                    all_sps2.extend(sps)
                for n_image, centroid in enumerate(all_centroids[n_phase]):
                    if streaking_direction == 'X':
                        all_sps2[n_image].axvline(centroid*1e3, color='cyan')
                    if streaking_direction == 'Y':
                        all_sps2[n_image].axhline(centroid*1e3, color='cyan')

    calibrations = np.array(calibrations)
    calibrations_err = np.array(calibrations_err)

    if len(zero_crossings) == 2:
        weighted_calibration = np.sum(np.abs(calibrations)*calibrations_err**-1)/np.sum(calibrations_err**-1)
        a1, b1 = all_fits[0]
        a2, b2 = all_fits[1]
        phase_cross = (b2 - b1)/(a1 - a2)
        sp_calib.axvline(phase_cross*180/np.pi, color='black', ls='--')
        textstr = 'Fits cross at %.3f ($\Delta$ %0.3f) deg' % ((np.mean(all_phases_rad[0])+phase_cross)*180/np.pi, phase_cross*180/np.pi)
        textstr += '\nWeighted avg cal.: $\pm$%.2f $\mu$m/fs' % (weighted_calibration/1e9)
        sp_calib.text(0.05, 0.05, textstr, transform=sp_calib.transAxes, verticalalignment='bottom', bbox=textbbox)
    else:
        separate_calibrations = True

    ms.plt.figure(fig_main.number)

    voltages, beamsizes, beamsizes_err = np.zeros(3), np.zeros(3), np.zeros(3)

    for ctr, (zero_crossing, axis, projections) in enumerate(zip(zero_crossings, all_streaked_axes, all_projections)):
        if force_cal:
            cal = force_cal*np.sign(calibrations[ctr])
        elif separate_calibrations:
            cal = calibrations[ctr]
        else:
            cal = weighted_calibration*np.sign(calibrations[ctr])

        fwhm = np.zeros([projections.shape[0], projections.shape[1]], float)
        rms = fwhm.copy()
        gauss = fwhm.copy()

        sp = subplot_main(sp_ctr_main, title='Zero crossing %i, %i profiles' % (zero_crossing, fwhm.size), xlabel='t (fs)', ylabel='I (kA)')
        sp_ctr_main += 1

        time = axis / cal
        if cal < 0:
            time = time[::-1]
            projections = projections[:,:,::-1]

        all_profiles = []
        for n_phase, n_image in itertools.product(range(projections.shape[0]), range(projections.shape[1])):
            proj = projections[n_phase, n_image]
            profile = beam_profile.BeamProfile(time, proj, energy_eV, charge)
            cutoff_factor = current_cutoff*(time[1]-time[0])/profile.charge_dist.max()
            profile.aggressive_cutoff(cutoff_factor)
            gauss[n_phase, n_image] = profile.gaussfit.sigma
            profile.crop()
            profile.plot_standard(sp, center=profile_center_plot)
            all_profiles.append(profile)
            fwhm[n_phase, n_image] = profile.fwhm()
            rms[n_phase, n_image] = profile.rms()

        mean_profile = get_mean_profile(all_profiles)
        mean_profile.plot_standard(sp_average_profile, label='Zc %i' % zero_crossing, center=profile_center_plot)

        phases_plot = all_phases_plot[ctr]
        textstr = 'Head to the left.\nPlot center: %s\n' % profile_center_plot
        textstr += 'Calibration: %.2f $\mu$m/fs\n' % (cal*1e-9)
        textstr += 'Bunch durations:'
        for label, arr, color in [
                ('rms', rms, 'tab:blue'),
                ('fwhm', fwhm, 'tab:orange'),
                ('gauss $\sigma$', gauss, 'tab:green'),
                ]:
            label2 = 'Zc %i: %s' % (zero_crossing, label)
            if zero_crossing == 1:
                ls = 'solid'
            elif zero_crossing == 2:
                ls = 'dashed'
                label2 = None
            sp_bunch_duration.errorbar(phases_plot, np.mean(arr, axis=1)*1e15, yerr=np.std(arr, axis=1)*1e15, label=label2, color=color, ls=ls, capsize=5)
            textstr += '\n%s:\t%.2f $\pm$ %.2f fs' % (label, np.mean(arr)*1e15, np.std(arr)*1e15)
        sp.text(0.05, 0.95, textstr, transform=sp.transAxes, verticalalignment='top', bbox=textbbox)

        if len(zero_crossings) == 2:
            beamsizes[ctr*2] = np.mean(rms)*np.abs(cal)
            beamsizes_err[ctr*2] = np.std(rms)*np.abs(cal)
            voltages[ctr*2] = voltage*(-1)**ctr

    if len(zero_crossings) == 2:
        voltages[1] = 0
        beamsizes[1] = processed_data['Beam sizes without streaking']*1e-6
        sp_parabola.errorbar(voltages/1e6, beamsizes*1e6, yerr=beamsizes_err*1e6, ls='None', capsize=5)
        if beamsizes[1] != 0:
            beamsizes_err[1] = processed_data['Beam sizes without streaking errors']*1e-6
            par_fit = np.poly1d(np.polyfit(voltages, beamsizes, 2, w=1/beamsizes_err))
            xx = np.linspace(voltages[0], voltages[2], 100)
            yy = par_fit(xx)
            sp_parabola.plot(xx/1e6, yy*1e6, ls='--')
            textstr = 'First zero crossing to the right.\n'
            textstr += r'$\sigma$ ($\mu$m) = $%.2f \cdot E^2 (\mathrm{MV}^2) %+.2f \cdot E (\mathrm{MV}) %+.2f$ $\mu$m' % (par_fit[2]*1e18, par_fit[1]*1e12, par_fit[0]*1e6)
            min_volt = -par_fit[1]/(2*par_fit[2])
            sp_parabola.axvline(min_volt*1e-6, color='black', ls='--')
            textstr += '\nMin. rms beamsize at %.2f MV' % (min_volt*1e-6)
            sp_parabola.text(0.02, 0.5, textstr, transform=sp_parabola.transAxes, verticalalignment='top', bbox=textbbox, fontsize='x-small')
        else:
            sp_parabola.text(0.02, 0.5, 'Unstreaked beam size not measured', transform=sp_parabola.transAxes, verticalalignment='top', bbox=textbbox, fontsize='x-small')


    sp_calib.legend(loc='upper right', title='Zero crossing: cal. ($\mu$m/fs)')
    sp_residual.legend(loc='upper right', title=r'Zero crossing: $\chi^2_\nu$')
    sp_bunch_duration.legend()
    if len(zero_crossings) == 2:
        sp_average_profile.legend()

