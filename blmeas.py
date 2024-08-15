import itertools
import os
import copy
import numpy as np

from . import beam_profile
from . import config
from . import data_loader
from . import h5_storage
from . import myplotstyle as ms

def tilt_reconstruction(profile1, profile2, debug=False):
    # see appendix of Schmidt et al., https://doi.org/10.1103/PhysRevAccelBeams.23.062801
    profile1 = copy.deepcopy(profile1)
    profile2 = copy.deepcopy(profile2)
    assert profile1.total_charge == profile2.total_charge
    assert profile1.energy_eV == profile2.energy_eV
    for profile in profile1, profile2:
        profile.center('Mean')
        profile.crop()
    tmin = min(profile1.time[0], profile2.time[0])
    tmax = max(profile1.time[-1], profile2.time[-1])
    tlen = max(len(profile1), len(profile2))
    new_time = np.linspace(tmin, tmax, tlen)

    cumsum = np.zeros_like(new_time)
    new_charges = []
    for ctr, profile in enumerate([profile1, profile2]):
        new_charge = np.interp(new_time, profile.time, profile.charge_dist, left=0, right=0)
        new_charges.append(new_charge)
        cumsum += np.cumsum(new_charge)
    new_charge_dist = np.append(np.diff(cumsum/2), [0])
    new_profile = beam_profile.BeamProfile(new_time, new_charge_dist, profile1.energy_eV, profile1.total_charge)
    new_profile.center('Mean')

    if debug:
        old_fignum = ms.plt.gcf()
        sp = ms.plt.subplot(1, 1, 1)
        sp.plot(new_time, new_charges[0], label='New charge 1')
        sp.plot(new_time, new_charges[1], label='New charge 2')
        sp.plot(new_time, new_charge_dist, label='Combined')
        sp.legend()
        ms.plt.figure(old_fignum)

    return new_profile

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
            tt = np.linspace(min(bp1.time[0], bp2.time[0]), max(bp1.time[-1], bp2.time[-1]), size)
            minus = np.interp(tt, bp1.time, bp1.charge_dist, left=0, right=0) - np.interp(tt, bp2.time, bp2.charge_dist, left=0, right=0)
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
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis],
                'calibration': calibration,
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
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis],
                'calibration': calibration,
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

tds_freq_dict = {
        'SINDI01': 2.9988e9,
        'S30CB14': 5.712e9,
        'SATMA02': 11.9952e9,
        }

screen_tds_dict = {
        'SINDI02-DSCR075': 'SINDI01',
        'SINDI02-DLAC055': 'SINDI01',
        'S10DI02-DSCR020': 'SINDI01',
        'S10BD01-DSCR030': 'SINDI01',
        'SARCL01-DSCR170': 'S30CB14',
        'SARCL02-DSCR280': 'S30CB14',
        'SARBD01-DSCR050': 'S30CB14',
        'SARBD02-DSCR050': 'S30CB14',
        'SATBD01-DSCR120': 'SATMA02',
        'SATBD02-DSCR050': 'SATMA02',
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


def analyze_blmeas(file_or_dict, force_charge=None, force_cal=None, title=None, plot_all_images=False, error_of_the_average=True, separate_calibrations=False, current_cutoff=0.1e3, data_loader_options=None, streaking_direction=None):

    outp = {}
    if type(file_or_dict) is dict:
        all_data = file_or_dict
    else:
        all_data = h5_storage.loadH5Recursive(file_or_dict)
        if title is None:
            title = os.path.basename(file_or_dict)
    data = all_data['data']
    processed_data = data['Processed data']

    if data_loader_options is None:
        data_loader_options = config.get_default_data_loader_options()
        data_loader_options.update({
                'subtract_quantile': 0.5,
                'subtract_absolute': None,
                'void_cutoff': [None, None],
                'cutX': None,
                'cutY': None,
                'screen_cutoff': 0,
                'screen_cutoff_relative': True,
                'screen_cutoff_edge_points': 10,
                'screen_cutoff_relative_factor': 2,
                })

    energy_eV = data['Input data']['beamEnergy']*1e6
    _ii = processed_data['Current profile_image_0']
    _tt = processed_data['Good region time axis_image_0']
    if force_charge:
        charge = force_charge
    else:
        charge = np.abs(np.trapz(_ii, _tt*1e-15))
        #print('Charge %.2e' % charge)
    outp['charge'] = charge

    tds = screen_tds_dict[data['Input data']['profileMonitor']]
    tds_freq = tds_freq_dict[tds]
    if streaking_direction is None:
        streaking_direction = streaking_dict[data['Input data']['profileMonitor']]
    outp['tds'] = tds
    outp['tds_freq'] = tds_freq
    outp['streaking_direction'] = streaking_direction

    zero_crossings = [1,]
    if 'Beam images 2' in processed_data:
        zero_crossings.append(2)
    outp['zero_crossings'] = zero_crossings

    calibrations = []
    calibrations_err = []
    all_projections = []
    all_streaked_axes = []
    all_phases_plot = []
    all_phases_rad = []
    all_fits = []

    for zero_crossing in zero_crossings:
        outp[zero_crossing] = {}

        if zero_crossing == 1:
            zc_str = ''
        elif zero_crossing == 2:
            zc_str = ' 2'
        _phases_deg = processed_data['Phase'+zc_str]
        if type(_phases_deg) is float:
            phases_deg = np.array([_phases_deg])
        else:
            phases_deg = _phases_deg.astype(float).copy()
        outp[zero_crossing]['phases_raw'] = phases_deg.copy()
        #phases_deg0 = phases_deg.copy()
        phase_old = phases_deg[0]
        for n_phase, phase in enumerate(phases_deg[1:], 1):
            if abs(phase - phase_old) > 180:
                phases_deg[n_phase] += 360
            phase_old = phases_deg[n_phase]
        outp[zero_crossing]['phases_deg'] = phases_deg
        #print(phases_deg0)
        #print(phases_deg)

        images = processed_data['Beam images'+zc_str].astype(float)
        x_axis = processed_data['x axis'+zc_str].astype(float)*1e-6
        y_axis = processed_data['y axis'+zc_str].astype(float)*1e-6
        outp[zero_crossing]['x_axis'] = x_axis
        outp[zero_crossing]['y_axis'] = y_axis

        if streaking_direction == 'X':
            y_axis, x_axis = x_axis, y_axis

        if len(images.shape) == 3:
            images = images[np.newaxis,...]
        n_phases, n_images, leny, lenx = images.shape
        outp[zero_crossing]['n_images'] = n_images
        outp[zero_crossing]['n_phases'] = n_phases
        if x_axis[0] > x_axis[1]:
            x_axis = x_axis[::-1]
            images = images[:,:,:,::-1]
        if y_axis[0] > y_axis[1]:
            y_axis = y_axis[::-1]
            images = images[:,:,::-1]

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

            if n_phase == len(phases_deg)//2:
                outp[zero_crossing]['example_image'] = single_phase_data.images[n_images//2]

        if n_phases >= 2:
            centroids = np.nanmean(all_centroids, axis=1)
            centroids_err = np.nanstd(all_centroids, axis=1)
            if error_of_the_average and n_images > 1:
                centroids_err /= np.sqrt(n_images-1)

            outp[zero_crossing]['centroids'] = centroids
            outp[zero_crossing]['centroids_err'] = centroids_err

            phases_rad = phases_deg * np.pi / 180
            all_phases_rad.append(phases_rad)

            if zero_crossing == 1:
                phases_rad_fit = phases_rad
            if zero_crossing == 2:
                phases_rad_fit = phases_rad - np.pi
            phases_rad_fit = phases_rad_fit - np.mean(phases_rad_fit)

            weights0 = 1/centroids_err
            weights = np.clip(weights0, 0, np.mean(weights0))

            notnan = ~np.isnan(centroids)
            p, cov = np.polyfit(phases_rad_fit[notnan], centroids[notnan], 1, w=weights[notnan], cov='unscaled')
            poly = np.poly1d(p)
            outp[zero_crossing]['polyfit'] = p
            outp[zero_crossing]['polyfit_cov'] = cov
            all_fits.append(poly)
            centroids_fit = poly(phases_rad_fit)
            outp[zero_crossing]['centroids_fit'] = centroids_fit
            residuals = centroids - centroids_fit
            chi_square = np.sum(residuals**2/centroids_err**2)
            nu = len(phases_deg) - 2
            chi_square_red = chi_square/nu
            outp[zero_crossing]['residuals'] = residuals
            outp[zero_crossing]['chi_square_red'] = chi_square_red

            calibration = p[0] * 2*np.pi*tds_freq
            outp[zero_crossing]['calibration_fit'] = calibration
            calibration_error = np.sqrt(cov[0,0]) * 2*np.pi*tds_freq

            calibrations.append(calibration)
            calibrations_err.append(calibration_error)

            if zero_crossing == 2:
                phases_plot = phases_deg - 180
            else:
                phases_plot = phases_deg.copy()
            phases_plot -= phases_plot.mean()
            outp[zero_crossing]['phases_plot'] = phases_plot
            all_phases_plot.append(phases_plot)

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
        elif not force_cal:
            raise ValueError('Not enough phase set points and calibration not provided')
        else:
            outp[zero_crossing]['calibration_fit'] = 0
            outp[zero_crossing]['chi_square_red'] = 0
            outp[zero_crossing]['residuals'] = 0

    if n_phases >= 2:
        calibrations = np.array(calibrations)
        calibrations_err = np.array(calibrations_err)
    else:
        calibrations = np.array([force_cal, -force_cal])
        calibrations_err = None

    if len(zero_crossings) == 2:
        if calibrations_err is None:
            weighted_calibration = np.mean(np.abs(calibrations))
        else:
            weighted_calibration = np.sum(np.abs(calibrations)*calibrations_err**-1)/np.sum(calibrations_err**-1)
    else:
        separate_calibrations = True
    if force_cal:
        weighted_calibration = force_cal
    else:
        weighted_calibration = np.mean(np.abs(calibrations))

    voltages, beamsizes, beamsizes_err = np.zeros(3), np.zeros(3)*np.nan, np.zeros(3)*np.nan
    outp['beamsizes'] = beamsizes
    outp['beamsizes_err'] = beamsizes_err
    outp['voltages'] = voltages
    outp['calibrations0'] = calibrations.copy()
    outp['calibrations_err'] = calibrations_err
    outp['weighted_calibration'] = weighted_calibration
    outp['all_phases_rad'] = np.array(all_phases_rad)
    outp['all_phases_plot'] = np.array(all_phases_plot)
    outp['separate_calibrations'] = separate_calibrations

    for ctr, (zero_crossing, axis, projections) in enumerate(zip(zero_crossings, all_streaked_axes, all_projections)):
        if force_cal:
            cal = force_cal*np.sign(calibrations[ctr])
        elif separate_calibrations:
            cal = calibrations[ctr]
        else:
            cal = weighted_calibration*np.sign(calibrations[ctr])

        calibrations[ctr] = cal

        fwhm = np.zeros([projections.shape[0], projections.shape[1]], float)
        rms = fwhm.copy()
        gauss = fwhm.copy()

        outp[zero_crossing]['fwhm'] = fwhm
        outp[zero_crossing]['rms'] = rms
        outp[zero_crossing]['gauss'] = gauss

        time = axis / cal
        if cal < 0:
            time = time[::-1]
            projections = projections[:,:,::-1]

        all_profiles = []
        outp[zero_crossing]['profiles'] = all_profiles
        for n_phase, n_image in itertools.product(range(projections.shape[0]), range(projections.shape[1])):
            proj = projections[n_phase, n_image]
            profile = beam_profile.BeamProfile(time, proj, energy_eV, charge)
            cutoff_factor = current_cutoff*(time[1]-time[0])/profile.charge_dist.max()
            gauss[n_phase, n_image] = profile.gaussfit.sigma
            profile.aggressive_cutoff(cutoff_factor)
            profile.crop()
            all_profiles.append(profile)
            fwhm[n_phase, n_image] = profile.fwhm()
            rms[n_phase, n_image] = profile.rms()

        outp[zero_crossing]['y_rms'] = rms*np.abs(cal)

        mean_profile = get_mean_profile(all_profiles)
        outp[zero_crossing]['representative_profile'] = mean_profile


        if len(zero_crossings) == 2:
            beamsizes[ctr*2] = np.nanmean(rms)*np.abs(cal)
            beamsizes_err[ctr*2] = np.nanstd(rms)*np.abs(cal)
            voltage = abs(processed_data['Voltage axis'][0])*1e6
            voltages[ctr*2] = voltage*(-1)**ctr

    corr_rms_blen = 0
    corr_rms_blen_err = 0
    resolution = 0
    outp['corrected_profile'] = None
    outp['beamsizes_sq_err'] = 0
    outp['calibrations'] = calibrations
    if len(zero_crossings) == 2:
        voltages[1] = 0
        beamsizes[1] = processed_data['Beam sizes without streaking']*1e-6
        if beamsizes[1] != 0:
            beamsizes_err[1] = processed_data['Beam sizes without streaking errors']*1e-6
            beamsizes_sq_err = 2*beamsizes*beamsizes_err
            popt, pcov = np.polyfit(voltages, beamsizes**2, 2, w=1/beamsizes_sq_err, cov='unscaled')
            outp['parabola_popt'] = popt

            par_fit = np.poly1d(popt)
            xx = np.linspace(voltages[0], voltages[2], 100)
            yy = par_fit(xx)
            outp['parabola_x'] = xx
            outp['parabola_y'] = yy

            corr_rms_blen = np.sqrt(popt[0])*voltage/weighted_calibration
            corr_rms_blen_err = corr_rms_blen/(2*popt[0])*np.sqrt(pcov[0,0])
            resolution = beamsizes[1] / weighted_calibration
            outp['resolution'] = resolution
            outp['corr_rms_blen'] = corr_rms_blen
            outp['corr_rms_blen_err'] = corr_rms_blen_err
            outp['beamsizes_sq_err'] = 2*beamsizes*beamsizes_err

    if len(zero_crossings) == 2:
        outp['corrected_profile'] = tilt_reconstruction(outp[1]['representative_profile'], outp[2]['representative_profile'])

    return outp

