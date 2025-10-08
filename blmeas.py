import itertools
import os
import copy
import numpy as np

from . import beam_profile
from . import gaussfit
from . import image_analysis
from . import plot_results
from . import config
from . import data_loader
from . import h5_storage

def tilt_reconstruction2(profile1, profile2):
    # Algorithm by H. Loos, as described in Schmidt et al., Benchmarking coherent radiation spectroscopy as a tool for high-resolution bunch shape reconstruction at free-electron lasers,PRAB 23, 062801 (2020) https://doi.org/10.1103/PhysRevAccelBeams.23.062801
    assert profile1.total_charge == profile2.total_charge
    assert profile1.energy_eV == profile2.energy_eV
    n_interp = max(len(profile1), len(profile2))

    sd_plus = beam_profile.ScreenDistribution(profile1.time, profile1.charge_dist, subtract_min=False, total_charge=profile1.total_charge)
    sd_minus = beam_profile.ScreenDistribution(-profile2.time[::-1], profile2.charge_dist[::-1], subtract_min=False, total_charge=profile1.total_charge)

    q_plus = np.cumsum(sd_plus.intensity)*(sd_plus.x[1] - sd_plus.x[0])
    q_minus = np.cumsum(sd_minus.intensity[::-1]) * (sd_minus.x[1] - sd_minus.x[0])
    q_interp_min = min(np.min(q_plus), np.min(q_minus))
    q_interp_max = max(np.max(q_plus), np.max(q_minus))
    q_interp = np.linspace(q_interp_min, q_interp_max, n_interp)

    x_interp_minus = np.interp(q_interp, q_minus, sd_minus.x[::-1])
    x_interp_plus = np.interp(q_interp, q_plus, sd_plus.x)

    x_mean = (x_interp_plus - x_interp_minus)/2
    x_mean_evenly = np.linspace(x_mean.min(), x_mean.max(), len(x_mean))
    q_interp_evenly = np.interp(x_mean_evenly, x_mean, q_interp)

    charge_dist_mean = np.diff(q_interp_evenly)
    corrected_profile = beam_profile.BeamProfile(x_mean_evenly[:-1], charge_dist_mean, profile1.energy_eV, profile1.total_charge)
    corrected_profile.center('Mean')

    outp = {
            'corrected_profile': corrected_profile,
            'sd_plus': sd_plus,
            'sd_minus': sd_minus,
            'q_plus': q_plus,
            'q_minus': q_minus,
            'q_interp': q_interp,
            'x_mean': x_mean,
            }
    return outp

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

    min_t = min(bp._xx[0] for bp in profile_list)
    max_t = max(bp._xx[-1] for bp in profile_list)
    tt = np.linspace(min_t, max_t, size)
    for n_row in range(len(squares_mat)):
        for n_col in range(n_row):
            bp1 = profile_list[n_row]
            bp2 = profile_list[n_col]
            minus = np.interp(tt, bp1._xx, bp1._yy, left=0, right=0) - np.interp(tt, bp2._xx, bp2._yy, left=0, right=0)
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

def straighten_out_phase(phases):
    phases = phases.copy()
    phase_old = phases[0]
    for n_phase, phase in enumerate(phases[1:], 1):
        if abs(phase - phase_old) > 180:
            phases[n_phase] += 360
        phase_old = phases[n_phase]
    return phases

def analyze_blmeas(file_or_dict, force_charge=False, force_cal=False, title=None, plot_all_images=False, error_of_the_average=True, separate_calibrations=False, current_cutoff=0.1e3, data_loader_options=None, streaking_direction=None, aggressive_cutoff=True, forced_charge=None, forced_cal=None):

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
        data_loader_options = config.get_blmeas_data_loader_options()
    energy_eV = data['Input data']['beamEnergy']*1e6
    outp['energy_eV'] = energy_eV
    _ii = processed_data['Current profile_image_0']
    _tt = processed_data['Good region time axis_image_0']
    charge0 = np.abs(np.trapz(_ii, _tt*1e-15))
    outp['charge0'] = charge0
    if force_charge:
        charge = forced_charge
    else:
        charge = charge0
    outp['charge'] = charge

    profile_monitor = data['Input data']['profileMonitor']
    tds = screen_tds_dict[profile_monitor]
    tds_freq = tds_freq_dict[tds]
    if streaking_direction is None:
        streaking_direction = streaking_dict[data['Input data']['profileMonitor']]
    outp['tds'] = tds
    outp['tds_freq'] = tds_freq
    outp['streaking_direction'] = streaking_direction
    outp['profile_monitor'] = profile_monitor

    zero_crossings = [1,]
    if 'Beam images 2' in processed_data:
        zero_crossings.append(2)
    outp['zero_crossings'] = zero_crossings

    calibrations = []
    calibrations_err = []
    all_projections = []
    all_streaked_axes = []
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

        phases_deg = straighten_out_phase(phases_deg)
        outp[zero_crossing]['phases_deg'] = phases_deg

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
            fit_phase_delta = np.mean(phases_rad_fit)
            outp[zero_crossing]['fit_phase_delta'] = fit_phase_delta
            phases_rad_fit = phases_rad_fit - fit_phase_delta

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

            for n_phase, phase_deg in enumerate(phases_deg):
                if plot_all_images:
                    figs, all_sps = data_objs[n_phase].plot_all(2, 3, title='Zc %i Phase %.3f (number %i)' % (zero_crossing, phase_deg, n_phase), plot_kwargs={'sqrt': True}, subplots_adjust_kwargs={'wspace': 0.35})
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
        calibrations = np.array([forced_cal, -forced_cal])
        calibrations_err = None

    if len(zero_crossings) == 2 and n_phases >= 2:
        a1, b1 = outp[1]['polyfit']
        a2, b2 = outp[2]['polyfit']
        phase_cross_rel = (b2 - b1)/(a1 - a2)
        phase_cross_abs = phase_cross_rel + outp[1]['fit_phase_delta']
        outp['phase_cross_rel'] = phase_cross_rel
        outp['phase_cross_abs'] = phase_cross_abs
        if calibrations_err is None:
            weighted_calibration = np.mean(np.abs(calibrations))
        else:
            weighted_calibration = np.sum(np.abs(calibrations)*calibrations_err**-1)/np.sum(calibrations_err**-1)
    elif len(zero_crossings) == 1 and n_phases >= 2:
        weighted_calibration = abs(calibrations[0])
    else:
        weighted_calibration = None
        separate_calibrations = True
        outp['phase_cross_rel'] = None
        outp['phase_cross_abs'] = None

    if force_cal:
        weighted_calibration = abs(forced_cal)

    voltages, beamsizes, beamsizes_err = np.zeros(3), np.zeros(3)*np.nan, np.zeros(3)*np.nan
    outp['beamsizes'] = beamsizes
    outp['beamsizes_err'] = beamsizes_err
    outp['voltages'] = voltages
    outp['calibrations0'] = calibrations.copy()
    outp['calibrations_err'] = calibrations_err
    outp['weighted_calibration'] = weighted_calibration
    outp['all_phases_rad'] = np.array(all_phases_rad)
    outp['separate_calibrations'] = separate_calibrations

    for ctr, (zero_crossing, axis, projections) in enumerate(zip(zero_crossings, all_streaked_axes, all_projections)):
        if force_cal:
            cal = abs(forced_cal)*np.sign(calibrations[ctr])
            #print(calibrations[ctr], cal)
        elif separate_calibrations:
            cal = calibrations[ctr]
        else:
            cal = weighted_calibration*np.sign(calibrations[ctr])
        #print('Calibration for zero crossing %i: %.2f um/fs' % (zero_crossing, cal/1e9))

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
            if aggressive_cutoff:
                profile.aggressive_cutoff(cutoff_factor)
            else:
                profile.cutoff(cutoff_factor)
            profile.cutoff(cutoff_factor)
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
        outp['corrected_profile'] = tilt_reconstruction2(outp[1]['representative_profile'], outp[2]['representative_profile'])['corrected_profile']

    return outp

def analyze_separate_measurements(file_or_dict1, file_or_dict2, force_cal1, force_cal2, do_plot=False, **blmeas_kwargs):
    assert force_cal1 == -force_cal2

    result1 = analyze_blmeas(file_or_dict1, force_cal=force_cal1, **blmeas_kwargs)
    result2 = analyze_blmeas(file_or_dict2, force_cal=force_cal2, **blmeas_kwargs)
    return analyze_separate_results(result1, result2, do_plot=do_plot)

def analyze_separate_results(result1, result2, do_plot=False):
    outp = {}
    outp['result1'] = result1
    outp['result2'] = result2

    if do_plot:
        plot_results.plot_blmeas_analysis(result1)
        plot_results.plot_blmeas_analysis(result2)

    representative_profiles = [result1[1]['representative_profile'], result2[1]['representative_profile']]
    mean_charge = np.mean([x.total_charge for x in representative_profiles])

    for profile in representative_profiles:
        profile.total_charge = mean_charge
        profile.center('Mean')

    outp['corrected_profile'] = tilt_reconstruction2(*representative_profiles)['corrected_profile']
    return outp


def get_projections(images, x_axis, y_axis, charge, streaking_direction):

    data_loader_options = config.get_blmeas_data_loader_options()
    if len(images.shape) == 4:
        n_phases, n_images, len_y, len_x = images.shape
        images_reshaped = images.reshape(n_phases*n_images, len_y, len_x)
    else:
        images_reshaped = images

    zc_data = data_loader.DataLoaderSimple(images_reshaped, x_axis, y_axis, charge, 1, data_loader_options)
    zc_data.prepare_data()
    #zc_data.init_images()
    zc_data.init_screen_distributions(streaking_direction)

    if streaking_direction == 'Y':
        projections = zc_data.image_data.sum(axis=2)
    elif streaking_direction == 'X':
        projections = zc_data.image_data.sum(axis=1)
    example_image = image_analysis.Image(images_reshaped[len(images_reshaped)//2], x_axis, y_axis, charge=charge)
    centroids = zc_data.sd_dict[streaking_direction]['mean']
    return projections, centroids, example_image

def analyze_zero_crossing(phases_deg, projections, centroids, tds_freq, example_image):

    phases_deg0 = phases_deg.copy()
    phases_deg = np.ravel(phases_deg)
    #mask = np.ones_like(phases_deg, bool)
    #for ctr, phase in enumerate(phases_deg[:-1]):
    #    if abs(phases_deg[ctr+1] - phases_deg[ctr]) > 10:
    #        mask[:ctr+1] = 0
    #if np.any(mask == 0):
    #    print('Removed %i images due to wrong phase' % (mask==0).sum())


    #phases_deg = phases_deg[mask]
    #projections = projections[mask]
    #centroids = centroids[mask]

    phases_deg = straighten_out_phase(phases_deg)

    phases_rad = phases_deg * np.pi / 180
    fit_phase_delta = np.mean(phases_rad)
    phases_rad_fit = phases_rad - fit_phase_delta

    notnan = ~np.isnan(centroids)
    p, cov = np.polyfit(phases_rad_fit[notnan], centroids[notnan], 1, cov=True)
    calibration = p[0] * 2*np.pi*tds_freq
    calibration_err = np.sqrt(cov[0,0]) * 2*np.pi*tds_freq
    #print('Calibration: %.3f+/-%.3f um/fs' % (calibration/1e9, calibration_err/1e9))

    poly = np.poly1d(p)
    centroids_fit = poly(phases_rad_fit)
    residuals = centroids - centroids_fit

    outp = {
            'centroids': centroids,
            'projections': projections,
            'example_image': example_image,
            'polyfit': p,
            'centroids_fit': centroids_fit,
            'calibration_fit': calibration,
            'calibration_fit_err': calibration_err,
            'phases_deg': phases_deg,
            'phases_deg0': phases_deg0,
            'x_axis': example_image.x_axis,
            'y_axis': example_image.y_axis,
            'n_images': len(projections),
            'n_phases': len(phases_deg),
            'charge': example_image.charge,
            'residuals': residuals,
            }
    return outp

class LongitudinalBeamMeasurement:
    def __init__(self, data_files_or_dict, **kwargs):
        self.analysis_config = {
                'force_calibration': False,
                'forced_calibration': 0,
                'current_cutoff': 5,
                'aggressive_cutoff': True,
                'force_charge': False,
                'forced_charge': 200e-12,
                'n_repeat': 2,
                }
        self.analysis_config.update(kwargs)
        if type(data_files_or_dict) is dict:
            self.data = data_files_or_dict
        elif type(data_files_or_dict) is str:
            self.data = h5_storage.loadH5Recursive(data_files_or_dict)
        elif type(data_files_or_dict) in (list, tuple) and len(data_files_or_dict) == 2:
            self.data = h5_storage.loadH5Recursive(data_files_or_dict[0])
            self.data['input']['measure_both_zc'] = True
            data2 = h5_storage.loadH5Recursive(data_files_or_dict[1])
            scans1 = sorted(self.data['raw_data'].keys())
            scans2 = sorted(data2['raw_data'].keys())
            raw_data0 = self.data['raw_data']
            self.data['raw_data'] = {
                    'scan0000': raw_data0[scans1[0]],
                    'scan0001': data2['raw_data'][scans2[0]],
                    }
            if len(scans1) > 1:
                self.data['raw_data']['scan0002'] = raw_data0[scans1[1]]
        else:
            raise ValueError('Cannot deal with data_filesor_dict', data_files_or_dict)

    def analyze_current_measurement(self):
        data = self.data
        self.scans = sorted(data['raw_data'].keys())
        self.tds_name = data['input']['tds']
        self.tds_freq = tds_freq_dict[self.tds_name]
        self.phase_pv = self.tds_name+'-RLLE-DSP'
        self.zero_crossings = [1, 2] if data['input']['measure_both_zc'] else [1, ]
        self.unstreaked_measured = len(self.scans) > len(self.zero_crossings)
        try:
            self.voltage = data['snapshot'][self.scans[0]][self.tds_name+'-RSYS']['SET-ACC-VOLT']*1e6
        except KeyError:
            print('No voltage in snapshot')
            self.voltage = 1

        all_charge = []
        charge_bpm_name = data['input']['charge_bpm_name']
        for scan in self.scans:
            charge_dict = data['raw_data'][scan][charge_bpm_name]
            this_charge = np.ravel(charge_dict['Q1']['data'])
            if 'SATMA02' not in charge_bpm_name:
                this_charge += np.ravel(charge_dict['Q2']['data'])
            charge_list = list(this_charge*1e-12)
            all_charge.extend(charge_list)
        charge0 = np.median(all_charge)
        charge = self.analysis_config['forced_charge'] if self.analysis_config['force_charge'] else charge0

        result = self.result = {
                'charge0': charge0,
                'charge': charge,
                'zero_crossings': self.zero_crossings,
                'calibrations': np.zeros(2, float),
                'calibrations_err': np.zeros(2, float),
                'voltages': np.array([0, 0, 0], float),
                'beamsizes': np.array([0, 0, 0], float),
                'beamsizes_err': np.array([0, 0, 0], float),
                'input': self.data['input'],
                'analysis_config': self.analysis_config
                }

        for ctr, (zero_crossing, scan) in enumerate(zip(self.zero_crossings, self.scans)):
            phases_deg = data['raw_data'][scan][self.phase_pv]['PHASE-VS']['data']
            images = data['raw_data'][scan]['image']['data'].astype(float)
            x_axis = data['raw_data'][scan]['x_axis']['data'].astype(float)*1e-6
            y_axis = data['raw_data'][scan]['y_axis']['data'].astype(float)*1e-6

            while len(x_axis.shape) > 1:
                x_axis = x_axis[0]
            while len(y_axis.shape) > 1:
                y_axis = y_axis[0]

            if x_axis[1] < x_axis[0]:
                x_axis = x_axis[::-1]
                images = images[...,::-1]
            if y_axis[1] < y_axis[0]:
                y_axis = y_axis[::-1]
                images = images[...,::-1,:]

            projections, centroids, example_image = get_projections(images, x_axis, y_axis, charge, self.data['input']['streaking_direction'])
            result[zero_crossing] = analyze_zero_crossing(phases_deg, projections, centroids, self.tds_freq, example_image)

        print('Calibrations in um/fs:', np.array([result[zero_crossing]['calibration_fit'] for zero_crossing in self.zero_crossings])/1e9)
        self.calc_current_profiles()

        for _ in range(self.analysis_config['n_repeat']):
            for ctr, (zero_crossing, scan) in enumerate(zip(self.zero_crossings, self.scans)):
                phases_deg = result[zero_crossing]['phases_deg0']
                projections = result[zero_crossing]['projections']
                centroids = np.array([p.mean() for p in result[zero_crossing]['profiles']]) * result[zero_crossing]['calibration_fit']
                example_image = result[zero_crossing]['example_image']
                result[zero_crossing] = analyze_zero_crossing(phases_deg, projections, centroids, self.tds_freq, example_image)

            print('Calibrations in um/fs:', np.array([result[zero_crossing]['calibration_fit'] for zero_crossing in self.zero_crossings])/1e9)
            self.calc_current_profiles()
        return result

    def calc_current_profiles(self):
        result = self.result
        streaking_direction = self.data['input']['streaking_direction']
        for zero_crossing in result['zero_crossings']:
            result['calibrations'][zero_crossing-1] = result[zero_crossing]['calibration_fit']
            result['calibrations_err'][zero_crossing-1] = result[zero_crossing]['calibration_fit_err']
            result['voltages'][(zero_crossing-1)*2] = self.voltage*(-1)**zero_crossing
            has_cal = 'calibration_fit' in result[zero_crossing]
            if self.analysis_config['force_calibration']:
                cal = self.analysis_config['forced_calibration']
                if has_cal:
                    cal = abs(cal) * np.sign(result[zero_crossing]['calibration_fit'])
            elif has_cal:
                cal = result[zero_crossing]['calibration_fit']
            else:
                raise ValueError('Calibration must be specified!')
            result[zero_crossing]['applied_calibration'] = cal

            axis = result[zero_crossing]['%s_axis' % streaking_direction.lower()] / cal
            projections = result[zero_crossing]['projections']

            if axis[1] < axis[0]:
                axis = axis[::-1]
                projections = projections[:,::-1]

            fwhm = result[zero_crossing]['fwhm'] = np.zeros(len(projections), float)
            rms = result[zero_crossing]['rms'] = fwhm.copy()
            gauss = result[zero_crossing]['gauss'] = fwhm.copy()

            all_profiles = []
            result[zero_crossing]['profiles'] = all_profiles

            for ctr, proj in enumerate(projections):
                profile = beam_profile.BeamProfile(axis, proj, 1, result[zero_crossing]['charge'])
                cutoff_factor = self.analysis_config['current_cutoff']*(axis[1]-axis[0])/profile.charge_dist.max()
                gauss[ctr] = profile.gaussfit.sigma
                if cutoff_factor <= 1:
                    if self.analysis_config['aggressive_cutoff']:
                        profile.aggressive_cutoff(cutoff_factor)
                    else:
                        profile.cutoff(cutoff_factor)
                else:
                    print('Cutoff factor too large. No cutoff!')
                profile.crop()
                all_profiles.append(profile)
                fwhm[ctr] = profile.fwhm()
                rms[ctr] = profile.rms()

            mean_profile = get_mean_profile(all_profiles)
            result[zero_crossing]['representative_profile'] = mean_profile
            result['beamsizes'][(zero_crossing-1)*2] = np.nanmean(rms) * abs(cal)
            result['beamsizes_err'][(zero_crossing-1)*2] = np.nanstd(rms) * abs(cal)

        if len(result['zero_crossings']) == 2:
            result['corrected_profile'] = tilt_reconstruction2(result[1]['representative_profile'], result[2]['representative_profile'])['corrected_profile']
        else:
            result['corrected_profile'] = None

        if len(result['zero_crossings']) == 2:
            a1, b1 = result[1]['polyfit']
            a2, b2 = result[2]['polyfit']
            phase_cross_rel = (b2 - b1)/(a1 - a2)
            phase_cross_abs = phase_cross_rel + result[1]['phases_deg0'].mean()/180*np.pi
            result['phase_cross_rel'] = phase_cross_rel
            result['phase_cross_abs'] = phase_cross_abs

        cals_applied = []
        for zero_crossing in result['zero_crossings']:
            cals_applied.append(result[zero_crossing]['applied_calibration'])
        cal_applied = np.mean(np.abs(cals_applied))

        if self.unstreaked_measured:
            scan = self.scans[-1]
            data = self.data
            images = data['raw_data'][scan]['image']['data'].astype(float)
            x_axis = data['raw_data'][scan]['x_axis']['data'].astype(float)*1e-6
            y_axis = data['raw_data'][scan]['y_axis']['data'].astype(float)*1e-6

            while len(x_axis.shape) > 1:
                x_axis = x_axis[0]
            while len(y_axis.shape) > 1:
                y_axis = y_axis[0]

            axis = x_axis if streaking_direction == 'X' else y_axis
            projections = images.sum(axis=(1 if streaking_direction == 'X' else 2))
            sizes = [gaussfit.GaussFit(axis, projection).sigma for projection in projections]
            median_size = np.nanmedian(sizes)
            result['beamsizes'][1] = median_size
            result['beamsizes_err'][1] = np.nanstd(sizes)
            result['resolution'] = median_size / cal_applied

        beamsizes = result['beamsizes']
        beamsizes_err = result['beamsizes_err']
        result['beamsizes_sq_err'] = beamsizes_sq_err = 2*beamsizes*beamsizes_err
        if len(result['zero_crossings']) == 2:
            voltages = result['voltages']
            if beamsizes[1] != 0:
                popt, pcov = np.polyfit(voltages, beamsizes**2, 2, w=1/beamsizes_sq_err, cov='unscaled')
                result['parabola_popt'] = popt

                par_fit = np.poly1d(popt)
                xx = np.linspace(voltages[0], voltages[2], 100)
                yy = par_fit(xx)
                result['parabola_x'] = xx
                result['parabola_y'] = yy

                corr_rms_blen = np.sqrt(popt[0])*self.voltage/cal_applied
                corr_rms_blen_err = corr_rms_blen/(2*popt[0])*np.sqrt(pcov[0,0])
                resolution = beamsizes[1] / cal_applied
                result['resolution'] = resolution
                result['corr_rms_blen'] = corr_rms_blen
                result['corr_rms_blen_err'] = corr_rms_blen_err
        return result

