import os
import copy
import numpy as np

from . import h5_storage
from . import resolution
from . import data_loader
from . import calibration
from . import blmeas
from . import lattice
from . import beam_profile
from . import myplotstyle as ms
from . import image_analysis

espread_current_exponent = 2./3.

def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
    #power[power<0] = 0
    return power

def power_Eloss_err(slice_time, slice_current, slice_E_on, slice_E_off, slice_current_err, slice_E_on_err, slice_E_off_err, mask0):
    delta_E = slice_E_off-slice_E_on
    power = slice_current * delta_E
    #power[power<0] = 0
    err_sq_1 = (delta_E * slice_current_err)**2
    err_sq_2 = (slice_current * slice_E_on_err)**2
    err_sq_3 = (slice_current * slice_E_off_err)**2
    power_err = np.sqrt(err_sq_1+err_sq_2+err_sq_3)
    power[mask0] = 0
    power_err[mask0] = 0
    energy = np.trapz(power, slice_time)
    return {
            'time': slice_time,
            'power': power,
            'energy': energy,
            'power_err': power_err,
            }

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total=None, photon_energy_factors=1, norm_factor=None):
    power0 = slice_current**espread_current_exponent * slice_Espread_sqr_increase * photon_energy_factors
    if norm_factor is None:
        mask = np.ones_like(slice_t, dtype=bool)
        integral = np.trapz(power0[mask], slice_t[mask])
        power = power0/integral*E_total
    else:
        power = power0*norm_factor
    return power

def power_Espread_err(slice_t, slice_current, slice_Espread_on_sq, slice_Espread_off_sq, E_total, slice_current_err, slice_Espread_on_sq_err, slice_Espread_off_sq_err, photon_energy_factors=1, norm_factor=None):
    """
    Takes squared values of energy spread
    """
    exp = espread_current_exponent
    slice_Espread_sqr_increase = slice_Espread_on_sq - slice_Espread_off_sq
    power0 = slice_current**exp * slice_Espread_sqr_increase * photon_energy_factors
    power1 = power0.copy()
    power1[power0 < 0] = 0
    integral = np.trapz(power1, slice_t)

    if norm_factor is None:
        norm_factor = E_total/integral
    power = power0*norm_factor
    power0_err_1 = exp * slice_current**(exp-1) * slice_Espread_sqr_increase * photon_energy_factors * slice_current_err
    power0_err_2 = slice_current**(2/3) * photon_energy_factors * slice_Espread_off_sq_err
    power0_err_3 = slice_current**(2/3) * photon_energy_factors * slice_Espread_on_sq_err
    power0_err = np.sqrt(power0_err_1**2+power0_err_2**2+power0_err_3**2)
    power_err = power0_err*norm_factor
    energy = np.trapz(power, slice_t)

    return {
            'time': slice_t,
            'power': power,
            'power_err': power_err,
            'energy': energy,
            'norm_factor': norm_factor,
            'photon_energy_factors': photon_energy_factors,
            }

def obtain_lasing(tracker, file_or_dict_off, file_or_dict_on, lasing_options, pulse_energy, norm_factor=None):
    if type(file_or_dict_off) is dict:
        lasing_off_dict = file_or_dict_off
    else:
        lasing_off_dict = h5_storage.loadH5Recursive(file_or_dict_off)
    if type(file_or_dict_on) is dict:
        lasing_on_dict = file_or_dict_on
    else:
        lasing_on_dict = h5_storage.loadH5Recursive(file_or_dict_on)
    las_rec_images = {}

    # Allow supply of two trackers, the first for lasing off and the second for lasing on
    if hasattr(tracker, '__iter__'):
        tracker1, tracker2 = tracker
    else:
        tracker1 = tracker2 = tracker
    trackers = [tracker1, tracker2]

    for main_ctr, (data_dict, title, _tracker) in enumerate([(lasing_off_dict, 'Lasing Off', tracker1), (lasing_on_dict, 'Lasing On', tracker2)]):
        if main_ctr == 0:
            ref_y = None
        else:
            ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        if main_ctr == 1:
            profile = las_rec_images['Lasing Off'].profile
            ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        else:
            profile = None
            ref_slice_dict = None

        rec_obj = LasingReconstructionImages(title, _tracker, lasing_options, profile=profile, ref_y=ref_y)
        if trackers[main_ctr].structure.dim == 'X':
            screen_centerX, screen_centerY = _tracker.calib.screen_center, 0
        elif trackers[main_ctr].structure.dim == 'Y':
            screen_centerX, screen_centerY = 0, trackers[main_ctr].calib.screen_center
        rec_obj.add_dict(data_dict, screen_centerX=screen_centerX, screen_centerY=screen_centerY)
        rec_obj.process_data(ref_slice_dict=ref_slice_dict)
        las_rec_images[title] = rec_obj

    las_rec = LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], lasing_options, pulse_energy)
    las_rec.lasing_analysis(norm_factor=norm_factor)
    result_dict = las_rec.get_result_dict()
    outp = {
            'las_rec': las_rec,
            'result_dict': result_dict,
            'las_rec_images': las_rec_images,
            }
    return outp

def tds_obtain_lasing(blmeas_file_or_profile, tracker, file_or_dict_off, file_or_dict_on, lasing_options, pulse_energy, norm_factor=None, backward=False, blmeas_kwargs={}, blmeas_cutoff=None, camera_res=20e-6):
    if type(blmeas_file_or_profile) is str:
        blmeas_dict = blmeas.analyze_blmeas(blmeas_file_or_profile, separate_calibrations=False, **blmeas_kwargs)
        blmeas_profile = blmeas_dict['corrected_profile']
    elif type(blmeas_file_or_profile) is beam_profile.BeamProfile:
        blmeas_dict = None
        blmeas_profile = blmeas_file_or_profile
    if blmeas_cutoff:
        blmeas_profile.cutoff(blmeas_cutoff)
    blmeas_profile.center('Mean')

    if type(file_or_dict_off) is dict:
        lasing_off_dict = file_or_dict_off
    else:
        lasing_off_dict = h5_storage.loadH5Recursive(file_or_dict_off)
    if type(file_or_dict_on) is dict:
        lasing_on_dict = file_or_dict_on
    else:
        lasing_on_dict = h5_storage.loadH5Recursive(file_or_dict_on)

    # Allow supply of two trackers, the first for lasing off and the second for lasing on
    if hasattr(tracker, '__iter__'):
        tracker1, tracker2 = tracker
    else:
        tracker1 = tracker2 = tracker
    trackers = [tracker1, tracker2]

    las_rec_images = {}
    median_images = {}
    for main_ctr, (data_dict, title, _tracker) in enumerate([(lasing_off_dict, 'Lasing Off', tracker1), (lasing_on_dict, 'Lasing On', tracker2)]):

        calib0 = tracker.calib
        median_index = data_loader.get_median(lasing_off_dict['pyscan_result']['image'].astype(float).sum(axis=-2), 'mean', 'index')
        x_axis = lasing_off_dict['pyscan_result']['x_axis_m'].astype(float)
        y_axis = lasing_off_dict['pyscan_result']['y_axis_m'].astype(float)
        if tracker.structure.dim == 'X':
            x_axis = x_axis - calib0.screen_center
        if tracker.structure.dim == 'Y':
            y_axis = y_axis - calib0.screen_center
        image_arr = data_dict['pyscan_result']['image'][median_index].astype(float)
        if x_axis[0] > x_axis[1]:
            x_axis = x_axis[::-1]
            image_arr = image_arr[:,::-1]
        if y_axis[0] > y_axis[1]:
            y_axis = y_axis[::-1]
            image_arr = image_arr[::-1]
        screen_proj = image_arr.sum(axis=-2)
        median_images[title] = image_analysis.Image(image_arr, x_axis, y_axis, charge=blmeas_profile.total_charge)
        raw_screen = beam_profile.ScreenDistribution(x_axis, screen_proj, total_charge=blmeas_profile.total_charge)

        blmeas_profile.total_charge = tracker.total_charge
        for _ in range(2):
            tds_calib = calibration.tdc_calibration(tracker, blmeas_profile, raw_screen, backward=backward)
            tracker.update_calib(tds_calib['calib'])

        if main_ctr == 0:
            ref_y = None
        else:
            ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        if main_ctr == 1:
            profile = las_rec_images['Lasing Off'].profile
            ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        else:
            profile = None
            ref_slice_dict = None

        rec_obj = LasingReconstructionImages(title, _tracker, lasing_options, profile=profile, ref_y=ref_y)
        if trackers[main_ctr].structure.dim == 'X':
            screen_centerX, screen_centerY = _tracker.calib.screen_center, 0
        elif trackers[main_ctr].structure.dim == 'Y':
            screen_centerX, screen_centerY = 0, trackers[main_ctr].calib.screen_center
        rec_obj.add_dict(data_dict, screen_centerX=screen_centerX, screen_centerY=screen_centerY)
        #rec_obj.process_data(ref_slice_dict=ref_slice_dict)
        self = rec_obj

        self.convert_y()
        self.profile = blmeas_profile
        self.wake_t, self.wake_x = self.calc_wake()
        if self.lasing_options['adjust_beam_position']:
            self.get_streaker_offsets()
        self.convert_x_wake()

        self.slice_x()
        self.fit_slice()

        if ref_slice_dict is not None:
            self.ref_slice_dict = ref_slice_dict
        elif self.ref_slice_dict is None:
            self.ref_slice_dict = self.slice_dicts[0]

        if self.ref_slice_dict is None:
            raise ValueError('No ref_slice_dict defined!')
        self.interpolate_slice(self.ref_slice_dict)

        las_rec_images[title] = rec_obj

    las_rec = LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], lasing_options, pulse_energy)
    las_rec.lasing_analysis(norm_factor=norm_factor)
    result_dict = las_rec.get_result_dict()

    d1 = result_dict['images_off']['distances']
    d2 = result_dict['images_on']['distances']
    if d1 is not None and d2 is not None:
        mean_distance = np.mean(list(d1)+list(d2))
        qw = tracker.forward_options['quad_wake']
        tracker.forward_options['quad_wake'] = True
        profile2 = copy.deepcopy(profile)
        profile2.expand(0.3)
        res_dict = resolution.calc_resolution(profile, tracker.structure_gap, tracker.structure_gap/2 - mean_distance, tracker, camera_res=camera_res, dim=tracker.structure.dim.lower())
        tracker.forward_options['quad_wake'] = qw
    else:
        res_dict = None
    result_dict['resolution_dict'] = res_dict

    outp = {
            'las_rec': las_rec,
            'result_dict': result_dict,
            'las_rec_images': las_rec_images,
            'example_images': median_images,
            'tds_calib': tds_calib,
            'blmeas_dict': blmeas_dict,
            }
    return outp

def linear_obtain_lasing(file_or_dict_off, file_or_dict_on, lasing_options, pulse_energy, norm_factor=None, min_factor=None, max_factor=None):
    if type(file_or_dict_off) is dict:
        lasing_off_dict = file_or_dict_off
    else:
        lasing_off_dict = h5_storage.loadH5Recursive(file_or_dict_off)
    if type(file_or_dict_on) is dict:
        lasing_on_dict = file_or_dict_on
    else:
        lasing_on_dict = h5_storage.loadH5Recursive(file_or_dict_on)
    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        if main_ctr == 0:
            ref_y = None
            ref_slice_dict = None
            enforce_median_rms = lasing_options['adjust_linear_factor']
            enforce_rms = None
        else:
            ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
            ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
            enforce_median_rms = False
            if lasing_options['adjust_linear_factor']:
                enforce_rms = las_rec_images['Lasing Off'].median_rms
            else:
                enforce_rms = None

        rec_obj = LasingReconstructionImagesLinear(title, data_dict, lasing_options, ref_y=ref_y, ref_slice_dict=ref_slice_dict)
        rec_obj.add_dict(data_dict)
        rec_obj.convert_y()
        rec_obj.convert_x_linear(enforce_median_rms=enforce_median_rms, enforce_rms=enforce_rms, min_factor=min_factor, max_factor=max_factor)

        #rec_obj.process_data(ref_slice_dict=ref_slice_dict)
        #all_rms = [x.rms() for x in rec_obj.profiles]
        #all_mean = [x.mean() for x in rec_obj.profiles]
        #if enforce_rms is None:
        #    enforce_rms = 0
        #print(title, '%.1f %.1f %.1f %.3f %.3f' % (enforce_rms*1e15, np.mean(all_rms)*1e15, np.std(all_rms)*1e15, np.mean(all_mean)*1e15, np.std(all_mean)*1e15))

        rec_obj.slice_x()
        rec_obj.fit_slice()

        if ref_slice_dict is not None:
            rec_obj.ref_slice_dict = ref_slice_dict
        elif ref_slice_dict is None:
            rec_obj.ref_slice_dict = rec_obj.slice_dicts[0]
        rec_obj.interpolate_slice(rec_obj.ref_slice_dict)
        las_rec_images[title] = rec_obj
    las_rec = LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], lasing_options, pulse_energy)
    las_rec.lasing_analysis(norm_factor=norm_factor)
    result_dict = las_rec.get_result_dict()
    outp = {
            'las_rec': las_rec,
            'result_dict': result_dict,
            'las_rec_images': las_rec_images,
            }
    return outp

def modelfree_obtain_lasing(blmeas_file_or_profile, tracker, file_or_dict_off, file_or_dict_on, lasing_options, pulse_energy, norm_factor=None, blmeas_kwargs={}, blmeas_cutoff=None):
    if type(file_or_dict_off) is dict:
        lasing_off_dict = file_or_dict_off
    else:
        lasing_off_dict = h5_storage.loadH5Recursive(file_or_dict_off)
    if type(file_or_dict_on) is dict:
        lasing_on_dict = file_or_dict_on
    else:
        lasing_on_dict = h5_storage.loadH5Recursive(file_or_dict_on)

    if type(blmeas_file_or_profile) is str:
        blmeas_dict = blmeas.analyze_blmeas(blmeas_file_or_profile, separate_calibrations=False, **blmeas_kwargs)
        blmeas_profile = blmeas_dict['corrected_profile']
    elif type(blmeas_file_or_profile) is beam_profile.BeamProfile:
        blmeas_dict = None
        blmeas_profile = blmeas_file_or_profile
    if blmeas_cutoff:
        blmeas_profile.cutoff(blmeas_cutoff)
    blmeas_profile.center('Mean')

    las_rec_images = {}
    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        if main_ctr == 0:
            ref_y = None
        else:
            ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        if main_ctr == 1:
            profile = las_rec_images['Lasing Off'].profile
            ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        else:
            profile = None
            ref_slice_dict = None

        rec_obj = LasingReconstructionImages(title, tracker, lasing_options, profile=profile, ref_y=ref_y)
        rec_obj.profile = blmeas_profile
        if tracker.structure.dim == 'X':
            screen_centerX, screen_centerY = tracker.calib.screen_center, 0
        elif tracker.structure.dim == 'Y':
            screen_centerX, screen_centerY = 0, tracker.calib.screen_center
        rec_obj.add_dict(data_dict, screen_centerX=screen_centerX, screen_centerY=screen_centerY)
        rec_obj.process_data(ref_slice_dict=ref_slice_dict)
        las_rec_images[title] = rec_obj

    las_rec = LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], lasing_options, pulse_energy)
    las_rec.lasing_analysis(norm_factor=norm_factor)
    result_dict = las_rec.get_result_dict()
    outp = {
            'las_rec': las_rec,
            'result_dict': result_dict,
            'las_rec_images': las_rec_images,
            'blmeas_profile': blmeas_profile
            }
    return outp


class LasingReconstruction:
    def __init__(self, images_off, images_on, lasing_options, pulse_energy=None):
        if lasing_options['x_conversion'] == 'wake':
            assert images_off.profile == images_on.profile
        self.images_off = images_off
        self.images_on = images_on
        self.lasing_options = lasing_options
        self.pulse_energy = pulse_energy

        self.all_slice_dict = {}
        slice_method = self.lasing_options['slice_method']
        for images, title in [(self.images_off, 'Lasing Off'), (self.images_on, 'Lasing On')]:
            self.all_slice_dict[title] = images.generate_all_slice_dict(slice_method)

        mean_slice_dict = self.mean_slice_dict = {}
        for title in 'Lasing Off', 'Lasing On':
            mean_slice_dict[title] = all_slice_to_mean_slice_dict(self.all_slice_dict[title], cut_extremes=self.lasing_options['cut_extremes'])

        self.analysis_complete = False

    def lasing_analysis(self, photon_energy_factors=None, norm_factor=None):
        all_slice_dict = self.all_slice_dict
        mean_slice_dict = self.mean_slice_dict
        self.lasing_dict = lasing_dict = {}
        current_cutoff = self.lasing_options['current_cutoff']

        self.mean_current = mean_current = (mean_slice_dict['Lasing On']['current']['mean']+mean_slice_dict['Lasing Off']['current']['mean'])/2.
        self.current_mask = mask = np.abs(mean_current) > current_cutoff
        mean_current = mean_current[mask]
        err_current = (np.sqrt(mean_slice_dict['Lasing On']['current']['std']**2+mean_slice_dict['Lasing Off']['current']['std']**2)/2.)[mask]

        off_loss_mean = mean_slice_dict['Lasing Off']['loss']['mean'][mask]
        off_loss_err = mean_slice_dict['Lasing Off']['loss']['std'][mask]
        on_loss_mean = mean_slice_dict['Lasing On']['loss']['mean'][mask]
        on_loss_err = mean_slice_dict['Lasing On']['loss']['std'][mask]
        off_spread_mean = mean_slice_dict['Lasing Off']['spread']['mean'][mask]
        off_spread_err = mean_slice_dict['Lasing Off']['spread']['std'][mask]
        on_spread_mean = mean_slice_dict['Lasing On']['spread']['mean'][mask]
        on_spread_err = mean_slice_dict['Lasing On']['spread']['std'][mask]
        slice_time = mean_slice_dict['Lasing Off']['t']['mean'][mask]

        if photon_energy_factors is None:
            photon_energy_factors = np.ones_like(off_loss_mean)
        else:
            photon_energy_factors = photon_energy_factors

        t_lims = self.lasing_options['t_lims']
        if t_lims is not None:
            mask0 = np.logical_or(slice_time < t_lims[0], slice_time > t_lims[1])
            photon_energy_factors[mask0] = 0.
        else:
            mask0 = np.zeros_like(slice_time, bool)

        lasing_dict['time'] = slice_time
        lasing_dict['Eloss'] = power_Eloss_err(slice_time, mean_current, on_loss_mean, off_loss_mean, err_current, on_loss_err, off_loss_err, mask0)
        lasing_dict['Espread'] = power_Espread_err(slice_time, mean_current, on_spread_mean, off_spread_mean, self.pulse_energy, err_current, on_spread_err, off_spread_err, photon_energy_factors, norm_factor=norm_factor)
        lasing_dict['norm_factor'] = norm_factor = lasing_dict['Espread']['norm_factor']
        lasing_dict['photon_energy_factors'] = photon_energy_factors
        print('norm_factor', '%.3e' % norm_factor, 'pulse energy Espread', '%.3e uJ' % (lasing_dict['Espread']['energy']*1e6), 'pulse energy Eloss', '%.3e uJ' % (lasing_dict['Eloss']['energy']*1e6))

        n_images = len(all_slice_dict['Lasing On']['t'])
        all_loss = np.zeros([n_images, mask.sum()])
        all_spread = all_loss.copy()
        rms_eloss = np.zeros(n_images)
        fwhm_eloss = rms_eloss.copy()
        rms_espread = rms_eloss.copy()
        fwhm_espread = rms_eloss.copy()

        for ctr in range(n_images):
            current = all_slice_dict['Lasing On']['current'][ctr, mask]
            mask2 = np.logical_or(current < current_cutoff, mask0)
            on_loss = all_slice_dict['Lasing On']['loss'][ctr,mask]
            on_spread = all_slice_dict['Lasing On']['spread'][ctr,mask]

            loss = off_loss_mean - on_loss
            power_loss = power_Eloss(mean_current, loss)
            power_loss[mask2] = 0
            all_loss[ctr] = power_loss

            sq_increase = on_spread - off_spread_mean
            power_spread = power_Espread(slice_time, current, sq_increase, None, photon_energy_factors, norm_factor=norm_factor)
            power_spread[mask2] = 0
            all_spread[ctr] = power_spread

            for power_arr, rms_arr, fwhm_arr in [(power_loss, rms_eloss, fwhm_eloss), (power_spread, rms_espread, fwhm_espread)]:
                power_loss2 = power_arr.copy()
                power_loss2[power_loss2 < 0] = 0
                _, rms_arr[ctr] = image_analysis.calc_rms(slice_time, power_loss2)
                fwhm_arr[ctr] = image_analysis.calc_fwhm(slice_time, power_loss2)

        lasing_dict['all_Eloss'] = all_loss
        lasing_dict['all_Espread'] = all_spread
        lasing_dict['rms_Eloss'] = rms_eloss
        lasing_dict['fwhm_Eloss'] = fwhm_eloss
        lasing_dict['rms_Espread'] = rms_espread
        lasing_dict['fwhm_Espread'] = fwhm_espread

        self.analysis_complete = True

    def get_result_dict(self):
        outp = {
                'lasing_dict': self.lasing_dict,
                'all_slice_dict': self.all_slice_dict,
                'mean_slice_dict': self.mean_slice_dict,
                'mean_current': self.mean_current,
                'linear_conversion': int(self.lasing_options['x_conversion'] == 'linear'),
                'dispersion': np.array([self.images_off.dispersion, self.images_on.dispersion]),
                'lasing_options': self.lasing_options,
                }
        for key, obj in [('images_on', self.images_on), ('images_off', self.images_off)]:
            outp[key] = d = {}
            d['raw_images'] = obj.images_xy
            d['tE_images'] = obj.images_tE
            d['current_profile'] = obj.profile
            if hasattr(obj, 'tracker'):
                d['tracker_gap'] = obj.tracker.structure_gap
                d['tracker_beam_position'] = obj.tracker.beam_position
                d['tracker_distance'] = obj.tracker.structure_gap/2 - abs(obj.tracker.beam_position)
            else:
                d['tracker_gap'] = None
                d['tracker_beam_position'] = None
                d['tracker_distance'] = None
            if hasattr(obj, 'beam_positions') and obj.beam_positions is not None:
                d['distances'] = obj.tracker.structure_gap/2 - np.abs(obj.beam_positions)
                d['beam_positions'] = obj.beam_positions
                d['delta_distances'] = obj.delta_distances
                d['meas_screen_centroids'] = obj.meas_screen_centroids
            else:
                d['distances'] = None
                d['beam_positions'] = None
                d['delta_distances'] = None
                d['meas_screen_centroids'] = None
            if hasattr(obj, 'linear_factors') and obj.linear_factors is not None:
                d['linear_factors'] = obj.linear_factors
            else:
                d['linear_factors'] = None
        return outp

    def live_reconstruction(self, raw_image):

        tracker = self.images_on.tracker
        if tracker.structure.dim == 'Y':
            raw_image = np.transpose(raw_image, (1,0))

        x_axis = self.images_on.x_axis
        y_axis = self.images_on.y_axis
        charge = tracker.total_charge
        ref_profile = self.images_on.profile
        raw_image = prepare_raw_image(raw_image, self.lasing_options['subtract_quantile'], self.lasing_options['max_quantile'], self.lasing_options['max_absolute'])
        image_xy = image_analysis.Image(raw_image, x_axis, y_axis, charge)
        image_E, _ = self.images_on.convert_y_single(image_xy)

        meas_screen = beam_profile.ScreenDistribution(image_E.x_axis, image_E.image.sum(axis=-2), total_charge=charge)
        if self.lasing_options['adjust_beam_position']:
            position_dict = tracker.find_beam_position(tracker.beam_position, meas_screen, ref_profile)
            beam_position = position_dict['beam_position']
        else:
            beam_position = None
        wake_t, wake_x = self.images_on.calc_wake(beam_position=beam_position)
        cut_image = image_E.cut(wake_x.min(), wake_x.max())
        image_tE = cut_image.x_to_t(wake_x, wake_t)

        slice_dict = image_tE.fit_slice(
                rms_sigma=self.lasing_options['rms_sigma'],
                current_cutoff=self.lasing_options['current_cutoff'],
                E_lims=self.lasing_options['E_lims'],
                )
        slice_dict = interpolate_slice_dicts(self.images_on.ref_slice_dict, slice_dict)
        slice_method = self.lasing_options['slice_method']
        espread_on = slice_dict[slice_method]['sigma_sq']
        espread_off = self.mean_slice_dict['Lasing Off']['spread']['mean']
        espread_increase_sq = espread_on - espread_off
        norm_factor = self.lasing_dict['Espread']['norm_factor']
        pEspread = power_Espread(slice_dict['slice_x'], slice_dict['slice_current'], espread_increase_sq, norm_factor=norm_factor)

        outp = {
                'image_xy': image_xy,
                'image_tE': image_tE,
                'beam_position': beam_position,
                'slice_dict': slice_dict,
                'power_Espread': pEspread,
                'espread_off': np.sqrt(espread_off),
                'espread_on': np.sqrt(espread_on),
                }
        return outp


class LasingReconstructionImagesBase:
    def __init__(self, identifier, energy_eV, dispersion, total_charge, ref_y, ref_slice_dict, lasing_options):
        self.identifier = identifier
        self.dispersion = dispersion
        self.total_charge = total_charge
        self.energy_eV = energy_eV
        self.ref_y = ref_y
        self.ref_slice_dict = ref_slice_dict
        self.lasing_options = lasing_options

    def add_dict(self, data_dict, max_index=None, screen_centerX=0, screen_centerY=0):
        images = data_dict['pyscan_result']['image'].astype(np.float64)
        x_axis = data_dict['pyscan_result']['x_axis_m'].astype(np.float64)
        y_axis = data_dict['pyscan_result']['y_axis_m'].astype(np.float64)
        rotate = self.data['meta_data']['streaking_direction'] == 'Y'
        self.add_images(images, x_axis, y_axis, rotate, screen_centerX, screen_centerY, max_index)

    @property
    def ref_slice_dict(self):
        return self._ref_slice_dict

    @ref_slice_dict.setter
    def ref_slice_dict(self, ref):
        self._ref_slice_dict = ref
        if ref is None:
            self.n_slices = None
        else:
            self.n_slices = len(ref['slice_x'])

    def convert_y(self):
        self.ref_y_list = []
        self.images_E = []
        ref_y = self.ref_y
        if ref_y is None:
            mean_y_list = []
            for image in self.images_xy:
                profy = image.get_screen_dist('Y')
                mean_y_list.append(profy.gaussfit.mean)
            ref_y = np.mean(mean_y_list)
        for ctr, img in enumerate(self.images_xy):
            image_E, ref_y = img.y_to_eV(self.dispersion, self.energy_eV, ref_y=ref_y)
            self.images_E.append(image_E)
            self.ref_y_list.append(ref_y)

    def convert_y_single(self, image):
        return image.y_to_eV(self.dispersion, self.energy_eV, ref_y=self.ref_y)

    def convert_x_linear(self, factor):
        self.images_tE = []
        self.profiles = []
        for ctr, img in enumerate(self.images_E):
            new_img = img.x_to_t_linear(factor, mean_to_zero=True, current_cutoff=self.lasing_options['current_cutoff'])
            self.images_tE.append(new_img)
            new_profile = beam_profile.BeamProfile(new_img.x_axis, new_img.image.sum(axis=-2), self.energy_eV, self.total_charge)
            self.profiles.append(new_profile)

    def convert_x_modelfree(self):
        self.images_tE = []
        all_revert = []
        for ctr, img in enumerate(self.images_E):
            new_img, revert, _, _ = img.x_to_t_modelfree(self.profile, x_scale_factor=self.lasing_options['x_to_t_x_factor'], t_scale_factor=self.lasing_options['x_to_t_time_factor'])
            self.images_tE.append(new_img)
            all_revert.append(revert)
        assert all(all_revert) or not any(all_revert)

    def generate_all_slice_dict(self, slice_method):
        all_mean = np.zeros([len(self.images_tE), self.n_slices], dtype=float)
        all_sigma = all_mean.copy()
        all_x = all_mean.copy()
        all_current = all_mean.copy()
        all_chirp = all_mean.copy()

        for ctr, slice_dict in enumerate(self.slice_dicts):
            all_mean[ctr] = slice_dict[slice_method]['mean']
            all_sigma[ctr] = slice_dict[slice_method]['sigma_sq']
            all_chirp[ctr] = slice_dict[slice_method]['chirp']
            all_x[ctr] = slice_dict['slice_x']
            all_current[ctr] = slice_dict['slice_current']

        outp = {
                'loss': all_mean,
                'spread': all_sigma,
                't': all_x,
                'current': all_current,
                'chirp': all_chirp,
                }
        return outp

    def slice_x(self):
        slice_factor = self.lasing_options['slice_factor']
        if slice_factor == 1:
            self.images_sliced = self.images_tE
        else:
            self.images_sliced = []
            for n_image, image in enumerate(self.images_tE):
                n_slices = len(image.x_axis)//slice_factor
                image_sliced = image.slice_x(n_slices)
                self.images_sliced.append(image_sliced)

    def fit_slice(self):
        do_plot = self.lasing_options['plot_slice_analysis']
        save_path = self.lasing_options['plot_slice_analysis_save_path']
        self.slice_dicts = []
        old_fignums = ms.plt.get_fignums()
        if old_fignums:
            old_fignum = ms.plt.gcf().number
        else:
            old_fignum = None
        for n_image, image in enumerate(self.images_sliced):
            if self.lasing_options['x_conversion'] == 'linear':
                current_cutoff = self.lasing_options['current_cutoff']
            else:
                current_cutoff = None
            slice_method = self.lasing_options['slice_method']
            if slice_method in ('gauss', 'rms', 'cut'):
                slice_dict = image.fit_slice(rms_sigma=self.lasing_options['rms_sigma'], current_cutoff=current_cutoff, E_lims=self.lasing_options['E_lims'], do_plot=do_plot, ref_t=self.lasing_options['ref_t'])
            elif slice_method in ('full', 'cutoff'):
                slice_dict = image.fit_slice_simple(current_cutoff=current_cutoff, E_lims=self.lasing_options['E_lims'], ref_t=self.lasing_options['ref_t'])
            else:
                raise ValueError(slice_method)
            self.slice_dicts.append(slice_dict)

            if do_plot:
                new_fignums = sorted(set(ms.plt.get_fignums()) - set(old_fignums))
                for fignum in new_fignums:
                    save_path2 = os.path.expanduser(save_path)+'_%s_image_%i_%i.png' % (self.identifier.replace(' ','_'), n_image, fignum)
                    ms.plt.figure(fignum).subplots_adjust(hspace=0.5, wspace=0.3)
                    ms.plt.figure(fignum).savefig(save_path2)
                    ms.plt.close(fignum)
                    print('Saved %s' % save_path2)
                if old_fignum is not None:
                    ms.plt.figure(old_fignum)

    def interpolate_slice(self, ref):
        new_slice_dicts = []
        for slice_dict in self.slice_dicts:
            new_slice_dicts.append(interpolate_slice_dicts(ref, slice_dict))
        self.slice_dicts_old = self.slice_dicts
        self.slice_dicts = new_slice_dicts

    def plot_images(self, type_, title=None, plot_slice=True, figsize=(12,10), **kwargs):
        if title is None:
            title = self.identifier
        if type_ == 'raw':
            images = self.images_xy
        elif type_ == 'cut':
            images = self.cut_images
        elif type_ == 'tE':
            images = self.images_tE
        elif type_ == 'xE':
            images = self.images_E
        elif type_ == 'slice':
            images = self.images_sliced
        else:
            raise ValueError('%s is not a valid key.' % type_)

        sp_ctr = np.inf
        ny, nx = 3, 3
        subplot = ms.subplot_factory(ny, nx, grid=False)

        figs = []
        subplots = []
        for n_image, image in enumerate(images):
            if sp_ctr > ny*nx:
                fig = ms.figure('%s Images %s' % (title, type_), figsize=figsize)
                figs.append(fig)
                this_subplots = []
                subplots.append(this_subplots)
                sp_ctr = 1
            sp = subplot(sp_ctr, title='Image %i' % n_image, xlabel=image.xlabel, ylabel=image.ylabel)
            sp_ctr += 1
            this_subplots.append(sp)
            slice_dict = None
            if plot_slice and type_ in ('tE', 'slice') and hasattr(self, 'slice_dicts'):
                slice_dict = self.slice_dicts_old[n_image]
            image.plot_img_and_proj(sp, slice_dict=slice_dict, **kwargs)
        return figs, subplots

    def add_images(self, images, x_axis, y_axis, rotate, refx, refy, max_index=None):
        if rotate:
            x_axis, y_axis = y_axis, x_axis
            images = np.transpose(images, (0, 2, 1))

        subtract_quantile = self.lasing_options['subtract_quantile']
        subtract_absolute = self.lasing_options['subtract_absolute']
        max_absolute = self.lasing_options['max_absolute']
        max_quantile = self.lasing_options['max_quantile']
        if self.lasing_options['void_cutoff']:
            xcutoff, ycutoff = self.lasing_options['void_cutoff']
        else:
            xcutoff, ycutoff = None, None
        self.x_axis0 = x_axis
        self.x_axis = x_axis - refx
        self.y_axis0 = y_axis
        self.y_axis = y_axis - refy
        self.raw_images = images
        self.images_xy = []
        self.meas_screens = []
        rms_arr = []
        for n_image, img in enumerate(images):
            if max_index is not None and n_image >= max_index:
                break
            if subtract_absolute:
                img = img - subtract_absolute
            img = prepare_raw_image(img, subtract_quantile, max_quantile, max_absolute)
            image = image_analysis.Image(img, self.x_axis, y_axis, self.total_charge, self.energy_eV)
            image = image.cut_voids(xcutoff, ycutoff)
            self.images_xy.append(image)
            screen = beam_profile.ScreenDistribution(image.x_axis, image.image.sum(axis=-2), total_charge=self.total_charge)
            self.meas_screens.append(screen)
            rms_arr.append(screen.rms())
        self.median_meas_screen_index = np.argsort(np.array(rms_arr))[len(self.meas_screens)//2]
        self.meas_screen_centroids = np.array([abs(x.mean()) for x in self.meas_screens])

class LasingReconstructionImagesLinear(LasingReconstructionImagesBase):
    def __init__(self, identifier, filename_or_data, lasing_options, ref_y=None, ref_slice_dict=None):
        if type(filename_or_data) is dict:
            self.data = filename_or_data
        else:
            self.data = h5_storage.loadH5Recursive(filename_or_data)
        beamline = self.data['meta_data']['beamline']
        structure_name = self.data['meta_data']['structure_name']
        screen_name = self.data['meta_data']['screen_name']
        streaking_direction = self.data['meta_data']['streaking_direction']

        energy_eV = self.data['energy_eV']
        total_charge = self.data['total_charge']
        lat = lattice.get_beamline_lattice(beamline, self.data['meta_data_begin'])
        matrix = lat.get_matrix(structure_name.replace('-', '.'), screen_name.replace('-', '.'))
        dispersion = {'X': matrix[2,5], 'Y': matrix[0,5]}[streaking_direction]
        LasingReconstructionImagesBase.__init__(self, identifier, energy_eV, dispersion, total_charge, ref_y, ref_slice_dict, lasing_options)

    def convert_x_linear(self, enforce_median_rms=None, enforce_rms=None, min_factor=None, max_factor=None):
        factor = self.lasing_options['x_linear_factor']
        current_cutoff = self.lasing_options['current_cutoff']
        if enforce_rms and enforce_median_rms:
            raise ValueError('Cannot enforce both rms and median rms!')

        def convert(factors):
            self.images_tE = []
            self.profiles = []
            linear_factors = []
            for img, factor2 in zip(self.images_E, factors):
                if min_factor and abs(factor2) < min_factor:
                    continue
                if max_factor and abs(factor2) > max_factor:
                    continue
                new_img = img.x_to_t_linear(factor2, mean_to_zero=True, current_cutoff=current_cutoff)
                new_profile = new_img.get_profile()
                current = np.abs(new_profile.get_current())
                new_profile.cutoff(current_cutoff/current.max())
                new_profile.crop()
                self.images_tE.append(new_img)
                self.profiles.append(new_profile)
                linear_factors.append(factor2)
            self.linear_factors = np.array(linear_factors)

        convert(np.array([factor]*len(self.images_E)))
        rms_vals = np.array([profile.rms() for profile in self.profiles])
        index_median = np.argsort(rms_vals)[len(rms_vals)//2]
        self.profile = self.profiles[index_median]
        self.median_rms = rms_vals[index_median]

        if enforce_median_rms:
            factors = factor*self.median_rms/rms_vals
            convert(factors)

        if enforce_rms is not None:
            factors = factor*enforce_rms/rms_vals
            convert(factors)
        rms_vals = np.array([profile.rms() for profile in self.profiles])
        index_median = np.argsort(rms_vals)[len(rms_vals)//2]
        self.median_rms = rms_vals[index_median]


class LasingReconstructionImages(LasingReconstructionImagesBase):
    def __init__(self, identifier, tracker, lasing_options, profile=None, ref_slice_dict=None, ref_y=None):
        self.tracker = tracker
        self.profile = profile
        self.profiles = None
        self.gap = self.tracker.structure_gap

        self.do_recon_plot = False
        self.beam_positions = None
        self.index_median = None
        self.delta_distances = None
        LasingReconstructionImagesBase.__init__(self, identifier, tracker.energy_eV, tracker.disp, tracker.total_charge, ref_y, ref_slice_dict, lasing_options)

    def add_file(self, filename):
        data_dict = h5_storage.loadH5Recursive(filename)
        self.add_dict(data_dict)

    def add_dict(self, data_dict, max_index=None, screen_centerX=0, screen_centerY=0):
        images = data_dict['pyscan_result']['image'].astype(np.float64)
        x_axis = data_dict['pyscan_result']['x_axis_m'].astype(np.float64)
        y_axis = data_dict['pyscan_result']['y_axis_m'].astype(np.float64)
        rotate = self.tracker.structure.dim == 'Y'
        self.add_images(images, x_axis, y_axis, rotate, screen_centerX, screen_centerY, max_index)

    def get_current_profiles(self):
        self.profiles = []
        for meas_screen in self.meas_screens:
            gd = self.tracker.reconstruct_profile_Gauss(meas_screen)
            self.profiles.append(gd['reconstructed_profile'])

        for p in self.profiles:
            p._xx -= p._xx.min()

    def set_profile(self):
        rms = [x.rms() for x in self.profiles]
        self.index_median = np.argsort(rms)[len(rms)//2]
        self.profile = self.profiles[self.index_median]

    def get_streaker_offsets(self):
        beam_positions = []
        position_dicts = []
        for meas_screen in self.meas_screens:
            if self.lasing_options['adjust_beam_position_backward']:
                position_dict = self.tracker.find_beam_position_backward(self.tracker.beam_position, meas_screen, self.profile)
            else:
                position_dict = self.tracker.find_beam_position(self.tracker.beam_position, meas_screen, self.profile)
            position_dicts.append(position_dict)
            beam_positions.append(position_dict['beam_position'])
        self.beam_positions = np.array(beam_positions)
        self.delta_distances = self.beam_positions - self.tracker.beam_position
        self.average_distance = self.gap/2. - abs(self.beam_positions.mean())
        return position_dicts

    def calc_wake(self, beam_position=None, profile=None):
        r12 = self.tracker.r12
        if profile is None:
            profile = self.profile
        profile2 = copy.deepcopy(self.profile)
        profile2.expand(0.3)
        wake_dict = self.tracker.calc_wake(profile2, 'Dipole', force_beam_position=beam_position)
        wake_t = wake_dict['wake_time']
        wake_x = wake_dict['wake_potential'] / self.tracker.energy_eV * r12
        return wake_t, wake_x

    def convert_x_wake(self):
        self.images_tE = []
        self.cut_images = []
        #x_min, x_max = self.wake_x.min(), self.wake_x.max()
        for ctr, img in enumerate(self.images_E):
            img_cut = None
            if self.lasing_options['self_consistent_profile']:
                if self.profiles is None:
                    self.get_current_profiles()
                wake_t, wake_x = self.calc_wake(profile=self.profiles[ctr])
                #img_cut = img.cut(wake_x.min(), wake_x.max())
                img_tE = img.x_to_t(wake_x, wake_t, debug=False, size_factor=self.lasing_options['x_to_t_time_factor'])
            else:
                if self.beam_positions is None:
                    #img_cut = img.cut(x_min, x_max)
                    img_tE = img.x_to_t(self.wake_x, self.wake_t, debug=False)
                else:
                    wake_t, wake_x = self.calc_wake(self.beam_positions[ctr])
                    #img_cut = img.cut(wake_x.min(), wake_x.max())
                    img_tE = img.x_to_t(wake_x, wake_t, debug=False, current_profile=self.profile)
            self.images_tE.append(img_tE)
            self.cut_images.append(img_cut)

    def process_data(self, ref_slice_dict=None, slice_fit=True):
        self.convert_y()
        x_conversion = self.lasing_options['x_conversion']
        if x_conversion == 'wake':
            if self.profile is None:
                self.get_current_profiles()
                self.set_profile()
            self.wake_t, self.wake_x = self.calc_wake()
            if self.lasing_options['adjust_beam_position']:
                self.get_streaker_offsets()
            self.convert_x_wake()
        elif x_conversion  == 'linear':
            self.convert_x_linear(self.lasing_options['x_linear_factor'])
            if self.profile is None:
                self.set_profile()
        elif x_conversion == 'modelfree':
            self.convert_x_modelfree()
        else:
            raise ValueError(x_conversion)

        if slice_fit:
            self.slice_x()
            self.fit_slice()

            if ref_slice_dict is not None:
                self.ref_slice_dict = ref_slice_dict
            elif self.ref_slice_dict is None:
                self.ref_slice_dict = self.slice_dicts[0]

            if self.ref_slice_dict is None:
                raise ValueError('No ref_slice_dict defined!')
            self.interpolate_slice(self.ref_slice_dict)


def interpolate_slice_dicts(ref, alter):
    new_dict = {}
    xx_ref = ref['slice_x']
    xx_alter = alter['slice_x']
    for key, arr in alter.items():
        if key in ('E_lims', 'y_axis_Elim'):
            new_dict[key] = arr
        elif key == 'slice_x':
            new_dict[key] = xx_ref
        elif type(arr) is np.ndarray:
            new_dict[key] = np.interp(xx_ref, xx_alter, arr)
        elif type(arr) is dict:
            new_dict[key] = {}
            for key2, arr2 in arr.items():
                if type(arr2) is np.ndarray:
                    new_dict[key][key2] = np.interp(xx_ref, xx_alter, arr2)
    return new_dict

def subtract_long_wake(all_slice_dict, tracker, profile):
    long_wake = profile.calc_wake(tracker.structure, tracker.structure_gap, tracker.beam_position, 'Longitudinal')
    long_wake_interp = np.interp(all_slice_dict['t'][0], long_wake['wake_time'], long_wake['wake_potential'])
    loss2 = all_slice_dict['loss'] - long_wake_interp
    chirp2 = np.zeros_like(loss2)
    chirp2[:,:-1] = np.diff(loss2)/np.diff(all_slice_dict['t'])

    outp = copy.deepcopy(all_slice_dict)
    outp['loss'] = loss2
    outp['chirp'] = chirp2

    return {
            'all_slice_dict': outp,
            'long_wake_dict': long_wake,
            }

def all_slice_to_mean_slice_dict(all_slice_dict, cut_extremes=None):
    mean_slice_dict = {}
    for key, arr in all_slice_dict.items():
        if cut_extremes is not None:
            argsort = np.argsort(arr, axis=0)
            arr2 = np.take_along_axis(arr, argsort[cut_extremes:-cut_extremes], axis=0)
        else:
            arr2 = arr
        mean_slice_dict[key] = {
                'mean': np.nanmean(arr2, axis=0),
                'std': np.nanstd(arr2, axis=0),
                }
    return mean_slice_dict

def prepare_raw_image(img, subtract_quantile, max_quantile, max_absolute):
    if subtract_quantile is not None:
        img = img - np.quantile(img, subtract_quantile)
    if max_quantile is not None:
        img = img.clip(0, np.quantile(img, max_quantile))
    if max_absolute is not None:
        img = img.clip(0, max_absolute)
    img = img.clip(0, None)
    return img

