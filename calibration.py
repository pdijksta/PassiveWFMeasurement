import itertools
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from . import data_loader
from . import tracking
from . import beam_profile
from . import config
from . import h5_storage
from . import myplotstyle as ms
from .logMsg import LogMsgBase


class StructureCalibration:
    def __init__(self, structure_name, screen_center, delta_gap, structure_position0):
        self.structure_name = structure_name
        self.screen_center = screen_center
        self.delta_gap = delta_gap
        self.structure_position0 = structure_position0

    def __str__(self):
        return 'Structure %s: Screen center %.1f um; Structure position0 %i um; Delta gap %i um' % (self.structure_name, round(self.screen_center*1e6), round(self.structure_position0*1e6), round(self.delta_gap*1e6))

    __repr__ = __str__

    def gap_and_beam_position_from_meta(self, meta_data):
        gap0 = meta_data[self.structure_name+':GAP']*1e-3
        structure_position = meta_data[self.structure_name+':CENTER']*1e-3
        return self.gap_and_beam_position_from_gap0(gap0, structure_position)

    def gap_and_beam_position_from_gap0(self, gap0, structure_position):
        gap = gap0 + self.delta_gap
        beam_position = -(structure_position - self.structure_position0)
        distance = gap/2. - abs(beam_position)
        plate_positions = [-gap/2. + self.structure_position0, gap/2. + self.structure_position0]
        if distance < 0:
            raise ValueError('Distance between beam and plate is negative: %.3e' % distance)
        return {
                'gap0': gap0,
                'gap': gap,
                'structure_position': structure_position,
                'structure_position0': self.structure_position0,
                'beam_position': beam_position,
                'distance': distance,
                'plate_positions': plate_positions,
                }

    def to_dict_custom(self):
        return {
                'structure_name': self.structure_name,
                'screen_center': self.screen_center,
                'delta_gap': self.delta_gap,
                'structure_position0': self.structure_position0,
                }

    @staticmethod
    def from_dict(dict_):
        return StructureCalibration(**dict_)

    def copy(self):
        return self.from_dict(self.to_dict_custom())


class MeasScreens:
    def __init__(self, raw_images, plot_list_x, y_axis_list, meas_screens, beam_positions, raw_positions, streaking_factors):
        self.raw_images = raw_images
        self.plot_list_x = plot_list_x
        self.y_axis_list = y_axis_list
        self.meas_screens = meas_screens
        self.beam_positions = beam_positions
        self.raw_positions = raw_positions
        self.streaking_factors = streaking_factors

    def plot(self, plot_handles=None, plot0=True):
        if plot_handles is None:
            ms.figure('Structure calibration measured screens distributions')
            subplot = ms.subplot_factory(2,2)
            sp_ctr = 1
            sp_pos = subplot(sp_ctr, title='Raw structure position > 0', xlabel='x (mm)', ylabel=config.rho_label)
            sp_ctr += 1
            sp_neg = subplot(sp_ctr, title='Raw structure position < 0', xlabel='x (mm)', ylabel=config.rho_label)
        else:
            sp_pos, sp_neg = plot_handles
        for beam_pos, raw_pos, screen in zip(self.beam_positions, self.raw_positions, self.meas_screens):
            if raw_pos == 0:
                if plot0:
                    screen.plot_standard(sp_pos, color='black')
                    screen.plot_standard(sp_neg, color='black')
            else:
                # Inverted sign!
                if raw_pos > 0:
                    sp = sp_pos
                elif raw_pos < 0:
                    sp = sp_neg
                screen.plot_standard(sp, label='%.3f' % (beam_pos*1e3))
        sp_pos.legend(title='Beam position (mm)')
        sp_neg.legend(title='Beam position (mm)')
        return (sp_pos, sp_neg)

class StructureCalibrator(LogMsgBase):

    def __init__(self, tracker, structure_calib_options, file_or_dict=None, logger=None):
        self.logger = logger
        self.tracker = tracker
        self.structure_name = tracker.structure_name
        self.structure_calib_options = structure_calib_options
        self.structure_dim = tracker.structure.dim
        self.gap0 = tracker.structure_gap
        self.beamline = tracker.beamline
        self.total_charge = tracker.total_charge

        self.raw_struct_positions = []
        self.screen_center_arr = []
        self.centroids = []
        self.centroids_std = []
        self.all_rms = []
        self.all_centroids = []
        self.rms = []
        self.rms_std = []
        self.images = []
        self.sim_screen_dict = {}
        self.sim_screens = None
        self.plot_list_x = []
        self.plot_list_y = []
        self.plot_list_image = []
        self.y_axis_list = []
        self.raw_data = None
        self.meas_screens = None

        if file_or_dict is not None:
            self.add_file(file_or_dict)

        self.fit_dicts = {}
        self.logMsg('Structure calibrator initialized')

    def init_meas_screens(self):
        streaking_factors = []
        meas_screens = []
        index0 = np.argwhere(self.raw_struct_positions == 0).squeeze()
        shape = self.tracker.forward_options['len_screen']
        cutoff = self.tracker.forward_options['screen_cutoff']

        for x, y in zip(self.plot_list_x, self.plot_list_y):
            meas_screen = beam_profile.ScreenDistribution(x, y, total_charge=self.total_charge)
            meas_screen.aggressive_cutoff(cutoff)
            meas_screen.crop()
            meas_screen.reshape(shape)
            meas_screens.append(meas_screen)
            streaking_factors.append(meas_screen.rms())

        if index0:
            bs0 = streaking_factors[index0]
        else:
            bs0 = 1
        streaking_factors = np.array(streaking_factors)/bs0

        beam_positions = -(self.raw_struct_positions - np.mean(self.screen_center_arr))

        self.meas_screens = MeasScreens(self.plot_list_image, self.plot_list_x, self.y_axis_list, meas_screens, beam_positions, self.raw_struct_positions, streaking_factors)

    def get_meas_screens(self):
        if self.meas_screens is None:
            self.init_meas_screens()
        return self.meas_screens

    def add_data(self, raw_struct_positions, images, x_axis, y_axis):
        if self.structure_dim == 'Y':
            images = np.transpose(images, axes=(0,1,3,2))
            x_axis, y_axis = y_axis, x_axis

        proj_cutoff = self.structure_calib_options['proj_cutoff']

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            images = images[...,::-1]
        n_images = images.shape[1]
        centroids = np.zeros([len(raw_struct_positions), n_images])
        rms = np.zeros_like(centroids)

        images2 = np.zeros_like(images, dtype=np.float64)
        for n_o, n_i in itertools.product(range(len(raw_struct_positions)), range(n_images)):
            im = images[n_o,n_i].astype(np.float64)
            im = im - np.median(im)
            im = np.clip(im, 0, np.inf)
            images2[n_o,n_i] = im
        #import pdb; pdb.set_trace()
        images = images2
        proj_x = images.sum(axis=-2, dtype=np.float64)

        where0 = np.argwhere(raw_struct_positions == 0).squeeze()
        #assert where0.size == 1

        plot_list_y = []
        plot_list_image = []
        self.median_indices = np.zeros(len(raw_struct_positions), int)
        for n_o in range(len(raw_struct_positions)):
            for n_i in range(n_images):
                proj = proj_x[n_o,n_i]
                if np.all(proj == 0):
                    centroids[n_o,n_i] = np.nan
                    rms[n_o,n_i] = np.nan
                else:
                    #proj = proj - np.median(proj) #WTF was this??
                    #import pdb; pdb.set_trace()
                    proj[proj<proj.max()*proj_cutoff] = 0
                    centroids[n_o,n_i] = cc = np.sum(proj*x_axis) / np.sum(proj)
                    rms[n_o, n_i] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
            median_proj_index = data_loader.get_median(proj_x[n_o,:], 'mean', 'index')
            median_proj = proj_x[n_o, median_proj_index]
            plot_list_image.append(images[n_o, median_proj_index])
            plot_list_y.append(median_proj)
            self.median_indices[n_o] = median_proj_index
        centroid_mean = np.nanmean(centroids, axis=1)
        if where0.size == 1:
            screen_center = centroid_mean[where0]
        else:
            screen_center = 0
        centroid_mean -= screen_center
        centroids -= screen_center
        screen_center_arr = np.array([screen_center]*len(raw_struct_positions), float)
        centroid_std = np.nanstd(centroids, axis=1)
        rms_mean = np.nanmean(rms, axis=1)
        rms_std = np.nanstd(rms, axis=1)

        if 0 in self.raw_struct_positions:
            mask = raw_struct_positions != 0
        else:
            mask = np.ones_like(raw_struct_positions, dtype=bool)

        new_offsets = np.concatenate([self.raw_struct_positions, raw_struct_positions[mask]])
        sort = np.argsort(new_offsets)
        self.raw_struct_positions = new_offsets[sort]
        self.centroids = np.concatenate([self.centroids, centroid_mean[mask]])[sort]
        self.centroids_std = np.concatenate([self.centroids_std, centroid_std[mask]])[sort]
        self.rms = np.concatenate([self.rms, rms_mean[mask]])[sort]
        self.rms_std = np.concatenate([self.rms_std, rms_std[mask]])[sort]
        self.screen_center_arr = np.concatenate([self.screen_center_arr, screen_center_arr[mask]])[sort]

        plot_list_x = [x_axis - screen_center] * len(plot_list_y)
        y_axis_list = self.y_axis_list + [y_axis] * len(plot_list_y)
        new_plot_list_x = self.plot_list_x + plot_list_x
        new_plot_list_y = self.plot_list_y + plot_list_y
        new_plot_list_image = self.plot_list_image + plot_list_image
        new_images = self.images + [x for x in images]
        new_all_centroids = self.all_centroids + [x for x in centroids]
        new_all_rms = self.all_centroids + [x for x in rms]

        self.y_axis_list = []
        self.plot_list_x = []
        self.plot_list_y = []
        self.plot_list_image = []
        self.images = []
        self.all_rms = []
        self.all_centroids = []
        for new_index in sort:
            self.plot_list_x.append(new_plot_list_x[new_index])
            self.plot_list_y.append(new_plot_list_y[new_index])
            self.plot_list_image.append(new_plot_list_image[new_index])
            self.y_axis_list.append(y_axis_list[new_index])
            self.images.append(new_images[new_index])
            self.all_rms.append(new_all_rms[new_index])
            self.all_centroids.append(new_all_centroids[new_index])

    def add_file(self, filename_or_dict):
        if type(filename_or_dict) is dict:
            data_dict = filename_or_dict
        elif type(filename_or_dict) is str:
            data_dict = h5_storage.loadH5Recursive(filename_or_dict)
        else:
            raise ValueError(type(filename_or_dict))

        if 'raw_data' in data_dict:
            data_dict = data_dict['raw_data']
        if 'meta_data_begin' in data_dict:
            self.meta_data = data_dict['meta_data_begin']
        result_dict = data_dict['pyscan_result']
        images = result_dict['image'].squeeze()
        if 'x_axis_m' in result_dict:
            x_axis = result_dict['x_axis_m']
            y_axis = result_dict['y_axis_m']
        elif 'x_axis' in result_dict:
            x_axis = result_dict['x_axis']
            y_axis = result_dict['y_axis']
        else:
            print(result_dict.keys())
            raise KeyError

        raw_struct_positions = data_dict['streaker_offsets'].squeeze()
        self.add_data(raw_struct_positions, images, x_axis, y_axis)
        self.raw_data = data_dict

    def fit_type(self, type_):

        def beamsize_fit_func(raw_struct_positions, structure_position0, strength, order, semigap, const):
            sq0 = const**2
            c1 = np.abs((raw_struct_positions-structure_position0+semigap))**(-order*2)
            c2 = np.abs((raw_struct_positions-structure_position0-semigap))**(-order*2)
            if strength > 0:
                sq_add = strength**2 * (c1+c2)
            else:
                sq_add = np.zeros(len(raw_struct_positions))
            output = np.sqrt(sq0 + sq_add)
            return output

        def centroid_fit_func(raw_struct_positions, structure_position0, strength, order, semigap, const):
            c1 = np.abs((raw_struct_positions-structure_position0+semigap))**(-order)
            c2 = np.abs((raw_struct_positions-structure_position0-semigap))**(-order)
            return const + (c1 - c2)*strength

        fit_order = self.structure_calib_options['fit_order']
        fit_gap = self.structure_calib_options['fit_gap']

        raw_struct_positions = self.raw_struct_positions
        if len(raw_struct_positions) == 0:
            raise ValueError('No data!')
        semigap = self.gap0/2.
        where0 = np.argwhere(raw_struct_positions == 0).squeeze()

        if type_ == 'beamsize':
            yy_mean = self.rms
            yy_std = self.rms_std
            fit_func0 = beamsize_fit_func
            order0 = self.structure_calib_options['order_rms']
        elif type_ == 'centroid':
            yy_mean = self.centroids
            yy_std = self.centroids_std
            fit_func0 = centroid_fit_func
            order0 = self.structure_calib_options['order_centroid']

        const0 = yy_mean[where0]
        offset0 = (raw_struct_positions[0] + raw_struct_positions[-1])/2

        if type_ == 'centroid':
            s0_arr = (yy_mean-const0)/(np.abs((raw_struct_positions-offset0-semigap))**(-order0) - np.abs((raw_struct_positions-offset0+semigap))**(-order0))
        elif type_ == 'beamsize':
            s0_arr = (yy_mean-const0)/(np.abs((raw_struct_positions-offset0-semigap))**(-order0) + np.abs((raw_struct_positions-offset0+semigap))**(-order0))
        s0 = (s0_arr[0] + s0_arr[-1])/2
        p0 = [offset0, s0]
        if fit_order:
            p0.append(order0)
        if fit_gap:
            p0.append(semigap)

        def fit_func(*args):
            args = list(args)
            if fit_order:
                if fit_gap:
                    output = fit_func0(*args, const0)
                else:
                    output = fit_func0(*args, semigap, const0)
            else:
                if fit_gap:
                    output = fit_func0(*args[:-1], order0, args[-1], const0)
                else:
                    output = fit_func0(*args, order0, semigap, const0)
            return output

        try:
            p_opt, p_cov = curve_fit(fit_func, raw_struct_positions, yy_mean, p0, sigma=yy_std)
        except RuntimeError:
            print('Streaker calibration type %s did not converge' % type_)
            p_opt = p0

        structure_position0 = p_opt[0]
        if fit_gap:
            gap_fit = p_opt[-1]*2
        else:
            gap_fit = abs(semigap*2)
        if fit_order:
            order_fit = p_opt[2]
        else:
            order_fit = order0

        xx_fit = np.linspace(raw_struct_positions.min(), raw_struct_positions.max(), int(1e3))
        reconstruction = fit_func(xx_fit, *p_opt)
        initial_guess = fit_func(xx_fit, *p0)

        screen_center = np.mean(self.screen_center_arr)
        delta_gap = gap_fit - self.tracker.structure_gap0

        calibration = StructureCalibration(self.structure_name, screen_center, delta_gap, structure_position0)

        fit_dict = {
                'reconstruction': reconstruction,
                'initial_guess': initial_guess,
                'structure_position0': structure_position0,
                'gap_fit': gap_fit,
                'order_fit': order_fit,
                'p_opt': p_opt,
                'p0': p0,
                'xx_fit': xx_fit,
                'screen_rms0': const0,
                'screen_center_arr': self.screen_center_arr,
                'screen_center': screen_center,
                'calibration': calibration,
                'raw_struct_positions': self.raw_struct_positions,
                'meas_screens': self.get_meas_screens().meas_screens,
                'centroids': self.centroids,
                'centroids_std': self.centroids_std,
                'rms': self.rms,
                'rms_std': self.rms_std,
                }
        self.fit_dicts[type_] = fit_dict
        self.logMsg('structure_position0 and gap calibrated as %i um, %i um with method %s' % (round(structure_position0*1e6), round(gap_fit*1e6), type_))
        return fit_dict

    def fit(self):
        """
        Returns: fit_dict for centroid, fit_dict for beamsize
        """
        a = self.fit_type('centroid')
        b = self.fit_type('beamsize')
        return a, b

    def forward_propagate(self, blmeas_profile, force_gap=None, force_structure_offset=None, use_n_positions=None):
        sim_screens = []
        forward_dicts = []
        beam_positions = []
        tracker = self.tracker
        if force_structure_offset is None:
            structure_position0 = tracker.calib.structure_position0
        if force_gap is None:
            gap = tracker.structure_gap
        else:
            gap = force_gap

        beam = tracker.gen_beam(blmeas_profile)

        if use_n_positions is None:
            n_positions = range(len(self.raw_struct_positions))
        else:
            n_positions = use_n_positions

        for n_position in n_positions:
            raw_position = self.raw_struct_positions[n_position]
            beam_positions.append(-(raw_position-structure_position0))
            forward_dict = tracker.forward_propagate_forced(gap, beam_positions[-1], beam)
            forward_dicts.append(forward_dict)
            sim_screen = forward_dict['screen']
            sim_screens.append(sim_screen)
            #print(gap*1e3, 'mm', (gap/2-abs(beam_positions[-1]))*1e6, 'um')

        self.blmeas_profile = blmeas_profile
        self.sim_screen_dict[(gap, structure_position0)] = sim_screens
        self.sim_screens = sim_screens
        output = {
                'blmeas_profile': blmeas_profile,
                'sim_screens': sim_screens,
                'forward_dicts': forward_dicts,
                'beam_positions': beam_positions,
                }
        return output

    def reconstruct_current(self, gap, structure_position0, use_n_positions=None, plot_details=False):
        gauss_dicts = []
        beam_position_list = []
        if use_n_positions is None:
            n_positions = range(len(self.raw_struct_positions))
        else:
            n_positions = use_n_positions
        meas_screens0 = self.get_meas_screens().meas_screens
        for n_position in n_positions:
            position = self.raw_struct_positions[n_position]
            if position == 0:
                continue
            beam_position = -(position-structure_position0)
            if abs(beam_position) > gap/2.:
                raise ValueError(beam_position, gap/2.)
            beam_position_list.append(beam_position)
            meas_screen0 = meas_screens0[n_position]
            gauss_dict = self.tracker.reconstruct_profile_Gauss_forced(gap, beam_position, meas_screen0, plot_details=plot_details)
            gauss_dicts.append(gauss_dict)
        self.gauss_dicts = gauss_dicts
        return np.array(beam_position_list), gauss_dicts

    def calibrate_gap_and_struct_position(self, use_n_positions=None):
        n_positions = len(use_n_positions) if use_n_positions is not None else len(self.raw_struct_positions)-1
        delta_gap_range = self.structure_calib_options['delta_gap_range']
        delta_structure0_range = self.structure_calib_options['delta_structure0_range']
        delta_gap_scan_range = np.linspace(delta_gap_range.min() - delta_structure0_range.min()*2, delta_gap_range.max() + delta_structure0_range.max()*2, self.structure_calib_options['delta_gap_scan_n'])
        gap0 = self.tracker.structure_gap
        structure_position0 = self.tracker.structure_position0
        old_calibration = self.tracker.calib

        distance_rms_arr = np.zeros([n_positions, len(delta_gap_scan_range), 2])

        for n_delta, delta_gap in enumerate(delta_gap_scan_range):
            gap = gap0 + delta_gap
            beam_position_list, gauss_dicts = self.reconstruct_current(gap, structure_position0, use_n_positions=use_n_positions)
            rms_arr = np.array([x['reconstructed_profile'].rms() for x in gauss_dicts])
            distance_arr = gap/2. - np.abs(beam_position_list)
            distance_rms_arr[:,n_delta,0] = distance_arr
            distance_rms_arr[:,n_delta,1] = rms_arr

        beam_positions = np.array(beam_position_list)
        distances0 = gap0/2. - np.abs(beam_positions)
        mask_pos, mask_neg = beam_positions > 0, beam_positions < 0

        distance_rms_dict = {}
        for n in range(len(distance_rms_arr)):
            distance_arr = distance_rms_arr[n,:,0]
            rms_arr = distance_rms_arr[n,:,1]
            distance_plot = distance_arr - distance_arr.min()
            sort = np.argsort(distance_plot)
            distance_rms_dict[n] = [distance_arr[sort], rms_arr[sort]]

        def get_fit_param(delta_structure0, delta_gap):
            new_distances = np.zeros_like(distances0)
            new_distances[mask_pos] = distances0[mask_pos] - delta_structure0 + delta_gap/2.
            new_distances[mask_neg] = distances0[mask_neg] + delta_structure0 + delta_gap/2.
            new_rms_list = []
            for n, new_distance in enumerate(new_distances):
                distance_arr = distance_rms_dict[n][0]
                rms_arr = distance_rms_dict[n][1]
                if distance_arr[1] < distance_arr[0]:
                    distance_arr = distance_arr[::-1]
                    rms_arr = rms_arr[::-1]

                new_rms = interp1d(distance_arr, rms_arr, fill_value='extrapolate')(new_distance)
                new_rms_list.append(new_rms)
            new_rms_arr = np.array(new_rms_list)
            fit = np.poly1d(np.polyfit(new_distances, new_rms_arr, 1))
            fit_pos = np.poly1d(np.polyfit(new_distances[mask_pos], new_rms_arr[mask_pos], 1))
            fit_neg = np.poly1d(np.polyfit(new_distances[mask_neg], new_rms_arr[mask_neg], 1))
            outp = {
                    'fit': fit,
                    'fit_pos': fit_pos,
                    'fit_neg': fit_neg,
                    'new_distances': new_distances,
                    'new_rms': new_rms_arr,
                    'new_rms_average': np.mean(new_rms_arr),
                    }
            return outp

        fit_coefficients = np.zeros([len(delta_structure0_range), len(delta_gap_range)])
        mean_rms_arr = fit_coefficients.copy()
        mean_rms_pos = fit_coefficients.copy()
        mean_rms_neg = fit_coefficients.copy()
        fc_pos = fit_coefficients.copy()
        fc_neg = fit_coefficients.copy()

        all_fit_dicts = {}

        for n1, delta_structure0 in enumerate(delta_structure0_range):
            all_fit_dicts[n1] = {}
            for n2, delta_gap in enumerate(delta_gap_range):
                fit_dict = get_fit_param(delta_structure0, delta_gap)
                fit_coefficients[n1, n2] = fit_dict['fit'][1] / np.mean(fit_dict['new_rms'])
                fc_pos[n1, n2] = fit_dict['fit_pos'][1] / np.mean(fit_dict['new_rms'])
                fc_neg[n1, n2] = fit_dict['fit_neg'][1] / np.mean(fit_dict['new_rms'])
                mean_rms_arr[n1, n2] = np.mean(fit_dict['new_rms'])
                mean_rms_pos[n1, n2] = np.mean(fit_dict['new_rms'][mask_pos])
                mean_rms_neg[n1, n2] = np.mean(fit_dict['new_rms'][mask_neg])
                all_fit_dicts[n1][n2] = fit_dict

        def normalize(arr):
            return np.abs(arr)/np.max(np.abs(arr))

        fit_coefficients2 = normalize(fit_coefficients)
        fc_pos2 = normalize(fc_pos)
        fc_neg2 = normalize(fc_neg)

        diff_sides = np.abs(mean_rms_pos - mean_rms_neg)
        diff_sides /= diff_sides.max()

        combined_target = diff_sides**2 + fit_coefficients2**2 + fc_neg2**2 + fc_pos2**2
        combined_target /= combined_target.max()

        argmin = np.argwhere(combined_target == np.nanmin(combined_target))[0]
        new_structure_position0 = self.tracker.calib.structure_position0 + delta_structure0_range[argmin[0]]
        delta_gap = delta_gap_range[argmin[1]]
        new_gap = gap0 + delta_gap
        fit_dict = get_fit_param(structure_position0, delta_gap)
        new_distances = fit_dict['new_distances']
        new_rms = fit_dict['new_rms']
        new_rms_average = fit_dict['new_rms_average']

        calib = StructureCalibration(self.structure_name, np.mean(self.screen_center_arr), delta_gap+old_calibration.delta_gap, new_structure_position0)

        outp = {
                'diff_sides': diff_sides,
                'new_distances': new_distances,
                'new_rms': new_rms,
                'new_rms_average': new_rms_average,
                'fit_coefficients': fit_coefficients,
                'fit_coefficients2': fit_coefficients2,
                'distance_rms_arr': distance_rms_arr,
                'beam_positions': beam_positions,
                'delta_gap_range': delta_gap_range,
                'delta_structure0_range': delta_structure0_range,
                'delta_gap_scan_range': delta_gap_scan_range,
                'mean_rms': mean_rms_arr,
                'best_index': argmin,
                'all_fit_dicts': all_fit_dicts,
                'combined_target': combined_target,
                'calibration': calib,
                'old_calibration': old_calibration,
                'new_gap': new_gap,
                }
        self.logMsg('Streaker position and gap reconstructed as %i um %i um' % (round(calib.structure_position0*1e6), round(new_gap*1e6)))
        return outp


    def get_n_positions(self, gap, max_distance, min_distance=0):
        distances = gap/2. - np.abs(self.raw_struct_positions - self.fit_dicts['centroid']['structure_position0'])
        index_arr = np.arange(len(self.raw_struct_positions))
        return index_arr[(distances <= max_distance) * (distances >=min_distance)]

def tdc_calibration(tracker, blmeas_profile, meas_screen_raw, output_beam=False):
    position0 = tracker.beam_position
    result_dict = tracker.find_beam_position(position0, meas_screen_raw, blmeas_profile)
    delta_position = result_dict['delta_position']
    meas_screen = tracker.prepare_screen(meas_screen_raw)['screen']
    force_pos_old = tracker.force_beam_position

    try:
        tracker.force_beam_position = tracker.beam_position + delta_position
        back_dict = tracker.backward_propagate(meas_screen, blmeas_profile)
        beam = tracker.gen_beam(back_dict['profile'])
        forward_dict = tracker.forward_propagate(beam)
    finally:
        tracker.force_beam_position = force_pos_old

    screen_center = tracker.calib.screen_center
    delta_gap = tracker.calib.delta_gap
    structure_position0 = tracker.calib.structure_position0
    new_structure_center0 = structure_position0 + delta_position
    new_calib = StructureCalibration(tracker.structure_name, screen_center, delta_gap, new_structure_center0)
    outp = {
            'calib': new_calib,
            'old_calib': tracker.calib,
            'blmeas_profile': blmeas_profile,
            'meas_screen_raw': meas_screen_raw,
            'tdc_forward_screen': result_dict['sim_screen'],
            'find_beam_position_result': result_dict,
            'backward_dict': back_dict,
            'forward_dict': forward_dict,
            }
    if output_beam:
        outp['beam'] = beam
    return outp

class CentroidCalibrator(LogMsgBase):
    def __init__(self, multi_position_data, calib, centroid_calibrator_options, logger=None):
        self.logger = logger
        self.centroid_calibrator_options = centroid_calibrator_options
        self.data = multi_position_data
        structure = self.data.structure
        beamline = data_loader.beamline_from_structure(structure)
        meta_dict = data_loader.dlsp_to_meta_dict(self.data.single_position_data[0])
        self.tracker = tracking.get_default_tracker(beamline, structure, meta_dict, calib, self.data.screen_name, meta_data_type=1)
        self.current_calib = calib

    def forward_propagate_all(self, beamProfile):
        return self.forward_propagate_any(beamProfile, self.data.positions)

    def forward_propagate_any(self, beamProfile, positions):
        forward_dicts = []
        centroids = np.zeros(len(positions))
        beam_sizes = centroids.copy()
        for ctr, position in enumerate(positions):
            beam = self.tracker.gen_beam(beamProfile)
            _d = self.current_calib.gap_and_beam_position_from_gap0(self.data.structure_gap, position)
            gap = _d['gap']
            beam_position = _d['beam_position']
            forward_dict = self.tracker.forward_propagate_forced(gap, beam_position, beam)
            forward_dicts.append(forward_dict)
            centroids[ctr] = forward_dict['screen'].mean()
            beam_sizes[ctr] = forward_dict['screen'].rms()
        return {
                'forward_dicts': forward_dicts,
                'mean': centroids,
                'rms': beam_sizes,
                'calib': self.current_calib,
                'profile': beamProfile,
                'positions': positions,
                }

    def reconstruct_closest(self, **kwargs):
        positions = self.data.positions
        gap0 = self.data.structure_gap
        pos_dicts =[self.current_calib.gap_and_beam_position_from_gap0(gap0, pos) for pos in positions]
        index_min = np.argmin([x['distance'] for x in pos_dicts]).squeeze()
        singlepos = self.data.single_position_data[index_min]
        median_image = singlepos.get_median_image(self.data.streaking_direction)
        meas_screen = median_image.get_screen_dist(self.data.streaking_direction)
        _d = pos_dicts[index_min]
        return self.tracker.reconstruct_profile_Gauss_forced(_d['gap'], _d['beam_position'], meas_screen, **kwargs)

