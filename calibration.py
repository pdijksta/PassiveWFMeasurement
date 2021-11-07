import bisect
import numpy as np
from scipy.optimize import curve_fit

from . import data_loader
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

    def gap_and_beam_position_from_meta(self, meta_data):
        gap0 = meta_data[self.structure_name+':GAP']*1e-3
        gap = gap0 + self.delta_gap
        structure_position0 = meta_data[self.structure_name+':CENTER']*1e-3
        beam_position = -(structure_position0 - self.structure_position0)
        distance = gap/2. - abs(beam_position)
        if distance < 0:
            raise ValueError('Distance between beam and gap is negative')

        return {
                'gap0': gap0,
                'gap': gap,
                'structure_position0': structure_position0,
                'beam_position': beam_position,
                'distance': distance,
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


class MeasScreens:
    def __init__(self, meas_screens, beam_positions, streaking_factors):
        self.meas_screens = meas_screens
        self.beam_positions = beam_positions
        self.streaking_factors = streaking_factors

    def plot(self, plot_handles=None):
        if plot_handles is None:
            ms.figure('Structure calibration measured screens distributions')
            subplot = ms.subplot_factory(2,2)
            sp_ctr = 1
            sp_pos = subplot(sp_ctr, title='Raw structure position > 0', xlabel='x (mm)', ylabel=config.rho_label)
            sp_ctr += 1
            sp_neg = subplot(sp_ctr, title='Raw structure position < 0', xlabel='x (mm)', ylabel=config.rho_label)
        else:
            sp_pos, sp_neg = plot_handles
        for beam_pos, screen in zip(self.beam_positions, self.meas_screens):
            if beam_pos == 0:
                screen.plot_standard(sp_pos, color='black')
                screen.plot_standard(sp_neg, color='black')
            else:
                if beam_pos > 0:
                    sp = sp_pos
                elif beam_pos < 0:
                    sp = sp_neg
                screen.plot_standard(sp, label='%.3f' % (beam_pos*1e3))
        sp_pos.legend(title='Beam position (mm)')
        sp_neg.legend(title='Beam position (mm)')
        return (sp_pos, sp_neg)


class StructureCalibrator(LogMsgBase):

    def __init__(self, tracker, structure_calib_options, file_or_dict=None, blmeas_profile=None, logger=None):
        self.logger = logger
        self.tracker = tracker
        self.structure_name = tracker.structure_name
        self.structure_calib_options = structure_calib_options
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
        self.blmeas_profile = blmeas_profile
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

        bs0 = streaking_factors[index0]
        streaking_factors = streaking_factors/bs0

        beam_positions = -(self.raw_struct_positions - np.mean(self.screen_center_arr))

        self.meas_screens = MeasScreens(meas_screens, beam_positions, streaking_factors)

    def get_meas_screens(self):
        if self.meas_screens is None:
            self.init_meas_screens()
        return self.meas_screens

    def get_result_dict(self):
        fit_dict_centroid = self.fit_dicts_gap_order['centroid']
        fit_dict_rms = self.fit_dicts_gap_order['beamsize']
        meta_data = {
                 'centroid_mean': self.centroids,
                 'centroid_std': self.centroids_std,
                 'rms_mean': self.rms,
                 'rms_std': self.rms_std,
                 'raw_struct_positions': self.raw_struct_positions,
                 'semigap': fit_dict_centroid['gap_fit']/2.,
                 'structure_position0': fit_dict_centroid['structure_position0'],
                 'x_axis': self.plot_list_x[0],
                 'structure_name': self.structure_name,
                 'fit_dict_rms': fit_dict_rms,
                 'fit_dict_centroid': fit_dict_centroid
                 }
        output = {
                'raw_data': self.raw_data,
                'meta_data': meta_data,
                'blmeas_profile': self.blmeas_profile
                }
        return output

    def add_data(self, raw_struct_positions, images, x_axis, y_axis):

        proj_cutoff = self.structure_calib_options['proj_cutoff']

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            images = images[...,::-1]
        n_images = images.shape[1]
        centroids = np.zeros([len(raw_struct_positions), n_images])
        rms = np.zeros_like(centroids)
        proj_x = images.astype(np.float64).sum(axis=-2)

        where0 = np.argwhere(raw_struct_positions == 0).squeeze()
        assert where0.size == 1

        plot_list_y = []
        plot_list_image = []
        for n_o in range(len(raw_struct_positions)):
            for n_i in range(n_images):
                proj = proj_x[n_o,n_i]
                proj = proj - np.median(proj)
                proj[proj<proj.max()*proj_cutoff] = 0
                centroids[n_o,n_i] = cc = np.sum(proj*x_axis) / np.sum(proj)
                rms[n_o, n_i] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
            median_proj_index = data_loader.get_median(proj_x[n_o,:], method='mean', output='index')
            median_proj = proj_x[n_o, median_proj_index]
            plot_list_image.append(images[n_o, median_proj_index])
            plot_list_y.append(median_proj)
        centroid_mean = np.mean(centroids, axis=1)
        screen_center = centroid_mean[where0]
        centroid_mean -= screen_center
        centroids -= screen_center
        screen_center_arr = np.array([screen_center]*len(raw_struct_positions), float)
        centroid_std = np.std(centroids, axis=1)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

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

    @staticmethod
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

    @staticmethod
    def centroid_fit_func(raw_struct_positions, structure_position0, strength, order, semigap, const):
        c1 = np.abs((raw_struct_positions-structure_position0+semigap))**(-order)
        c2 = np.abs((raw_struct_positions-structure_position0-semigap))**(-order)
        return const + (c1 - c2)*strength

    def fit_type(self, type_):

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
            fit_func = self.beamsize_fit_func
            order0 = self.structure_calib_options['order_rms']
        elif type_ == 'centroid':
            yy_mean = self.centroids
            yy_std = self.centroids_std
            fit_func = self.centroid_fit_func
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

        def fit_func2(*args):
            args = list(args)
            if fit_order:
                if fit_gap:
                    output = fit_func(*args, const0)
                else:
                    output = fit_func(*args, semigap, const0)
            else:
                if fit_gap:
                    output = fit_func(*args[:-1], order0, args[-1], const0)
                else:
                    output = fit_func(*args, order0, semigap, const0)
            return output

        try:
            p_opt, p_cov = curve_fit(fit_func2, raw_struct_positions, yy_mean, p0, sigma=yy_std)
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
        xx_fit2_min = -(gap_fit/2-structure_position0-10e-6)
        xx_fit2_max = -xx_fit2_min + 2*structure_position0
        xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
        reconstruction = fit_func2(xx_fit, *p_opt)
        reconstruction2 = fit_func2(xx_fit2, *p_opt)
        initial_guess = fit_func2(xx_fit, *p0)

        screen_center = np.mean(self.screen_center_arr)
        delta_gap = gap_fit - self.tracker.structure_gap0

        calibration = StructureCalibration(self.structure_name, screen_center, delta_gap, structure_position0)

        fit_dict = {
                'reconstruction': reconstruction,
                'reconstruction2': reconstruction2,
                'initial_guess': initial_guess,
                'structure_position0': structure_position0,
                'gap_fit': gap_fit,
                'order_fit': order_fit,
                'p_opt': p_opt,
                'p0': p0,
                'xx_fit': xx_fit,
                'xx_fit2': xx_fit2,
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
        self.logMsg('structure_position0 calibrated as %i um' % (structure_position0*1e6))
        return fit_dict

    def fit(self):
        a = self.fit_type('beamsize')
        b = self.fit_type('centroid')
        return a, b

    def forward_propagate(self, blmeas_profile, tt_halfrange, tracker, type_='centroid', blmeas_cutoff=None, force_gap=None, force_streaker_offset=None):
        tracker.set_simulator(self.meta_data)
        if force_streaker_offset is None:
            structure_position0 = self.fit_dicts[type_]['structure_position0']
        else:
            structure_position0 = force_streaker_offset
        if force_gap is None:
            gap = self.fit_dicts[type_]['gap_fit']
        else:
            gap = force_gap
        if type(blmeas_profile) is beam_profile.BeamProfile:
            pass
        else:
            try:
                blmeas_profile = beam_profile.profile_from_blmeas(blmeas_profile, tt_halfrange, tracker.energy_eV, True)
            except Exception:
                print(type(blmeas_profile))
                print(type(beam_profile.BeamProfile))
                raise
            if blmeas_cutoff is None:
                blmeas_profile.cutoff2(tracker.profile_cutoff)
            else:
                blmeas_profile.cutoff2(blmeas_cutoff)
            blmeas_profile.crop()
            blmeas_profile.reshape(tracker.len_screen)

        len_screen = tracker.len_screen
        gaps = np.array([10., 10.])
        gaps[self.n_streaker] = gap
        beam_offsets0 = np.array([0., 0.])

        sim_screens = []
        forward_dicts = []
        for s_offset in self.raw_struct_positions:
            beam_offsets = beam_offsets0[:]
            beam_offsets[self.n_streaker] = -(s_offset-structure_position0)
            forward_dict = tracker.matrix_forward(blmeas_profile, gaps, beam_offsets)
            forward_dicts.append(forward_dict)
            sim_screen = forward_dict['screen']
            sim_screen.cutoff2(tracker.screen_cutoff)
            sim_screen.crop()
            sim_screen.reshape(len_screen)
            sim_screens.append(sim_screen)

        self.blmeas_profile = blmeas_profile
        self.sim_screen_dict[(gap, structure_position0)] = sim_screens
        self.sim_screens = sim_screens
        output = {
                'blmeas_profile': blmeas_profile,
                'sim_screens': sim_screens,
                'forward_dicts': forward_dicts,
                }
        return output


    def reconstruct_current(self, plot_details=False, force_gap=None, force_struct_position0=None, use_n_positions=None):
        if force_gap is not None:
            gap = force_gap
        else:
            gap = self.fit_dicts['centroid']['gap_fit']
        if force_struct_position0 is not None:
            structure_position0 = force_struct_position0
        else:
            structure_position0 = self.fit_dicts['centroid']['structure_position0']
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
            #print(beam_offsets[self.n_streaker])
            beam_position_list.append(beam_position)
            meas_screen0 = meas_screens0[n_position]
            gauss_dict = self.tracker.reconstruct_profile_Gauss_forced(gap, beam_position, meas_screen0, plot_details=plot_details)
            gauss_dicts.append(gauss_dict)
        self.gauss_dicts = gauss_dicts
        return np.array(beam_position_list), gauss_dicts

    def reconstruct_gap(self, use_n_positions=None):

        precision = self.structure_calib_options['gap_recon_precision']
        structure_position0 = self.fit_dicts['centroid']['structure_position0']
        gap_arr = self.tracker.structure_gap0 + np.array(self.structure_calib_options['gap_recon_delta'])

        gaps = []
        rms = []
        lin_fit = []
        lin_fit_const = []

        n_positions = len(use_n_positions) if use_n_positions is not None else len(self.raw_struct_positions)-1
        distance_rms_arr = [[] for _ in range(n_positions)]

        def one_gap(gap, force=False):
            if not force:
                gap = np.round(gap/precision)*precision
                if gap in gaps:
                    return
            beam_position_list, gauss_dicts = self.reconstruct_current(force_gap=gap, force_struct_position0=structure_position0, use_n_positions=use_n_positions)
            distance_arr = gap/2. - np.abs(beam_position_list)

            rms_arr = np.array([x['reconstructed_profile'].rms() for x in gauss_dicts])
            d_arr2 = distance_arr - distance_arr.min()
            sort = np.argsort(d_arr2)
            fit = np.polyfit(d_arr2[sort], rms_arr[sort], 1)

            for n in range(n_positions):
                distance_rms_arr[n].append((distance_arr[n], rms_arr[n]))

            index = bisect.bisect(gaps, gap)
            gaps.insert(index, gap)
            rms.insert(index, rms_arr)
            lin_fit.insert(index, fit[0])
            lin_fit_const.insert(index, fit[1])
            return beam_position_list, gauss_dicts

        def get_gap():
            lin_fit2 = np.array(lin_fit)
            gaps2 = np.array(gaps)
            err = False
            for i in range(len(gaps)-1):
                lin_fit_a = lin_fit2[i]
                lin_fit_b = lin_fit2[i+1]
                if np.sign(lin_fit_a) != np.sign(lin_fit_b):
                    gaps3 = np.array([gaps2[i], gaps2[i+1]])
                    lin_fit3 = np.array([lin_fit_a, lin_fit_b])
                    sort = np.argsort(lin_fit3)
                    gap = np.interp(0, lin_fit3[sort], gaps3[sort], left=np.nan, right=np.nan)
                    break
            else:
                err = True
            if not err and np.isnan(gap):
                err = True
            if err:
                sort = np.argsort(lin_fit2)
                gap = np.interp(0, lin_fit2[sort], gaps2[sort])
                #raise ValueError('Gap interpolated to %e. Gap_arr limits: %e, %e' % (gap, gap_arr.min(), gap_arr.max()))
                print('Gap interpolated to %e. Gap_arr limits: %e, %e' % (gap, gap_arr.min(), gap_arr.max()))
            return gap

        for gap in [gap_arr.min(), gap_arr.max()]:
            one_gap(gap)

        old_gap = np.inf
        for _ in range(20):
            gap = get_gap()
            if abs(old_gap - gap)<precision:
                break
            one_gap(gap)
            old_gap = gap

        gap = get_gap()
        beam_position_list, gauss_dicts = one_gap(gap, force=True)

        rms_arr = np.array(rms)
        rms_rms = np.std(rms_arr, axis=1)
        rms_mean = np.mean(rms_arr, axis=1)
        gap_arr = np.array(gaps)
        sort = np.argsort(gap_arr)
        assumed_rms = np.interp(gap, gap_arr[sort], rms_rms[sort])
        assumed_bunch_duration = np.interp(gap, gap_arr[sort], rms_mean[sort])

        best_index = np.argwhere(gap_arr == gap).squeeze()

        output = {
                'gap': gap,
                'gap0': self.tracker.structure_gap0,
                'delta_gap': gap - self.tracker.structure_gap0,
                'beamsize_rms': assumed_rms,
                'beamsize': assumed_bunch_duration,
                'gap_arr': np.array(gaps),
                'lin_fit': np.array(lin_fit),
                'lin_fit_const': np.array(lin_fit_const),
                'all_rms': rms_arr,
                'use_n_positions': use_n_positions,
                'final_gauss_dicts': gauss_dicts,
                'final_beam_positions': np.array(beam_position_list),
                'best_index': best_index,
                'distance_rms_arr': distance_rms_arr,
                }
        return output

    def get_n_positions(self, gap, max_distance):
        distances = gap/2. - abs(self.raw_struct_positions - self.fit_dicts['centroid']['structure_position0'])
        index_arr = np.arange(len(self.raw_struct_positions))
        return index_arr[distances <= max_distance]

