import bisect
import numpy as np
import matplotlib.pyplot as plt
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
        self.screen_center = screen_center,
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

class StreakerCalibration(LogMsgBase):

    order0_centroid = 2.75
    order0_rms = 2.75

    def __init__(self, beamline, n_streaker, gap0, charge, file_or_dict=None, offsets_range=None, images=None, x_axis=None, y_axis=None, fit_gap=True, fit_order=False, order_centroid=None, order_rms=None, proj_cutoff=0.03, logger=None):
        self.logger = logger
        self.order_rms = self.order0_rms if order_rms is None else order_rms
        self.order_centroid = self.order0_centroid if order_centroid is None else order_centroid
        self.fit_gap = fit_gap
        self.fit_order = fit_order
        self.gap0 = gap0
        self.n_streaker = n_streaker
        self.proj_cutoff = proj_cutoff
        self.beamline = beamline
        self.charge = charge

        self.offsets = []
        self.screen_x0_arr = []
        self.centroids = []
        self.centroids_std = []
        self.all_rms = []
        self.all_centroids = []
        self.rms = []
        self.rms_std = []
        self.images = []
        self.blmeas_profile = None
        self.sim_screen_dict = {}
        self.sim_screens = None
        self.plot_list_x = []
        self.plot_list_y = []
        self.plot_list_image = []
        self.y_axis_list = []
        self.raw_data = None
        self.meas_screens = None

        if offsets_range is not None:
            self.add_data(offsets_range, images, x_axis, y_axis)
        if file_or_dict is not None:
            self.add_file(file_or_dict)

        self.fit_dicts = {}

    def get_meas_screens(self, type_='centroid', cutoff=3e-2, shape=int(5e3)):
        streaking_factors = []
        meas_screens = []
        index0 = np.argwhere(self.offsets == 0).squeeze()

        for x, y in zip(self.plot_list_x, self.plot_list_y):
            meas_screen = beam_profile.ScreenDistribution(x, y, charge=self.charge)
            meas_screen.cutoff2(cutoff)
            meas_screen.crop()
            meas_screen.reshape(shape)
            meas_screens.append(meas_screen)
            streaking_factors.append(meas_screen.rms())
        self.meas_screens = meas_screens

        bs0 = streaking_factors[index0]
        self.streaking_factors = streaking_factors/bs0

        return meas_screens

    def get_result_dict(self):
        fit_dict_centroid = self.fit_dicts_gap_order['centroid']
        fit_dict_rms = self.fit_dicts_gap_order['beamsize']
        meta_data = {
                 'centroid_mean': self.centroids,
                 'centroid_std': self.centroids_std,
                 'rms_mean': self.rms,
                 'rms_std': self.rms_std,
                 'offsets': self.offsets,
                 'semigap': fit_dict_centroid['gap_fit']/2.,
                 'streaker_offset': fit_dict_centroid['streaker_offset'],
                 'x_axis': self.plot_list_x[0],
                 'streaker': config.streaker_names[self.beamline][self.n_streaker],
                 'fit_dict_rms': fit_dict_rms,
                 'fit_dict_centroid': fit_dict_centroid
                 }
        output = {
                'raw_data': self.raw_data,
                'meta_data': meta_data,
                'blmeas_profile': self.blmeas_profile
                }
        return output

    def add_data(self, offsets, images, x_axis, y_axis):

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            images = images[...,::-1]
        n_images = images.shape[1]
        centroids = np.zeros([len(offsets), n_images])
        rms = np.zeros_like(centroids)
        proj_x = images.astype(np.float64).sum(axis=-2)

        where0 = np.argwhere(offsets == 0).squeeze()
        assert where0.size == 1

        plot_list_y = []
        plot_list_image = []
        for n_o in range(len(offsets)):
            for n_i in range(n_images):
                proj = proj_x[n_o,n_i]
                proj = proj - np.median(proj)
                proj[proj<proj.max()*self.proj_cutoff] = 0
                centroids[n_o,n_i] = cc = np.sum(proj*x_axis) / np.sum(proj)
                rms[n_o, n_i] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
            median_proj_index = data_loader.get_median(proj_x[n_o,:], method='mean', output='index')
            median_proj = proj_x[n_o, median_proj_index]
            plot_list_image.append(images[n_o, median_proj_index])
            plot_list_y.append(median_proj)
        centroid_mean = np.mean(centroids, axis=1)
        screen_x0 = centroid_mean[where0]
        centroid_mean -= screen_x0
        centroids -= screen_x0
        screen_x0_arr = np.array([screen_x0]*len(offsets), float)
        centroid_std = np.std(centroids, axis=1)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

        if 0 in self.offsets:
            mask = offsets != 0
        else:
            mask = np.ones_like(offsets, dtype=bool)

        new_offsets = np.concatenate([self.offsets, offsets[mask]])
        sort = np.argsort(new_offsets)
        self.offsets = new_offsets[sort]
        self.centroids = np.concatenate([self.centroids, centroid_mean[mask]])[sort]
        self.centroids_std = np.concatenate([self.centroids_std, centroid_std[mask]])[sort]
        self.rms = np.concatenate([self.rms, rms_mean[mask]])[sort]
        self.rms_std = np.concatenate([self.rms_std, rms_std[mask]])[sort]
        self.screen_x0_arr = np.concatenate([self.screen_x0_arr, screen_x0_arr[mask]])[sort]

        plot_list_x = [x_axis - screen_x0] * len(plot_list_y)
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

        offsets = data_dict['streaker_offsets'].squeeze()
        self.add_data(offsets, images, x_axis, y_axis)
        self.raw_data = data_dict

    @staticmethod
    def beamsize_fit_func(offsets, streaker_offset, strength, order, semigap, const):
        sq0 = const**2
        c1 = np.abs((offsets-streaker_offset+semigap))**(-order*2)
        c2 = np.abs((offsets-streaker_offset-semigap))**(-order*2)
        if strength > 0:
            sq_add = strength**2 * (c1+c2)
        else:
            sq_add = np.zeros(len(offsets))
        output = np.sqrt(sq0 + sq_add)
        return output

    @staticmethod
    def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, semigap, const):
        c1 = np.abs((offsets-streaker_offset+semigap))**(-order)
        c2 = np.abs((offsets-streaker_offset-semigap))**(-order)
        return const + (c1 - c2)*strength

    def fit_type(self, type_):

        offsets = self.offsets
        if len(offsets) == 0:
            raise ValueError('No data!')
        semigap = self.gap0/2.
        where0 = np.argwhere(offsets == 0).squeeze()

        if type_ == 'beamsize':
            yy_mean = self.rms
            yy_std = self.rms_std
            fit_func = self.beamsize_fit_func
            order0 = self.order_rms
        elif type_ == 'centroid':
            yy_mean = self.centroids
            yy_std = self.centroids_std
            fit_func = self.streaker_calibration_fit_func
            order0 = self.order_centroid

        const0 = yy_mean[where0]
        offset0 = (offsets[0] + offsets[-1])/2

        if type_ == 'centroid':
            s0_arr = (yy_mean-const0)/(np.abs((offsets-offset0-semigap))**(-order0) - np.abs((offsets-offset0+semigap))**(-order0))
        elif type_ == 'beamsize':
            s0_arr = (yy_mean-const0)/(np.abs((offsets-offset0-semigap))**(-order0) + np.abs((offsets-offset0+semigap))**(-order0))
        s0 = (s0_arr[0] + s0_arr[-1])/2
        p0 = [offset0, s0]
        if self.fit_order:
            p0.append(order0)
        if self.fit_gap:
            p0.append(semigap)

        def fit_func2(*args):
            args = list(args)
            if self.fit_order:
                if self.fit_gap:
                    output = fit_func(*args, const0)
                else:
                    output = fit_func(*args, semigap, const0)
            else:
                if self.fit_gap:
                    output = fit_func(*args[:-1], order0, args[-1], const0)
                else:
                    output = fit_func(*args, order0, semigap, const0)
            return output

        try:
            p_opt, p_cov = curve_fit(fit_func2, offsets, yy_mean, p0, sigma=yy_std)
        except RuntimeError:
            print('Streaker calibration type %s did not converge' % type_)
            p_opt = p0

        streaker_offset = p_opt[0]
        if self.fit_gap:
            gap_fit = p_opt[-1]*2
        else:
            gap_fit = abs(semigap*2)
        if self.fit_order:
            order_fit = p_opt[2]
        else:
            order_fit = order0

        xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
        xx_fit2_min = -(gap_fit/2-streaker_offset-10e-6)
        xx_fit2_max = -xx_fit2_min + 2*streaker_offset
        xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
        reconstruction = fit_func2(xx_fit, *p_opt)
        reconstruction2 = fit_func2(xx_fit2, *p_opt)
        initial_guess = fit_func2(xx_fit, *p0)
        fit_dict = {
                'reconstruction': reconstruction,
                'reconstruction2': reconstruction2,
                'initial_guess': initial_guess,
                'streaker_offset': streaker_offset,
                'gap_fit': gap_fit,
                'order_fit': order_fit,
                'p_opt': p_opt,
                'p0': p0,
                'xx_fit': xx_fit,
                'xx_fit2': xx_fit2,
                'screen_rms0': const0,
                'screen_x0_arr': self.screen_x0_arr,
                'screen_x0': np.mean(self.screen_x0_arr),
                }
        self.fit_dicts[type_] = fit_dict
        return fit_dict

    def fit(self):
        a = self.fit_type('beamsize')
        b = self.fit_type('centroid')
        return a, b

    def forward_propagate(self, blmeas_profile, tt_halfrange, tracker, type_='centroid', blmeas_cutoff=None, force_gap=None, force_streaker_offset=None):
        tracker.set_simulator(self.meta_data)
        if force_streaker_offset is None:
            streaker_offset = self.fit_dicts[type_]['streaker_offset']
        else:
            streaker_offset = force_streaker_offset
        if force_gap is None:
            gap = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['gap_fit']
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
        for s_offset in self.offsets:
            beam_offsets = beam_offsets0[:]
            beam_offsets[self.n_streaker] = -(s_offset-streaker_offset)
            forward_dict = tracker.matrix_forward(blmeas_profile, gaps, beam_offsets)
            forward_dicts.append(forward_dict)
            sim_screen = forward_dict['screen']
            sim_screen.cutoff2(tracker.screen_cutoff)
            sim_screen.crop()
            sim_screen.reshape(len_screen)
            sim_screens.append(sim_screen)

        self.blmeas_profile = blmeas_profile
        self.sim_screen_dict[(gap, streaker_offset)] = sim_screens
        self.sim_screens = sim_screens
        output = {
                'blmeas_profile': blmeas_profile,
                'sim_screens': sim_screens,
                'forward_dicts': forward_dicts,
                }
        return output

    def plot_streaker_calib(self, plot_handles=None, figsize=None):

        if plot_handles is None:
            fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = streaker_calibration_figure(figsize)
        else:
            (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = plot_handles

        offsets = self.offsets
        fit_dict_centroid = self.fit_dicts_gap_order['centroid'][self.fit_gap][self.fit_order]
        fit_dict_rms = self.fit_dicts_gap_order['beamsize'][self.fit_gap][self.fit_order]
        blmeas_profile = self.blmeas_profile
        forward_propagate_blmeas = (blmeas_profile is not None)
        screen_x0 = 0

        meas_screens = self.get_meas_screens()
        rms_sim = np.zeros(len(offsets))
        centroid_sim = np.zeros(len(offsets))
        if self.sim_screens is not None:
            sim_screens = self.sim_screens
            #len_screen = len(sim_screens[0])
            for n_proj, (meas_screen, offset) in enumerate(zip(meas_screens, offsets)):
                color = ms.colorprog(n_proj, offsets)
                meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
                sim_screen = sim_screens[n_proj]
                sim_screen.plot_standard(sp_proj, color=color, ls='--')
                centroid_sim[n_proj] = sim_screen.mean()
                rms_sim[n_proj] = sim_screen.rms()
        else:
            sim_screens = None

        if forward_propagate_blmeas:
            blmeas_profile.plot_standard(sp_current, color='black', ls='--')

        for fit_dict, sp1, sp2, yy, yy_err, yy_sim in [
                (fit_dict_centroid, sp_center, sp_center2, self.centroids, self.centroids_std, centroid_sim),
                (fit_dict_rms, sp_sizes, sp_sizes2, self.rms, self.rms_std, rms_sim),
                ]:

            xx_fit = fit_dict['xx_fit']
            xx_fit2 = fit_dict['xx_fit2']
            reconstruction = fit_dict['reconstruction']
            reconstruction2 = fit_dict['reconstruction2']
            gap = fit_dict['gap_fit']
            fit_semigap = gap/2
            streaker_offset = fit_dict['streaker_offset']

            xx_plot = (offsets - streaker_offset)
            xx_plot_fit = (xx_fit - streaker_offset)
            sp1.errorbar(xx_plot*1e3, (yy-screen_x0)*1e3, yerr=yy_err*1e3, label='Data', ls='None', marker='o')
            sp1.plot(xx_plot_fit*1e3, (reconstruction-screen_x0)*1e3, label='Fit')

            mask_pos, mask_neg = offsets > 0, offsets < 0
            xx_plot2 = np.abs(fit_semigap - np.abs(xx_plot))
            for mask2, label in [(mask_pos, 'Positive'), (mask_neg, 'Negative')]:
                sp2.errorbar(xx_plot2[mask2]*1e6, np.abs(yy[mask2]-screen_x0)*1e3, yerr=yy_err[mask2]*1e3, label=label, marker='o', ls='None')

            if sim_screens is not None:
                plot2_sim = []
                for mask in mask_pos, mask_neg:
                    plot2_sim.extend([(a, np.abs(b)) for a, b in zip(xx_plot2[mask], yy_sim[mask])])
                plot2_sim.sort()
                xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
                xx_plot_sim = np.array(xx_plot_sim)
                yy_plot_sim = np.array(yy_plot_sim)
                sp2.plot(xx_plot_sim*1e6, yy_plot_sim*1e3, label='Simulated', ls='None', marker='o')
                sp1.plot(xx_plot*1e3, yy_sim*1e3, label='Simulated', marker='.', ls='None')

            xx_plot_fit2 = np.abs(fit_semigap - np.abs(xx_fit2 - streaker_offset))
            yy_plot_fit2 = np.abs(reconstruction2)-screen_x0
            xlims = sp_center2.get_xlim()
            mask_fit = np.logical_and(xx_plot_fit2*1e6 > xlims[0], xx_plot_fit2*1e6 < xlims[1])
            mask_fit = np.logical_and(mask_fit, xx_fit2 > 0)
            sp2.plot(xx_plot_fit2[mask_fit]*1e6, yy_plot_fit2[mask_fit]*1e3, label='Fit')
            sp2.set_xlim(*xlims)

            title = sp1.get_title()
            sp1.set_title('%s; Gap=%.2f mm' % (title, fit_dict['gap_fit']*1e3), fontsize=config.fontsize)
            title = sp2.get_title()
            sp2.set_title('%s; Center=%i $\mu$m' % (title, round(fit_dict['streaker_offset']*1e6)), fontsize=config.fontsize)
            sp1.legend()
            sp2.legend()

#    def reconstruct_current(self, tracker, gauss_kwargs, type_='centroid', plot_details=False, force_gap=None, force_streaker_offset=None, use_offsets=None):
#        fit_dict = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]
#
#        if force_gap is not None:
#            gap = force_gap
#        else:
#            gap = fit_dict['gap_fit']
#        if force_streaker_offset is not None:
#            streaker_offset = force_streaker_offset
#        else:
#            streaker_offset = fit_dict['streaker_offset']
#        #print('Streaker offset', '%i' % (streaker_offset*1e6))
#
#        gaps = [10e-3, 10e-3]
#        gaps[self.n_streaker] = gap
#        gauss_dicts = []
#        offset_list = []
#
#        if use_offsets is None:
#            n_offsets = range(len(self.offsets))
#        else:
#            n_offsets = use_offsets
#
#        for n_offset in n_offsets:
#            offset = self.offsets[n_offset]
#            if offset == 0:
#                continue
#            beam_offsets = [0., 0.]
#            beam_offsets[self.n_streaker] = -(offset-streaker_offset)
#            #print(beam_offsets[self.n_streaker])
#            offset_list.append(beam_offsets[self.n_streaker])
#
#            projx = self.images[n_offset].astype(np.float64).sum(axis=-2)
#            median_proj = data_loader.get_median(projx)
#            x_axis = self.plot_list_x[n_offset]
#            if x_axis[1] < x_axis[0]:
#                x_axis = x_axis[::-1]
#                median_proj = median_proj[::-1]
#            meas_screen = beam_profile.ScreenDistribution(x_axis, median_proj, subtract_min=True, charge=self.charge)
#            meas_screen.cutoff2(tracker.screen_cutoff)
#            meas_screen.crop()
#            meas_screen.reshape(tracker.len_screen)
#
#            gauss_kwargs = gauss_kwargs.copy()
#            gauss_kwargs['gaps'] = gaps
#            gauss_kwargs['beam_offsets'] = beam_offsets
#            gauss_kwargs['n_streaker'] = self.n_streaker
#            gauss_kwargs['meas_screen'] = meas_screen
#
#            #plt.figure()
#            #plt.plot(meas_screen._xx, meas_screen._yy)
#            #plt.show()
#            #import pdb; pdb.set_trace()
#
#            #gauss_dict = tracker.find_best_gauss(**gauss_kwargs)
#            gauss_dict = analysis.current_profile_rec_gauss(tracker, gauss_kwargs, do_plot=False)
#            if plot_details:
#                analysis.plot_rec_gauss(tracker, gauss_kwargs, gauss_dict)
#            gauss_dicts.append(gauss_dict)
#
#        self.gauss_dicts_gap_order[type_][self.fit_gap][self.fit_order] = gauss_dicts
#        return np.array(offset_list), gauss_dicts

    def plot_reconstruction(self, plot_handles=None, blmeas_profile=None, max_distance=350e-6, type_='centroid', figsize=None):
        center = 'Mean'
        if plot_handles is None:
            fig, (sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg) = gauss_recon_figure(figsize)
        else:
            fig, (sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg) = plot_handles

        #gap = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['gap_fit']
        #streaker_center = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['streaker_offset']
        gauss_dicts = self.gauss_dicts_gap_order[type_][self.fit_gap][self.fit_order]

        if len(gauss_dicts) == 0:
            raise ValueError
        for gauss_dict in gauss_dicts:
            beam_offset = gauss_dict['beam_offsets'][self.n_streaker]

            sp_screen = sp_screen_pos if beam_offset > 0 else sp_screen_neg
            sp_profile = sp_profile_pos if beam_offset > 0 else sp_profile_neg

            semigap = gauss_dict['gaps'][self.n_streaker]/2
            distance = semigap-abs(beam_offset)
            if distance > max_distance:
                continue

            rec_profile = gauss_dict['reconstructed_profile']
            label = '%i' % (round(rec_profile.rms()*1e15))
            rec_profile.plot_standard(sp_profile, label=label, center=center)

            meas_screen = gauss_dict['meas_screen']
            rec_screen = gauss_dict['reconstructed_screen']
            label = '%i' % round(distance*1e6)
            color = meas_screen.plot_standard(sp_screen, label=label)[0].get_color()
            rec_screen.plot_standard(sp_screen, ls='--', color=color)

        if blmeas_profile is not None:
            for _sp in sp_profile_pos, sp_profile_neg:
                blmeas_profile.plot_standard(_sp, color='black', center=center, ls='--', label='%i' % round(blmeas_profile.rms()*1e15))

        for _sp in sp_screen_pos, sp_screen_neg:
            _sp.legend(title='d ($\mu$m)')
        for _sp in sp_profile_pos, sp_profile_neg:
            _sp.legend(title='rms (fs)')

    def gap_reconstruction2(self, gap_arr, tracker, gauss_kwargs, streaker_offset, precision=1e-6, gap0=0, use_offsets=None):
        """
        Optimized version
        """
        #print('method', gauss_kwargs['method'])

        gaps = []
        rms = []
        lin_fit = []
        lin_fit_const = []

        def one_gap(gap, force=False):
            if not force:
                gap = np.round(gap/precision)*precision
                if gap in gaps:
                    return
            offset_list, gauss_dicts = self.reconstruct_current(tracker, gauss_kwargs, plot_details=False, force_gap=gap, force_streaker_offset=streaker_offset, use_offsets=use_offsets)
            distance_arr = gap/2. - np.abs(offset_list)

            rms_arr = np.array([x['reconstructed_profile'].rms() for x in gauss_dicts])
            d_arr2 = distance_arr - distance_arr.min()
            sort = np.argsort(d_arr2)
            fit = np.polyfit(d_arr2[sort], rms_arr[sort], 1)

            index = bisect.bisect(gaps, gap)
            gaps.insert(index, gap)
            rms.insert(index, rms_arr)
            lin_fit.insert(index, fit[0])
            lin_fit_const.insert(index, fit[1])
            return offset_list, gauss_dicts

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
        offset_list, gauss_dicts = one_gap(gap, force=True)

        rms_arr = np.array(rms)
        rms_rms = np.std(rms_arr, axis=1)
        rms_mean = np.mean(rms_arr, axis=1)
        gap_arr = np.array(gaps)
        sort = np.argsort(gap_arr)
        assumed_rms = np.interp(gap, gap_arr[sort], rms_rms[sort])
        assumed_bunch_duration = np.interp(gap, gap_arr[sort], rms_mean[sort])
        output = {
                'gap': gap,
                'gap0': gap0,
                'delta_gap': gap - gap0,
                'beamsize_rms': assumed_rms,
                'beamsize': assumed_bunch_duration,
                'gap_arr': np.array(gaps),
                'lin_fit': np.array(lin_fit),
                'lin_fit_const': np.array(lin_fit_const),
                'all_rms': rms_arr,
                'input': {
                    'gap_arr': gap_arr,
                    'gauss_kwargs': gauss_kwargs,
                    'streaker_offset': streaker_offset,
                    },
                'use_offsets': use_offsets,
                'final_gauss_dicts': gauss_dicts,
                'final_offset_list': offset_list,
                }
        return output

    def plot_gap_reconstruction(self, gap_recon_dict, plot_handles=None, figsize=None, exclude_gap_ctrs=()):
        if plot_handles is None:
            fig, plot_handles = gap_recon_figure(figsize=figsize)
        (sp_rms, sp_overview, sp_std, sp_fit) = plot_handles

        gap_arr = gap_recon_dict['gap_arr']
        all_rms_arr = gap_recon_dict['all_rms']
        lin_fit = gap_recon_dict['lin_fit']
        lin_fit_const = gap_recon_dict['lin_fit_const']
        gap0 = gap_recon_dict['gap0']
        use_offsets = gap_recon_dict['use_offsets']
        streaker_offset = gap_recon_dict['input']['streaker_offset']

        if use_offsets is None:
            offsets = self.offsets
        else:
            offsets = np.take(self.offsets, use_offsets)

        for gap_ctr in list(range(len(gap_arr)))[::-1]:
            if gap_ctr in exclude_gap_ctrs:
                continue
            gap = gap_arr[gap_ctr]
            distance_arr = gap/2. - np.abs(offsets[offsets != 0] - streaker_offset)
            d_arr2 = distance_arr - distance_arr.min()
            sort = np.argsort(d_arr2)
            _label = '%i' % round((gap-gap0)*1e6)
            #sp_centroid.plot(d_arr2, centroid_arr, label=_label)
            rms_arr = all_rms_arr[gap_ctr]
            #color = ms.colorprog(gap_ctr, gap_arr)
            color= sp_rms.plot(d_arr2[sort]*1e6, rms_arr[sort]*1e15, label=_label, marker='.')[0].get_color()
            fit_yy = lin_fit_const[gap_ctr] + lin_fit[gap_ctr]*d_arr2[sort]
            sp_rms.plot(d_arr2[sort]*1e6, fit_yy*1e15, color=color, ls='--')


        sp_overview.errorbar(gap_arr*1e3, all_rms_arr.mean(axis=-1)*1e15, yerr=all_rms_arr.std(axis=-1)*1e15)
        sp_std.plot(gap_arr*1e3, all_rms_arr.std(axis=-1)/all_rms_arr.mean(axis=-1), marker='.')
        sp_fit.plot(gap_arr*1e3, lin_fit*1e15/1e6, marker='.')

        sp_fit.axhline(0, color='black', ls='--')
        sp_fit.axvline(gap_recon_dict['gap']*1e3, color='black', ls='--')

        sp_rms.legend(title='$\Delta$g ($\mu$m)')


def gauss_recon_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Current reconstruction')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 2, grid=False)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+4)))
    clear_gauss_recon(*plot_handles)
    return fig, plot_handles

def clear_gauss_recon(sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg):
    for sp, title, xlabel, ylabel in [
            (sp_screen_pos, 'Screen profile (+)', 'x (mm)', config.rho_label),
            (sp_screen_neg, 'Screen profile (-)', 'x (mm)', config.rho_label),
            (sp_profile_pos, 'Beam current (+)', 't (fs)', 'Current (kA)'),
            (sp_profile_neg, 'Beam current (-)', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def streaker_calibration_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker center calibration')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 3)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+6)))
    clear_streaker_calibration(*plot_handles)
    return fig, plot_handles

def clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Centroid shift', 'Streaker center (mm)', 'Beam X centroid (mm)'),
            (sp_sizes, 'Size increase', 'Streaker center (mm)', 'Beam X rms (mm)'),
            (sp_center2, 'Centroid shift', 'Distance from jaw ($\mu$m)', 'Beam X centroid (mm)'),
            (sp_sizes2, 'Size increase', 'Distance from jaw ($\mu$m)', 'Beam X rms (mm)'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            (sp_current, 'Beam current', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def gap_recon_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker gap reconstruction')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 2, grid=False)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+4)))
    clear_gap_recon(*plot_handles)
    return fig, plot_handles

def clear_gap_recon(sp_rms, sp_overview, sp_std, sp_fit):
    for sp, title, xlabel, ylabel in [
            (sp_rms, 'Rms bunch duration', '$\Delta$d ($\mu$m)', 'rms (fs)'),
            (sp_overview, 'Rms bunch duration', 'Gap (mm)', 'rms (fs)'),
            (sp_std, 'Relative beamsized error', 'Gap (mm)', r'$\Delta \tau / \tau$'),
            (sp_fit, 'Fit coefficient', 'Gap (mm)', r'$\Delta \tau / \Delta$Gap (fs/$\mu$m)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)


