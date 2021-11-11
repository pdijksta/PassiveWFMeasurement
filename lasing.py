import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from . import h5_storage
from . import beam_profile
from . import myplotstyle as ms
from . import image_analysis
from . import plot_results

def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
    power[power<0] = 0
    return power

def power_Eloss_err(slice_time, slice_current, slice_E_on, slice_E_off, slice_current_err, slice_E_on_err, slice_E_off_err):
    delta_E = slice_E_off-slice_E_on
    power = slice_current * delta_E
    power[power<0] = 0
    energy = np.trapz(power, slice_time)

    err_sq_1 = (delta_E * slice_current_err)**2
    err_sq_2 = (slice_current * slice_E_on_err)**2
    err_sq_3 = (slice_current * slice_E_off_err)**2
    power_err = np.sqrt(err_sq_1+err_sq_2+err_sq_3)

    return {
            'time': slice_time,
            'power': power,
            'energy': energy,
            'power_err': power_err,
            }

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total, norm_factor=None):
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    power0[power0<0] = 0
    integral = np.trapz(power0, slice_t)
    if norm_factor is None:
        power = power0/integral*E_total
    else:
        power = power0*norm_factor
    return power

def power_Espread_err(slice_t, slice_current, slice_Espread_on, slice_Espread_off, E_total, slice_current_err, slice_Espread_on_err, slice_Espread_off_err, norm_factor=None):
    """
    Takes squared values of energy spread
    """

    slice_Espread_sqr_increase = slice_Espread_on - slice_Espread_off
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    power0[power0 < 0] = 0
    integral = np.trapz(power0, slice_t)
    if norm_factor is None:
        norm_factor = E_total/integral
    power = power0*norm_factor

    power0_err_1 = (slice_current**(-1/3) * slice_Espread_sqr_increase * slice_current_err)**2
    power0_err_2 = slice_current**(2/3) * slice_Espread_off_err
    power0_err_3 = slice_current**(2/3) * slice_Espread_on_err
    power0_err = np.sqrt(power0_err_1+power0_err_2+power0_err_3)

    power_err = power0_err*norm_factor
    energy = np.trapz(power, slice_t)

    return {
            'time': slice_t,
            'power': power,
            'power_err': power_err,
            'energy': energy,
            'norm_factor': norm_factor
            }

def obtain_lasing(image_off, image_on, n_slices, wake_x, wake_t, len_profile, dispersion, energy_eV, charge, pulse_energy, debug=False):

    all_slice_dict = OrderedDict()
    all_images = OrderedDict()

    for ctr, (image_obj, label) in enumerate([(image_off, 'Lasing_off'), (image_on, 'Lasing_on')]):

        image_cut = image_obj.cut(wake_x.min(), wake_x.max())
        image_reshaped = image_cut.reshape_x(len_profile)
        image_t = image_reshaped.x_to_t(wake_x, wake_t)
        if ctr == 0:
            ref_y = None
        image_tE, ref_y = image_t.y_to_eV(dispersion, energy_eV, ref_y)
        image_t_reduced = image_tE.slice_x(n_slices)
        slice_dict = image_t_reduced.fit_slice(charge=charge, smoothen_first=True, smoothen=1e6)
        all_slice_dict[label] = slice_dict
        all_images[label] = {
                'image_xy': image_obj,
                'image_tE': image_tE,
                'image_cut': image_cut,
                'image_t': image_t,
                'image_t_reduced': image_t_reduced,
                }

    slice_time = all_slice_dict['Lasing_off']['slice_x']
    mean_current = (all_slice_dict['Lasing_off']['slice_current']+all_slice_dict['Lasing_on']['slice_current'])/2.

    delta_E = all_slice_dict['Lasing_off']['slice_mean'] - all_slice_dict['Lasing_on']['slice_mean']
    delta_std_sq = all_slice_dict['Lasing_on']['slice_sigma'] - all_slice_dict['Lasing_off']['slice_sigma']
    np.clip(delta_std_sq, 0, None, out=delta_std_sq)

    power_from_Eloss = power_Eloss(mean_current, delta_E)
    E_total = np.trapz(power_from_Eloss, slice_time)
    power_from_Espread = power_Espread(slice_time, mean_current, delta_std_sq, pulse_energy)

    if debug:
        ms.figure('Lasing')
        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1
        sp_power = subplot(sp_ctr, title='Power')

        sp_ctr += 1
        sp_current = subplot(sp_ctr, title='Current')
        sp_ctr += 1
        sp_current.plot(slice_time, all_slice_dict['Lasing_off']['slice_current'], label='Off')
        sp_current.plot(slice_time, all_slice_dict['Lasing_on']['slice_current'], label='On')


        sp_power.plot(slice_time, power_from_Eloss)
        sp_power.plot(slice_time, power_from_Espread)
        plt.show()

    output = {
            'all_slice_dict': all_slice_dict,
            'power_Eloss': power_from_Eloss,
            'energy_Eloss': E_total,
            'power_Espread': power_from_Espread,
            'all_images': all_images,
            'current': mean_current,
            'slice_time': slice_time,
            }

    return output

class LasingReconstruction:
    def __init__(self, images_off, images_on, pulse_energy=None, current_cutoff=1e3, key_mean='slice_cut_mean', key_sigma='slice_cut_rms_sq', norm_factor=None):
        assert images_off.profile == images_on.profile
        self.images_off = images_off
        self.images_on = images_on
        self.current_cutoff = current_cutoff
        self.pulse_energy = pulse_energy
        self.key_mean = key_mean
        self.key_sigma = key_sigma
        self.norm_factor = norm_factor

        self.generate_all_slice_dict()
        #self.cap_rms(max_rms)
        self.calc_mean_slice_dict()
        self.lasing_analysis()

    def generate_all_slice_dict(self):
        self.all_slice_dict = {}
        for images, title, ls in [(self.images_off, 'Lasing Off', None), (self.images_on, 'Lasing On', '--')]:
            all_mean = np.zeros([len(images.images_tE), images.n_slices], dtype=float)
            all_sigma = all_mean.copy()
            all_x = all_mean.copy()
            all_current = all_mean.copy()

            for ctr, slice_dict in enumerate(images.slice_dicts):
                all_mean[ctr] = slice_dict[self.key_mean]
                all_sigma[ctr] = slice_dict[self.key_sigma]
                all_x[ctr] = slice_dict['slice_x']
                all_current[ctr] = slice_dict['slice_current']

            self.all_slice_dict[title] = {
                    'loss': all_mean,
                    'spread': all_sigma,
                    't': all_x,
                    'current': all_current,
                    }

    def calc_mean_slice_dict(self):
        mean_slice_dict = self.mean_slice_dict = {}
        for title in 'Lasing Off', 'Lasing On':
            mean_slice_dict[title] = {}
            for key, arr in self.all_slice_dict[title].items():
                mean_slice_dict[title][key] = {
                        'mean': np.nanmean(arr, axis=0),
                        'std': np.nanstd(arr, axis=0),
                        }

    def lasing_analysis(self):
        all_slice_dict = self.all_slice_dict
        mean_slice_dict = self.mean_slice_dict
        self.lasing_dict = lasing_dict = {}

        mean_current = (mean_slice_dict['Lasing On']['current']['mean']+mean_slice_dict['Lasing Off']['current']['mean'])/2.
        self.current_mask = mask = np.abs(mean_current) > self.current_cutoff
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

        lasing_dict['time'] = slice_time
        lasing_dict['Eloss'] = power_Eloss_err(slice_time, mean_current, on_loss_mean, off_loss_mean, err_current, on_loss_err, off_loss_err)
        lasing_dict['Espread'] = power_Espread_err(slice_time, mean_current, on_spread_mean, off_spread_mean, self.pulse_energy, err_current, on_spread_err, off_spread_err, norm_factor=self.norm_factor)
        self.norm_factor = norm_factor = lasing_dict['Espread']['norm_factor']
        print('norm_factor', '%.3e' % norm_factor, 'pulse energy Espread', '%.3e uJ' % (lasing_dict['Espread']['energy']*1e6), 'pulse energy Eloss', '%.3e uJ' % (lasing_dict['Eloss']['energy']*1e6))

        n_images = len(all_slice_dict['Lasing On']['t'])
        all_loss = np.zeros([n_images, mask.sum()])
        all_spread = all_loss.copy()

        for ctr in range(n_images):
            current = all_slice_dict['Lasing On']['current'][ctr, mask]
            mask2 = current < self.current_cutoff
            on_loss = all_slice_dict['Lasing On']['loss'][ctr,mask]
            on_spread = all_slice_dict['Lasing On']['spread'][ctr,mask]

            loss = off_loss_mean - on_loss
            power_loss = power_Eloss(mean_current, loss)
            power_loss[mask2] = 0
            all_loss[ctr] = power_loss

            sq_increase = on_spread - off_spread_mean
            power_spread = power_Espread(slice_time, current, sq_increase, self.pulse_energy, norm_factor=norm_factor)
            power_spread[mask2] = 0
            all_spread[ctr] = power_spread
        lasing_dict['all_Eloss'] = all_loss
        lasing_dict['all_Espread'] = all_spread

    def plot(self, plot_handles=None, figsize=None, n_shots=None):
        mask = self.current_mask

        if plot_handles is None:
            _, plot_handles = plot_results.lasing_figure(figsize=figsize)
        sp_image_on, sp_image_on2, sp_image_off, sp_slice_mean, sp_slice_sigma, sp_current, sp_lasing_loss, sp_lasing_spread, sp_orbit = plot_handles

        for sp_image_tE, sp_image_xy, obj in [
                (sp_image_on2, sp_image_on, self.images_on),
                (sp_image_off, None, self.images_off),
                ]:
            index_median = 0
            image_xy = obj.raw_image_objs[index_median]
            image_tE = obj.images_tE[index_median]
            if sp_image_xy is not None:
                image_xy.plot_img_and_proj(sp_image_xy)
            image_tE.plot_img_and_proj(sp_image_tE)

        current_center = []
        for title, ls, mean_color in [('Lasing Off', None, 'black'), ('Lasing On', '--', 'red')]:
            all_slice_dict = self.all_slice_dict[title]
            mean_slice_dict = self.mean_slice_dict[title]
            for ctr in range(len(all_slice_dict['t'])):
                xx_plot = all_slice_dict['t'][ctr,mask]
                sp_slice_mean.plot(xx_plot*1e15, all_slice_dict['loss'][ctr,mask]/1e6, ls=ls)
                sp_slice_sigma.plot(xx_plot*1e15, np.sqrt(all_slice_dict['spread'][ctr,mask])/1e6, ls=ls)

            mean_mean = mean_slice_dict['loss']['mean'][mask]
            mean_std = mean_slice_dict['loss']['std'][mask]
            sigma_mean = mean_slice_dict['spread']['mean'][mask]
            sigma_std = mean_slice_dict['spread']['std'][mask]
            current_mean = mean_slice_dict['current']['mean']
            current_std = mean_slice_dict['current']['std']
            #import pdb; pdb.set_trace()
            sp_slice_mean.errorbar(xx_plot*1e15, mean_mean/1e6, yerr=mean_std/1e6, color=mean_color, ls=ls, lw=3, label=title)
            _yy = np.sqrt(sigma_mean)
            _yy_err = sigma_std/(2*_yy)
            sp_slice_sigma.errorbar(xx_plot*1e15, _yy/1e6, yerr=_yy_err/1e6, color=mean_color, ls=ls, lw=3, label=title)
            sp_current.errorbar(mean_slice_dict['t']['mean']*1e15, current_mean/1e3, yerr=current_std/1e3, label=title, color=mean_color)
            current_center.append(np.sum(mean_slice_dict['t']['mean']*current_mean)/current_mean.sum())

        self.images_off.profile.plot_standard(sp_current, center_float=np.mean(current_center), label='Reconstructed')
        sp_current.axhline(self.current_cutoff/1e3, color='black', ls='--')
        sp_current.legend()
        sp_slice_mean.legend()
        sp_slice_sigma.legend()

        lasing_dict = self.lasing_dict

        for key, sp in [('all_Eloss', sp_lasing_loss), ('all_Espread', sp_lasing_spread)]:
            if n_shots is None:
                n_shots = len(lasing_dict[key])
            for n_shot, y_arr in enumerate(lasing_dict[key]):
                if n_shot > len(lasing_dict[key]) - n_shots - 1:
                    sp.plot(lasing_dict['time']*1e15, y_arr/1e9, ls='--')

        for key, label, sp in [
                ('Eloss', '$\Delta E$', sp_lasing_loss),
                ('Espread', r'$\Delta \langle E^2 \rangle$', sp_lasing_spread)]:
            xx_plot = lasing_dict[key]['time']*1e15
            #yy_plot = lasing_dict[key]['power']/1e9
            #yy_err = lasing_dict[key]['power_err']/1e9
            #sp.errorbar(xx_plot, yy_plot, yerr=yy_err, label=label, color='red', lw=3)
            yy_plot = np.nanmean(lasing_dict['all_'+key], axis=0)/1e9
            yy_err = np.nanstd(lasing_dict['all_'+key], axis=0)/1e9
            sp.errorbar(xx_plot, yy_plot, yerr=yy_err, color='black')

        #sp_lasing_loss.legend()
        #sp_lasing_spread.legend()

        for label, recon_image in [('Lasing On', self.images_on), ('Lasing Off', self.images_off)]:
            delta_distance = recon_image.beam_positions - (-recon_image.streaker_center)
            mean_x = np.array([abs(x.mean()) for x in recon_image.meas_screens])
            sp_orbit.scatter(mean_x*1e3, delta_distance*1e6, label=label)
        sp_orbit.legend()


class LasingReconstructionImages:
    def __init__(self, tracker, lasing_options, profile=None, ref_slice_dict=None, ref_y=None):
        self.tracker = tracker
        self.charge = tracker.total_charge
        self.profile = profile

        self.subtract_quantile = lasing_options['subtract_quantile']
        self.noise_cut = lasing_options['noise_cut']
        self.slice_factor = lasing_options['slice_factor']

        self.ref_slice_dict = None
        self.ref_y = ref_y

        self.do_recon_plot = False
        self.beam_positions = None
        self.index_median = None

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

    def add_file(self, filename):
        data_dict = h5_storage.loadH5Recursive(filename)
        self.add_dict(data_dict)

    def add_dict(self, data_dict, max_index=None):
        meta_data = data_dict['meta_data_begin']
        self.tracker.meta_data = meta_data
        images = data_dict['pyscan_result']['image'].astype(float)
        x_axis = data_dict['pyscan_result']['x_axis_m'].astype(float)
        y_axis = data_dict['pyscan_result']['y_axis_m'].astype(float)
        self.add_images(meta_data, images, x_axis, y_axis, max_index)

    def add_images(self, meta_data, images, x_axis, y_axis, max_index=None):
        self.meta_data = meta_data
        self.gap = self.tracker.structure_gap
        self.x_axis0 = x_axis
        self.x_axis = x_axis - self.tracker.calib.screen_center
        self.y_axis = y_axis
        self.raw_images = images
        self.raw_image_objs = []
        self.meas_screens = []
        rms_arr = []
        for n_image, img in enumerate(images):
            if max_index is not None and n_image >= max_index:
                break
            img = img - np.quantile(img, self.subtract_quantile)
            img[img < 0] = 0
            image = image_analysis.Image(img, self.x_axis, y_axis)
            self.raw_image_objs.append(image)
            screen = beam_profile.ScreenDistribution(image.x_axis, image.image.sum(axis=-2), total_charge=self.charge)
            self.meas_screens.append(screen)
            rms_arr.append(screen.rms())
        self.median_meas_screen_index = np.argsort(np.array(rms_arr))[len(self.meas_screens)//2]

    def get_current_profiles(self, blmeas_file=None):
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
            offset_dict = self.tracker.find_beam_position(self.tracker.beam_position, meas_screen, self.profile)
            position_dicts.append(offset_dict)
            beam_positions.append(offset_dict['beam_offset'])
        self.beam_positions = np.array(beam_positions)
        return position_dicts

    def calc_wake(self, beam_position=None):
        r12 = self.tracker.r12
        wake_dict = self.tracker.calc_wake(self.profile, 'Dipole', force_beam_position=beam_position)
        wake_t = wake_dict['wake_time']
        wake_x = wake_dict['wake_potential'] / self.tracker.energy_eV * r12
        return wake_t, wake_x

    def convert_axes(self):
        #print('convert_axes, self.ref_y', self.ref_y)
        dispersion = self.tracker.calcDisp()[self.n_streaker]
        self.dispersion = dispersion
        self.images_tE = []
        self.ref_y_list = []
        self.cut_images = []
        x_min, x_max = self.wake_x.min(), self.wake_x.max()

        images_E = []
        ref_y0 = self.ref_y
        for ctr, img in enumerate(self.raw_image_objs):
            image_E, ref_y = img.y_to_eV(dispersion, self.tracker.energy_eV, ref_y=ref_y0)
            #if ref_y0 is None:
            #    ref_y0 = 100000
            #print(ctr, int(ref_y0*1e6), int(ref_y*1e6))
            if ctr == 0:
                ref_y0 = ref_y
            images_E.append(image_E)
            self.ref_y_list.append(ref_y)

        for ctr, img in enumerate(images_E):
            if self.beam_positions is None:
                img_cut = img.cut(x_min, x_max)
                img_tE = img_cut.x_to_t(self.wake_x, self.wake_t, debug=False)
            else:
                wake_t, wake_x = self.calc_wake(self.beam_positions[ctr])
                img_cut = img.cut(wake_x.min(), wake_x.max())
                img_tE = img_cut.x_to_t(wake_x, wake_t, debug=False)
            self.images_tE.append(img_tE)
            self.cut_images.append(img_cut)

    def slice_x(self):
        if self.slice_factor == 1:
            self.images_sliced = self.images_tE
        else:
            self.images_sliced = []
            for n_image, image in enumerate(self.images_tE):
                n_slices = len(image.x_axis)//self.slice_factor
                image_sliced = image.slice_x(n_slices)
                self.images_sliced.append(image_sliced)

    def fit_slice(self):
        self.slice_dicts = []
        for image in self.images_sliced:
            slice_dict = image.fit_slice(charge=self.charge, noise_cut=self.noise_cut)
            self.slice_dicts.append(slice_dict)

    def interpolate_slice(self, ref):
        new_slice_dicts = []
        for slice_dict in self.slice_dicts:
            new_slice_dicts.append(interpolate_slice_dicts(ref, slice_dict))
        self.slice_dicts_old = self.slice_dicts
        self.slice_dicts = new_slice_dicts

    def process_data(self):
        if self.profile is None:
            self.get_current_profiles()
            self.set_profile()

        self.wake_t, self.wake_x = self.calc_wake()
        self.get_streaker_offsets()
        self.convert_axes()
        self.slice_x()
        self.fit_slice()
        if self.index_median is not None:
            self.ref_slice_dict = self.slice_dicts[self.index_median]
        if self.ref_slice_dict is not None:
            self.interpolate_slice(self.ref_slice_dict)

    def plot_images(self, type_, title='', **kwargs):
        if type_ == 'raw':
            images = self.raw_image_objs
        elif type_ == 'cut':
            images = self.cut_images
        elif type_ == 'tE':
            images = self.images_tE
        elif type_ == 'slice':
            images = self.images_sliced

        sp_ctr = np.inf
        ny, nx = 3, 3
        subplot = ms.subplot_factory(ny, nx, grid=False)

        figs = []
        subplots = []
        for n_image, image in enumerate(images):
            if sp_ctr > ny*nx:
                fig = ms.figure('%s Images %s' % (title, type_))
                figs.append(fig)
                this_subplots = []
                subplots.append(this_subplots)
                sp_ctr = 1
            sp = subplot(sp_ctr, title='Image %i' % n_image, xlabel=image.xlabel, ylabel=image.ylabel)
            sp_ctr += 1
            this_subplots.append(sp)
            slice_dict = None
            if type_ in ('tE', 'slice') and hasattr(self, 'slice_dicts'):
                slice_dict = self.slice_dicts[n_image]
            image.plot_img_and_proj(sp, slice_dict=slice_dict, **kwargs)
        return figs, subplots

def interpolate_slice_dicts(ref, alter):
    new_dict = {}
    xx_ref = ref['slice_x']
    xx_alter = alter['slice_x']
    for key, arr in alter.items():
        if key == 'slice_x':
            new_dict[key] = xx_ref
        elif type(arr) is np.ndarray:
            new_dict[key] = np.interp(xx_ref, xx_alter, arr)
    return new_dict


