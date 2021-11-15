import numpy as np

from . import h5_storage
from . import beam_profile
from . import myplotstyle as ms
from . import image_analysis

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

        self.mean_current = mean_current = (mean_slice_dict['Lasing On']['current']['mean']+mean_slice_dict['Lasing Off']['current']['mean'])/2.
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

    def get_result_dict(self):
        outp = {
                'lasing_dict': self.lasing_dict,
                'all_slice_dict': self.all_slice_dict,
                'mean_slice_dict': self.mean_slice_dict,
                'current_cutoff': self.current_cutoff,
                'mean_current': self.mean_current,
                }
        for key, obj in [('images_on', self.images_on), ('images_off', self.images_off)]:
            outp[key] = d = {}
            d['raw_images'] = obj.raw_image_objs
            d['tE_images'] = obj.images_tE
            d['current_profile'] = obj.profile
            d['beam_positions'] = obj.beam_positions
            d['delta_distances'] = obj.delta_distances
            d['meas_screen_centroids'] = obj.meas_screen_centroids
        return outp

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
        self.meas_screen_centroids = np.array([abs(x.mean()) for x in self.meas_screens])

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
            position_dict = self.tracker.find_beam_position(self.tracker.beam_position, meas_screen, self.profile)
            position_dicts.append(position_dict)
            beam_positions.append(position_dict['beam_position'])
        self.beam_positions = np.array(beam_positions)
        self.delta_distances = self.beam_positions - self.tracker.beam_position
        return position_dicts

    def calc_wake(self, beam_position=None):
        r12 = self.tracker.r12
        wake_dict = self.tracker.calc_wake(self.profile, 'Dipole', force_beam_position=beam_position)
        wake_t = wake_dict['wake_time']
        wake_x = wake_dict['wake_potential'] / self.tracker.energy_eV * r12
        return wake_t, wake_x

    def convert_axes(self):
        #print('convert_axes, self.ref_y', self.ref_y)
        dispersion = self.tracker.disp
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


