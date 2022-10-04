import copy
import numpy as np

from . import h5_storage
from . import beam_profile
from . import myplotstyle as ms
from . import image_analysis

espread_current_exponent = 2./3.

def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
    #power[power<0] = 0
    return power

def power_Eloss_err(slice_time, slice_current, slice_E_on, slice_E_off, slice_current_err, slice_E_on_err, slice_E_off_err):
    delta_E = slice_E_off-slice_E_on
    power = slice_current * delta_E
    #power[power<0] = 0
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

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total, photon_energy_factors=1, norm_factor=None, t_lims=None):
    power0 = slice_current**espread_current_exponent * slice_Espread_sqr_increase * photon_energy_factors
    if norm_factor is None:
        if t_lims:
            mask = np.logical_and(slice_t > t_lims[0], slice_t < t_lims[1])
        else:
            mask = np.ones_like(slice_t, dtype=bool)
        integral = np.trapz(power0[mask], slice_t[mask])
        power = power0/integral*E_total
    else:
        power = power0*norm_factor
    return power

def power_Espread_err(slice_t, slice_current, slice_Espread_on_sq, slice_Espread_off_sq, E_total, slice_current_err, slice_Espread_on_sq_err, slice_Espread_off_sq_err, photon_energy_factors=1, norm_factor=None, t_lims=None):
    """
    Takes squared values of energy spread
    """
    exp = espread_current_exponent
    slice_Espread_sqr_increase = slice_Espread_on_sq - slice_Espread_off_sq
    power0 = slice_current**exp * slice_Espread_sqr_increase * photon_energy_factors
    #power0[power0 < 0] = 0
    if t_lims:
        mask = np.logical_and(slice_t > t_lims[0], slice_t < t_lims[1])
    else:
        mask = np.ones_like(slice_t, dtype=bool)
    mask_power = power0 > 0
    mask = np.logical_and(mask_power, mask)
    integral = np.trapz(power0[mask], slice_t[mask])

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

def obtain_lasing(tracker, file_or_dict_off, file_or_dict_on, lasing_options, pulse_energy, norm_factor=None, slice_method=None, t_lims=None):
    if type(file_or_dict_off) is dict:
        lasing_off_dict = file_or_dict_off
    else:
        lasing_off_dict = h5_storage.loadH5Recursive(file_or_dict_off)
    if type(file_or_dict_on) is dict:
        lasing_on_dict = file_or_dict_on
    else:
        lasing_on_dict = h5_storage.loadH5Recursive(file_or_dict_on)
    current_cutoff = lasing_options['current_cutoff']
    las_rec_images = {}
    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        if main_ctr == 0:
            ref_y = None
        else:
            ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        rec_obj = LasingReconstructionImages(tracker, lasing_options, ref_y=ref_y)
        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
            rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        rec_obj.process_data()
        las_rec_images[title] = rec_obj

    linear_conversion = (lasing_options['x_conversion'] == 'linear')
    las_rec = LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], linear_conversion, pulse_energy, current_cutoff, slice_method=slice_method)
    if t_lims:
        tt = las_rec.mean_slice_dict['Lasing Off']['t']['mean']
        photon_energy_factors = np.logical_and(tt > t_lims[0], tt < t_lims[1]).astype(float)
        import pdb; pdb.set_trace()
    else:
        photon_energy_factors = None
    las_rec.lasing_analysis(photon_energy_factors=photon_energy_factors, norm_factor=norm_factor)
    result_dict = las_rec.get_result_dict()
    outp = {
            'las_rec': las_rec,
            'result_dict': result_dict,
            'las_rec_images': las_rec_images,
            }
    return outp

class LasingReconstruction:
    def __init__(self, images_off, images_on, linear_conversion, pulse_energy=None, current_cutoff=1e3, slice_method=None):
        assert images_off.profile == images_on.profile
        self.images_off = images_off
        self.images_on = images_on
        self.linear_conversion = linear_conversion
        self.current_cutoff = current_cutoff
        self.pulse_energy = pulse_energy
        if slice_method is None:
            self.slice_method = 'cut'
        else:
            self.slice_method = slice_method

        self.generate_all_slice_dict()
        self.calc_mean_slice_dict()

    def generate_all_slice_dict(self):
        self.all_slice_dict = {}
        for images, title, ls in [(self.images_off, 'Lasing Off', None), (self.images_on, 'Lasing On', '--')]:
            all_mean = np.zeros([len(images.images_tE), images.n_slices], dtype=float)
            all_sigma = all_mean.copy()
            all_x = all_mean.copy()
            all_current = all_mean.copy()

            for ctr, slice_dict in enumerate(images.slice_dicts):
                all_mean[ctr] = slice_dict[self.slice_method]['mean']
                all_sigma[ctr] = slice_dict[self.slice_method]['sigma_sq']
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

    def lasing_analysis(self, photon_energy_factors=1, norm_factor=None, t_lims=None):
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
        try:
            lasing_dict['Espread'] = power_Espread_err(slice_time, mean_current, on_spread_mean, off_spread_mean, self.pulse_energy, err_current, on_spread_err, off_spread_err, photon_energy_factors, norm_factor=norm_factor, t_lims=t_lims)
        except:
            import pdb; pdb.set_trace()
        lasing_dict['norm_factor'] = norm_factor = lasing_dict['Espread']['norm_factor']
        lasing_dict['photon_energy_factors'] = photon_energy_factors
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
            power_spread = power_Espread(slice_time, current, sq_increase, None, photon_energy_factors, norm_factor=norm_factor)
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
                'linear_conversion': int(self.linear_conversion),
                }
        for key, obj in [('images_on', self.images_on), ('images_off', self.images_off)]:
            outp[key] = d = {}
            d['raw_images'] = obj.images_xy
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
        self.max_quantile = lasing_options['max_quantile']
        self.slice_factor = lasing_options['slice_factor']
        self.x_conversion = lasing_options['x_conversion']
        self.x_factor = lasing_options['x_linear_factor']
        self.rms_sigma = lasing_options['rms_sigma']
        self.current_cutoff = lasing_options['current_cutoff']

        self.ref_slice_dict = None
        self.ref_y = ref_y

        self.do_recon_plot = False
        self.beam_positions = None
        self.index_median = None
        self.delta_distances = None

        if tracker.structure.dim == 'Y':
            raise ValueError('Structure streaking dimension Y not supported for lasing reconstruction!')

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
        images = data_dict['pyscan_result']['image'].astype(np.float64)
        x_axis = data_dict['pyscan_result']['x_axis_m'].astype(np.float64)
        y_axis = data_dict['pyscan_result']['y_axis_m'].astype(np.float64)
        self.add_images(meta_data, images, x_axis, y_axis, max_index)

    def add_images(self, meta_data, images, x_axis, y_axis, max_index=None):
        self.meta_data = meta_data
        self.gap = self.tracker.structure_gap
        self.x_axis0 = x_axis
        self.x_axis = x_axis - self.tracker.calib.screen_center
        self.y_axis = y_axis
        self.raw_images = images
        self.images_xy = []
        self.meas_screens = []
        rms_arr = []
        for n_image, img in enumerate(images):
            if max_index is not None and n_image >= max_index:
                break
            img = img - np.quantile(img, self.subtract_quantile)
            if self.max_quantile is not None:
                img = img.clip(0, np.quantile(img, self.max_quantile))
            else:
                img = img.clip(0, None)
            image = image_analysis.Image(img, self.x_axis, y_axis, self.charge)
            self.images_xy.append(image)
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
        self.average_distance = self.gap/2. - abs(self.beam_positions.mean())
        return position_dicts

    def calc_wake(self, beam_position=None):
        r12 = self.tracker.r12
        profile2 = copy.deepcopy(self.profile)
        profile2.expand(0.3)
        wake_dict = self.tracker.calc_wake(profile2, 'Dipole', force_beam_position=beam_position)
        wake_t = wake_dict['wake_time']
        wake_x = wake_dict['wake_potential'] / self.tracker.energy_eV * r12
        return wake_t, wake_x

    def convert_x_wake(self):
        self.images_tE = []
        self.cut_images = []
        x_min, x_max = self.wake_x.min(), self.wake_x.max()
        for ctr, img in enumerate(self.images_E):
            if self.beam_positions is None:
                img_cut = img.cut(x_min, x_max)
                img_tE = img_cut.x_to_t(self.wake_x, self.wake_t, debug=False)
            else:
                wake_t, wake_x = self.calc_wake(self.beam_positions[ctr])
                img_cut = img.cut(wake_x.min(), wake_x.max())
                img_tE = img_cut.x_to_t(wake_x, wake_t, debug=False)
            self.images_tE.append(img_tE)
            self.cut_images.append(img_cut)

    def convert_x_linear(self, factor):
        #self.cut_images = []
        #for ctr, img in enumerate(self.images_E):
        #    tt = img.x_axis*factor
        #    cc = np.sum(img.image, axis=0)
        #    current = cc*self.charge/np.trapz(cc, tt)
        #    mask = current > self.current_cutoff
        #    new_x = img.x_axis[mask]
        #    new_img = img.image[:,mask]
        #    cut_image = img.child(new_img, new_x, img.y_axis)
        #    self.cut_images.append(cut_image)

        self.images_tE = []
        for ctr, img in enumerate(self.images_E):
            new_img = img.x_to_t_linear(factor, mean_to_zero=True)
            self.images_tE.append(new_img)

    def convert_y(self):
        self.ref_y_list = []
        self.dispersion = self.tracker.disp
        self.images_E = []
        ref_y0 = self.ref_y
        for ctr, img in enumerate(self.images_xy):
            image_E, ref_y = img.y_to_eV(self.dispersion, self.tracker.energy_eV, ref_y=ref_y0)
            if ctr == 0:
                ref_y0 = ref_y
            self.images_E.append(image_E)
            self.ref_y_list.append(ref_y)

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
            if self.x_conversion == 'linear':
                current_cutoff = self.current_cutoff
            else:
                current_cutoff = None
            slice_dict = image.fit_slice(rms_sigma=self.rms_sigma, current_cutoff=current_cutoff)
            self.slice_dicts.append(slice_dict)

    def interpolate_slice(self, ref):
        new_slice_dicts = []
        for slice_dict in self.slice_dicts:
            new_slice_dicts.append(interpolate_slice_dicts(ref, slice_dict))
        self.slice_dicts_old = self.slice_dicts
        self.slice_dicts = new_slice_dicts

    def process_data(self):
        self.convert_y()
        if self.x_conversion == 'wake':
            if self.profile is None:
                self.get_current_profiles()
                self.set_profile()
            self.wake_t, self.wake_x = self.calc_wake()
            self.get_streaker_offsets()
            self.convert_x_wake()
        elif self.x_conversion == 'linear':
            self.convert_x_linear(self.x_factor)
        else:
            raise ValueError(self.x_conversion)
        self.slice_x()
        self.fit_slice()
        if self.x_conversion == 'wake':
            if self.index_median is not None:
                self.ref_slice_dict = self.slice_dicts[self.index_median]
            if self.ref_slice_dict is not None:
                self.interpolate_slice(self.ref_slice_dict)
        elif self.x_conversion == 'linear':
            self.ref_slice_dict = self.slice_dicts[self.median_meas_screen_index]
            self.interpolate_slice(self.ref_slice_dict)

    def plot_images(self, type_, title='', **kwargs):
        if type_ == 'raw':
            images = self.images_xy
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
            new_dict[key] = np.interp(xx_ref, xx_alter, arr, left=0, right=0)
        elif type(arr) is dict:
            new_dict[key] = {}
            for key2, arr2 in arr.items():
                if type(arr2) is np.ndarray:
                    new_dict[key][key2] = np.interp(xx_ref, xx_alter, arr2, left=0, right=0)
    return new_dict

