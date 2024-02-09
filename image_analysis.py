import time
import operator
import numpy as np
import scipy
import numba
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt

from .gaussfit import GaussFit
from . import beam_profile
from . import myplotstyle as ms
from .logMsg import LogMsgBase


class Image(LogMsgBase):
    def __init__(self, image, x_axis, y_axis, charge=1, energy_eV=1, x_unit='m', y_unit='m', subtract_median=False, xlabel='x (mm)', ylabel='y (mm)', logger=None, slice_dict=None):
        self.logger = logger
        if x_axis.size <=1:
            raise ValueError('Size of x_axis is %i' % x_axis.size)

        if image.shape[0] != y_axis.size or image.shape[1] != x_axis.size:
            raise ValueError('Wrong shapes!', image.shape, y_axis.size, x_axis.size)

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            image = image[:,::-1]
        if y_axis[1] < y_axis[0]:
            y_axis = y_axis[::-1]
            image = image[::-1,:]
        if np.any(np.diff(y_axis) == 0):
            raise ValueError
        if np.any(np.diff(x_axis) == 0):
            raise ValueError
        if subtract_median:
            image = image - np.median(image)
            np.clip(image, 0, None, out=image)
        self.image = image
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.charge = charge
        self.energy_eV = energy_eV
        self.slice_dict = slice_dict

    def to_dict_custom(self):
        outp = {
                'image': self.image,
                'x_axis': self.x_axis,
                'y_axis': self.y_axis,
                'x_unit': self.x_unit,
                'y_unit': self.y_unit,
                'xlabel': self.xlabel,
                'ylabel': self.ylabel,
                'charge': self.charge,
                'energy_eV': self.energy_eV,
                'slice_dict': self.slice_dict,
                }
        return outp

    def child(self, new_i, new_x, new_y, x_unit=None, y_unit=None, xlabel=None, ylabel=None):
        x_unit = self.x_unit if x_unit is None else x_unit
        y_unit = self.y_unit if y_unit is None else y_unit
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        return Image(new_i, new_x, new_y, self.charge, self.energy_eV, x_unit, y_unit, xlabel=xlabel, ylabel=ylabel)

    def noisecut(self, noiselevel):
        new_image = self.image.copy()
        new_image[new_image < noiselevel] = 0
        return self.child(new_image, self.x_axis, self.y_axis)

    def mean(self, dimension):
        axis, proj = self._get_axis_and_proj(dimension)
        return np.sum(axis*proj)/np.sum(proj)

    def get_screen_dist(self, dimension, **kwargs):
        if dimension == 'X':
            axis = self.x_axis
            proj = self.image.sum(axis=0)
        elif dimension == 'Y':
            axis = self.y_axis
            proj = self.image.sum(axis=1)
        return beam_profile.ScreenDistribution(axis, proj, total_charge=self.charge, **kwargs)

    def get_profile(self, dimension='X'):
        if dimension == 'X':
            axis = self.x_axis
            proj = self.image.sum(axis=0)
        elif dimension == 'Y':
            axis = self.y_axis
            proj = self.image.sum(axis=1)
        return beam_profile.BeamProfile(axis, proj, self.energy_eV, self.charge)

    def rms(self, dimension):
        axis, proj = self._get_axis_and_proj(dimension)
        mean = self.mean(dimension)
        return np.sqrt(np.sum((axis-mean)**2*proj)/np.sum(proj))

    def transpose(self):
        return self.child(self.image.T.copy(), self.y_axis.copy(), self.x_axis.copy(), self.y_unit, self.x_unit, self.ylabel, self.xlabel)

    def invertX(self):
        return self.child(self.image.copy()[:,::-1], self.x_axis.copy(), self.y_axis.copy())

    def invertY(self):
        return self.child(self.image.copy()[::-1], self.x_axis.copy(), self.y_axis.copy())

    def cut(self, x_min, x_max):
        x_axis = self.x_axis
        x_mask = np.logical_and(x_axis >= x_min, x_axis <= x_max)
        new_image = self.image[:,x_mask]
        new_x_axis = x_axis[x_mask]
        return self.child(new_image, new_x_axis, self.y_axis)

    def cutY(self, y_min, y_max):
        y_axis = self.y_axis
        y_mask = np.logical_and(y_axis >= y_min, y_axis <= y_max)
        new_image = self.image[y_mask,:]
        new_y_axis = y_axis[y_mask]
        return self.child(new_image, self.x_axis, new_y_axis)

    def cut_voids(self, xcutoff, ycutoff):
        new_image = self
        for dim, cutoff, funcname in [
                ('X', xcutoff, 'cut'),
                ('Y', ycutoff, 'cutY'),
                ]:
            if cutoff is None:
                continue
            prof = new_image.get_screen_dist(dim, subtract_min=True)
            prof.aggressive_cutoff(cutoff)
            prof.crop()
            func = getattr(new_image, funcname)
            new_image = func(prof.x[0], prof.x[-1])
        return new_image

    def reshape_x(self, new_length):
        """
        If new length is larger than current length
        """
        image2 = np.zeros([len(self.y_axis), new_length])
        x_axis2 = np.linspace(self.x_axis.min(), self.x_axis.max(), new_length)
        # Fast interpolation
        delta_x = np.zeros_like(self.image)
        delta_x[:,:-1] = self.image[:,1:] - self.image[:,:-1]
        index_float = (x_axis2 - self.x_axis[0]) / (self.x_axis[1] - self.x_axis[0])
        index = index_float.astype(int)
        index_delta = index_float-index
        np.clip(index, 0, len(self.x_axis)-1, out=index)
        image2 = self.image[:, index] + index_delta * delta_x[:,index]
        image2 = image2 / image2.sum() * self.image.sum()
        return self.child(image2, x_axis2, self.y_axis)

    def slice_x(self, n_slices):
        x_axis, y_axis = self.x_axis, self.y_axis
        max_x_index = len(x_axis) - len(x_axis) % n_slices
        image_extra = np.reshape(self.image[:,:max_x_index], [len(y_axis), n_slices, max_x_index//n_slices])
        new_image = np.mean(image_extra, axis=-1)
        x_axis_reshaped = np.linspace(x_axis[0], x_axis[max_x_index-1], n_slices)
        output = self.child(new_image, x_axis_reshaped, y_axis)
        return output

    def fit_slice_simple(self, slice_cutoff=0.02, current_cutoff=None, E_lims=None, ref_t=None):
        y_axis = self.y_axis
        n_slices = len(self.x_axis)
        slice_full_mean = []
        slice_full_rms = []
        slice_cutoff_mean = []
        slice_cutoff_rms = []
        slice_cutoff_lim1 = []
        slice_cutoff_lim2 = []

        proj = np.sum(self.image, axis=-2)
        proj = proj / np.sum(proj) * abs(self.charge)
        current = proj / (self.x_axis[1] - self.x_axis[0])
        current -= current.min()
        slice_x = self.x_axis
        current = current / current.sum() * abs(self.charge) / (self.x_axis[1] - self.x_axis[0])

        if E_lims is not None:
            projY = np.sum(self.image, axis=-1)
            meanY = np.sum(y_axis*projY)/np.sum(projY)
            mask_Elim = np.logical_and(y_axis >= meanY+E_lims[0], y_axis <= meanY+E_lims[1])
        else:
            mask_Elim = np.ones_like(y_axis, dtype=bool)
        y_axis = y_axis[mask_Elim]

        def addzero():
            slice_full_mean.append(0)
            slice_full_rms.append(0)
            slice_cutoff_mean.append(0)
            slice_cutoff_rms.append(0)
            slice_cutoff_lim1.append(0)
            slice_cutoff_lim2.append(0)

        for n_slice in range(n_slices):

            if current_cutoff is not None and current[n_slice] < current_cutoff:
                addzero()
                continue

            intensity = self.image[mask_Elim,n_slice]
            intensity = intensity - intensity.min()

            if np.sum(intensity) == 0:
                addzero()
                continue

            mean_full, rms_full = calc_rms(y_axis, intensity)
            prof = beam_profile.AnyProfile(y_axis, intensity)
            prof.aggressive_cutoff(0.02, center='Mean')
            status = prof.crop(quiet=True)

            if (not status) or (len(prof) < 3):
                slice_cutoff_mean.append(0)
                slice_cutoff_rms.append(0)
                slice_cutoff_lim1.append(0)
                slice_cutoff_lim2.append(0)
            else:
                slice_cutoff_mean.append(prof.mean())
                slice_cutoff_rms.append(prof.rms())
                slice_cutoff_lim1.append(prof.xx[0])
                slice_cutoff_lim2.append(prof.xx[-1])

            slice_full_mean.append(mean_full)
            slice_full_rms.append(rms_full**2)

        slice_dict = {
                'slice_x': slice_x,
                'slice_intensity': proj,
                'slice_current': current,
                'y_axis_Elim': y_axis,
                'E_lims': E_lims,
                'full': {
                    'mean': np.array(slice_full_mean),
                    'sigma_sq': np.array(slice_full_rms),
                    },
                'cutoff': {
                    'mean': np.array(slice_cutoff_mean),
                    'sigma_sq': np.array(slice_cutoff_rms)**2,
                    'lim1': np.array(slice_cutoff_lim1),
                    'lim2': np.array(slice_cutoff_lim2),
                    },
                }
        diff_time = np.diff(slice_x)
        for key in ['full', 'cutoff']:
            slice_dict[key]['chirp'] = np.concatenate([np.diff(slice_dict[key]['mean'])/diff_time, [0.]])
            if ref_t is not None:
                eref = np.interp(ref_t, slice_dict['slice_x'], slice_dict[key]['mean'])
                slice_dict[key]['mean'] -= eref
                slice_dict[key]['eref'] = eref

        self.slice_dict = slice_dict
        return slice_dict

    def fit_slice(self, rms_sigma=5, current_cutoff=None, E_lims=None, do_plot=False, ref_t=None):
        y_axis = self.y_axis
        n_slices = len(self.x_axis)
        slice_mean = []
        slice_sigma = []
        slice_gf = []
        slice_rms = []
        slice_mean_rms = []
        slice_cut_mean = []
        slice_cut_rms = []
        slice_cut_lim1 = []
        slice_cut_lim2 = []
        slice_lima1 = []
        slice_lima2 = []
        slice_full_mean = []
        slice_full_rms = []

        proj = np.sum(self.image, axis=-2)
        proj = proj / np.sum(proj) * abs(self.charge)
        current = proj / (self.x_axis[1] - self.x_axis[0])
        current -= current.min()
        slice_x = self.x_axis
        current = current / current.sum() * abs(self.charge) / (self.x_axis[1] - self.x_axis[0])

        if E_lims is not None:
            projY = np.sum(self.image, axis=-1)
            meanY = np.sum(y_axis*projY)/np.sum(projY)
            mask_Elim = np.logical_and(y_axis >= meanY+E_lims[0], y_axis <= meanY+E_lims[1])
        else:
            mask_Elim = np.ones_like(y_axis, dtype=bool)
        y_axis = y_axis[mask_Elim]

        def addzero():
            slice_mean.append(0)
            slice_sigma.append(0)
            slice_gf.append(None)
            slice_rms.append(0)
            slice_mean_rms.append(0)
            slice_cut_rms.append(0)
            slice_cut_mean.append(0)
            slice_full_mean.append(0)
            slice_full_rms.append(0)
            slice_cut_lim1.append(0)
            slice_cut_lim2.append(0)
            slice_lima1.append(0)
            slice_lima2.append(0)

        sp_ctr = np.inf
        nx, ny = 4, 4
        subplot = ms.subplot_factory(ny, nx, False)
        if plt.get_fignums():
            fignum0 = ms.plt.gcf().number
        else:
            fignum0 = None

        image_sum = self.image.sum()

        for n_slice in range(n_slices):

            if current_cutoff is not None and current[n_slice] < current_cutoff:
                addzero()
                continue

            intensity = self.image[mask_Elim,n_slice]
            intensity = intensity - intensity.min()

            if np.sum(intensity) == 0:
                addzero()
                continue

            try:
                gf = GaussFit(y_axis, intensity, fit_const=True, raise_=True)
            except (RuntimeError, OptimizeWarning):
                gf = None
                addzero()
            else:
                mean_full, rms_full = calc_rms(y_axis, intensity)
                slice_full_mean.append(mean_full)
                slice_full_rms.append(rms_full**2)

                where_max = y_axis[np.argmax(gf.reconstruction)]
                lima1 = where_max - abs(gf.sigma)*rms_sigma
                lima2 = where_max + abs(gf.sigma)*rms_sigma
                slice_lima1.append(lima1)
                slice_lima2.append(lima2)
                mask_rms = np.logical_and(y_axis >= lima1, y_axis <= lima2)
                y_rms = y_axis[mask_rms]
                data_rms = intensity[mask_rms]
                if np.sum(mask_rms) <= 1 or np.sum(data_rms) == 0:
                    mean_rms, rms = 0, 0
                else:
                    mean_rms, rms = calc_rms(y_rms, data_rms)

                slice_gf.append(gf)
                slice_mean.append(gf.mean)
                slice_sigma.append(gf.sigma**2)
                slice_rms.append(rms**2)
                slice_mean_rms.append(mean_rms)

                prof_y = intensity.copy()
                lim1, lim2 = mean_rms-(rms_sigma/2)*rms, mean_rms+(rms_sigma/2)*rms
                slice_cut_lim1.append(lim1)
                slice_cut_lim2.append(lim2)
                mask_cut = np.logical_and(y_axis >= lim1, y_axis <= lim2)
                if np.sum(mask_cut) <= 1 or np.sum(prof_y[mask_cut]) == 0:
                    slice_cut_rms.append(0)
                    slice_cut_mean.append(0)
                else:
                    cut_mean, cut_rms = calc_rms(y_axis[mask_cut], prof_y[mask_cut])
                    slice_cut_rms.append(cut_rms**2)
                    slice_cut_mean.append(cut_mean)

            if do_plot:
                if sp_ctr >= nx*ny:
                    ms.figure('Plot slice analysis')
                    sp_ctr = 1
                intensity_ratio = np.sum(self.image[:,n_slice])/image_sum * n_slices
                sp = subplot(sp_ctr, title='Slice %i int. %.1f' % (n_slice, intensity_ratio), xlabel='y', ylabel='intensity', scix=True, sciy=True)
                sp_ctr += 1

                sp.plot(y_axis, intensity)
                if gf is not None:
                    sp.plot(gf.xx, gf.reconstruction, label='%.1e' % gf.sigma)
                    sp.axvline(lima1, color='black', ls='--', label='%.1e' % np.sqrt(slice_rms[-1]))
                    sp.axvline(lima2, color='black', ls='--')
                    sp.axvline(lim1, color='gray', ls='--', label='%.1e' % np.sqrt(slice_cut_rms[-1]))
                    sp.axvline(lim2, color='gray', ls='--')
                    sp.legend()

        if do_plot and fignum0 is not None:
            ms.plt.figure(fignum0)

        slice_dict = {
                'slice_x': slice_x,
                'slice_intensity': proj,
                'slice_current': current,
                'slice_gf': slice_gf,
                'y_axis_Elim': y_axis,
                'E_lims': E_lims,
                'gauss': {
                    'mean': np.array(slice_mean),
                    'sigma_sq': np.array(slice_sigma),
                    },
                'rms': {
                    'mean': np.array(slice_mean_rms),
                    'sigma_sq': np.array(slice_rms),
                    'lim1': np.array(slice_lima1),
                    'lim2': np.array(slice_lima2),
                    },
                'cut': {
                    'mean': np.array(slice_cut_mean),
                    'sigma_sq': np.array(slice_cut_rms),
                    'lim1': np.array(slice_cut_lim1),
                    'lim2': np.array(slice_cut_lim2),
                    },
                'full': {
                    'mean': np.array(slice_full_mean),
                    'sigma_sq': np.array(slice_full_rms),
                    },
                }
        diff_time = np.diff(slice_x)
        for key in ['gauss', 'rms', 'cut', 'full']:
            slice_dict[key]['chirp'] = np.concatenate([np.diff(slice_dict[key]['mean'])/diff_time, [0.]])
            if ref_t is not None:
                eref = np.interp(ref_t, slice_dict['slice_x'], slice_dict[key]['mean'])
                slice_dict[key]['mean'] -= eref
                slice_dict[key]['eref'] = eref

        self.slice_dict = slice_dict
        return slice_dict

    def y_to_eV(self, dispersion, energy_eV, ref_y=None):
        if ref_y is None:
            ref_y = GaussFit(self.y_axis, np.sum(self.image, axis=-1)).mean
        if dispersion == 0:
            raise ValueError
        E_axis = (self.y_axis-ref_y) / dispersion * energy_eV
        return self.child(self.image, self.x_axis, E_axis, y_unit='eV', ylabel='$\Delta$ E (MeV)'), ref_y

    def x_to_t_linear(self, factor, mean_to_zero=True, current_cutoff=None):
        new_x_axis = self.x_axis*factor
        proj = self.image.sum(axis=0)
        image = self.image
        if factor < 0:
            new_x_axis = new_x_axis[::-1]
            proj = proj[::-1]
            image = self.image[:,::-1]

        if mean_to_zero:
            profile = beam_profile.BeamProfile(new_x_axis, proj, 1, self.charge)
            if current_cutoff:
                cutoff = current_cutoff/(np.abs(profile.get_current()).max())
                profile.aggressive_cutoff(cutoff)
                profile.crop()
                if not np.any(profile._yy):
                    print('Warning! Current cutoff too radical!')
                    profile = beam_profile.BeamProfile(new_x_axis, proj, 1, self.charge)
            refx = profile.mean()
        else:
            refx = 0
        return self.child(image, new_x_axis-refx, self.y_axis, x_unit='s', xlabel='t (fs)')

    def x_to_t(self, wake_x, wake_time, debug=False, print_=False, current_profile=None, time_smoothing=1e-15, size_factor=10):
        if print_:
            t0 = time.time()

        diff = np.diff(wake_time)
        assert np.all(diff >= 0) or np.all(diff <= 0)
        if wake_x[1] < wake_x[0]:
            wake_x = wake_x[::-1]
            wake_time = wake_time[::-1]

        new_time_len = self.image.shape[1]*size_factor
        new_t_axis = np.linspace(wake_time.min(), wake_time.max(), new_time_len)
        new_t_axis_final = np.linspace(wake_time.min(), wake_time.max(), self.image.shape[1])
        new_arr = np.zeros([len(self.y_axis), len(new_t_axis)], float)

        indicesX = np.arange(len(self.x_axis))
        sign_wake = np.sign(np.mean(wake_x))
        indicesT = np.arange(new_time_len)
        indices2 = np.interp(wake_time, new_t_axis, indicesT)

        for op in operator.lt, operator.gt:
            mask = op(self.x_axis, 0)
            x_axis = self.x_axis[mask]
            indices = indicesX[mask]
            if np.sign(np.mean(x_axis)) != sign_wake:
                x_axis = -x_axis[::-1]
                indices = indices[::-1]
            _x_to_t_inner(self.image, indices, indices2, x_axis, wake_x, new_arr)

        new_arr[:,0] = new_arr[:,-1] = 0
        new_arr *= self.image.sum()/new_arr.sum()
        new_arr = np.reshape(new_arr, [self.image.shape[0], self.image.shape[1], size_factor]).sum(axis=-1)

        if time_smoothing is not None:
            nconv = int(time_smoothing / (new_t_axis_final[1] - new_t_axis_final[0]))
            kernel = np.ones(nconv)/nconv
            new_arr1 = scipy.ndimage.correlate1d(new_arr, kernel, axis=1)
        else:
            new_arr1 = new_arr
        new_img = self.child(new_arr1, new_t_axis_final, self.y_axis, xlabel='t (fs)', x_unit='s')

        if debug:

            old_img = self.old_x_to_t(wake_x, wake_time)

            fig = ms.figure('Test new image analysis')
            fig.subplots_adjust(hspace=0.35)

            subplot = ms.subplot_factory(3, 3, False)
            sp_ctr = 1

            sp_current = subplot(sp_ctr, title='Current profiles', xlabel='t (fs)', ylabel='I (kA)')
            sp_ctr += 1

            if current_profile:
                current_profile.plot_standard(sp_current, color='black', label='I (t)')
                charge = current_profile.total_charge
            else:
                charge = self.charge

            sp_y = subplot(sp_ctr, title='Y projections', xlabel='y (mm)', ylabel='Intensity (arb. units)')

            sp_ctr += 1

            for img, label in [
                    (new_img, 'New output'),
                    (self, 'Input'),
                    (old_img, 'Old output'),
                    ]:
                sp = subplot(sp_ctr, title=label, xlabel='x (mm)', ylabel=img.ylabel)
                sp_ctr += 1
                self.plot_img_and_proj(sp)
                if img.x_unit == 's':
                    profile = beam_profile.BeamProfile(img.x_axis, img.image.sum(axis=0), self.energy_eV, charge)
                    profile.plot_standard(sp_current, label=label)

                dist = img.get_screen_dist('Y')
                dist.total_charge = charge
                sp_y.plot(dist.x, dist.intensity, label=label)

            sp_current.legend()
            sp_y.legend()
        if print_:
            print(time.time()-t0)
        return new_img

    def old_x_to_t(self, wake_x, wake_time, debug=False, print_=False, current_profile=None):
        diff = np.diff(wake_time)
        assert np.all(diff >= 0) or np.all(diff <= 0)
        if wake_time[1] < wake_time[0]:
            wake_x = wake_x[::-1]
            wake_time = wake_time[::-1]

        assert wake_x[0] == 0

        new_img1 = np.zeros_like(self.image)
        new_t_axis = np.linspace(wake_time.min(), wake_time.max(), self.image.shape[1])
        x_interp = np.interp(new_t_axis, wake_time, wake_x)

        old_proj = np.sum(self.image, axis=0)
        old_proj_cumsum = np.cumsum(old_proj)

        # Make sure the piecewise integrals of the Intensity profiles match when the axis is converted.
        if self.x_axis[1] > self.x_axis[0]:
            wake_x_interp = self.x_axis
            proj_cumsum_interp = old_proj_cumsum
        else:
            wake_x_interp = self.x_axis[::-1]
            proj_cumsum_interp = old_proj_cumsum[::-1]

        cumsum0 = np.interp(0, wake_x_interp, proj_cumsum_interp)
        max_x = wake_x[np.argmax(np.abs(wake_x)).squeeze()]
        cumsum1 = np.interp(max_x, wake_x_interp, proj_cumsum_interp)
        full_intensity = cumsum1 - cumsum0

        to_print = []
        x_indices = []
        prev_cumsum = None
        for t_index, (t, x) in enumerate(zip(new_t_axis, x_interp)):
            x_index = np.argmin((self.x_axis - x)**2)
            x_indices.append(x_index)
            cumsum = np.interp(x, wake_x_interp, proj_cumsum_interp)
            if prev_cumsum is None:
                delta_cumsum = 0
            else:
                delta_cumsum = cumsum - prev_cumsum
            # Normalize such that I(x) * dx = I(t) * dt
            sum_slice = np.sum(self.image[:,x_index])
            if (delta_cumsum == 0 and full_intensity == 0) or sum_slice == 0:
                new_img1[:,t_index] = 0
            else:
                new_img1[:,t_index] = self.image[:,x_index] / sum_slice * delta_cumsum / full_intensity
            if np.any(np.isnan(new_img1[:,t_index])):
                raise ValueError
                #import pdb; pdb.set_trace()
            prev_cumsum = cumsum

            if print_:
                to_print.append('%i %i %.1f %.1f' % (t_index, x_index, t*1e15, x*1e6))
        if print_:
            print('\n'.join(to_print))

        output = self.child(new_img1, new_t_axis, self.y_axis, x_unit='s', xlabel='t (fs)')
        output.x_indices = x_indices

        if debug:
            fig = ms.figure('Debug x to t'); fig
            subplot = ms.subplot_factory(2,3)
            sp_ctr = 1

            sp = subplot(sp_ctr, title='Wake', xlabel='time [fs]', ylabel='Screen x [mm]')
            sp_ctr += 1
            sp.plot(wake_time*1e15, wake_x*1e3)
            if current_profile is not None:
                sp2 = sp.twinx()
                sp2.plot(current_profile.time*1e15, current_profile.charge_dist, color='black')
                sp2.set_ylabel('Profile (arb. units)')

            sp = subplot(sp_ctr, title='Image projection X', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(self.x_axis*1e3, self.image.sum(axis=-2))

            sp = subplot(sp_ctr, title='Image projection T', xlabel='t [fs]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(output.x_axis*1e15, output.image.sum(axis=-2))
            if current_profile is not None:
                sp2 = sp.twinx()
                sp2.plot(current_profile.time*1e15, current_profile.charge_dist, color='black')
                sp2.set_ylabel('Profile (arb. units)')

            sp = subplot(sp_ctr, title='Image old', xlabel='x [mm]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            self.plot_img_and_proj(sp)

            sp = subplot(sp_ctr, title='Image new', xlabel='t [fs]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            try:
                output.plot_img_and_proj(sp)
            except:
                print('Cannot plot output')
            #fig.savefig('/tmp/debug_fig.pdf')
        return output

    def plot_img_and_proj(self, sp, x_factor=None, y_factor=None, plot_proj=True, log=False, revert_x=False, plot_gauss=True, slice_dict=None, xlim=None, ylim=None, cmapname='hot', slice_cutoff=0, gauss_color=('orange', 'orange'), proj_color=('green', 'green'), slice_color='deepskyblue', slice_method='cut', plot_gauss_x=False, plot_gauss_y=False, plot_proj_x=False, plot_proj_y=False, gauss_alpha=None, cut_intensity_quantile=None, hlines=None, hline_color='deepskyblue', vlines=None, vline_color='deepskyblue', sqrt=False, plot_slice_lims=False):

        def unit_to_factor(unit):
            if unit == 'm':
                factor = 1e3
            elif unit == 's':
                factor = 1e15
            elif unit == 'eV':
                factor = 1e-6
            else:
                factor = 1
            return factor

        if x_factor is None:
            x_factor = unit_to_factor(self.x_unit)
        if y_factor is None:
            y_factor = unit_to_factor(self.y_unit)

        x_axis, y_axis, image = self.x_axis, self.y_axis, self.image
        if xlim is None:
            index_x_min, index_x_max = 0, len(x_axis)
        else:
            index_x_min, index_x_max = sorted([np.argmin((x_axis-xlim[1])**2), np.argmin((x_axis-xlim[0])**2)])
        if ylim is None:
            index_y_min, index_y_max = 0, len(y_axis)
        else:
            index_y_min, index_y_max = sorted([np.argmin((y_axis-ylim[1])**2), np.argmin((y_axis-ylim[0])**2)])

        x_axis = x_axis[index_x_min:index_x_max]
        y_axis = y_axis[index_y_min:index_y_max]
        image = image[index_y_min:index_y_max,index_x_min:index_x_max]

        extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[0]*y_factor, y_axis[-1]*y_factor]

        if cut_intensity_quantile:
            image = np.clip(image, 0, np.quantile(image, cut_intensity_quantile))

        if log:
            image_ = np.clip(image, 1, None)
            log = np.log(image_)
        elif sqrt:
            log = np.sqrt(image)
        else:
            log = image

        sp.imshow(log, aspect='auto', extent=extent, origin='lower', cmap=plt.get_cmap(cmapname))

        if slice_dict is not None:
            old_lim = sp.get_xlim(), sp.get_ylim()
            mask = slice_dict['slice_current'] > slice_cutoff
            xx = slice_dict['slice_x'][mask]*x_factor
            yy = slice_dict[slice_method]['mean'][mask]*y_factor
            if 'eref' in slice_dict:
                yy = yy + slice_dict[slice_method]['eref']*y_factor
            yy_err = np.sqrt(slice_dict[slice_method]['sigma_sq'][mask])*y_factor
            sp.errorbar(xx, yy, yerr=yy_err, color=slice_color, ls='None', marker='None', lw=1)

            if plot_slice_lims and slice_method in ('cut', 'rms'):
                lim1 = slice_dict[slice_method]['lim1'][mask]*y_factor
                lim2 = slice_dict[slice_method]['lim2'][mask]*y_factor
                if 'eref' in slice_dict:
                    lim1 = lim1 + slice_dict[slice_method]['eref']*y_factor
                    lim2 = lim2 + slice_dict[slice_method]['eref']*y_factor
                sp.plot(xx, lim1, color='yellow')
                sp.plot(xx, lim2, color='yellow')
            sp.set_xlim(*old_lim[0])
            sp.set_ylim(*old_lim[1])


        gf_y = gf_x = None
        if plot_proj or plot_proj_x:
            proj = image.sum(axis=-2)
            proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*y_factor
            sp.plot(x_axis*x_factor, proj_plot, color=proj_color[0])
            if plot_gauss or plot_gauss_x:
                gf = gf_x = GaussFit(x_axis, proj_plot-proj_plot.min(), fit_const=False)
                sp.plot(x_axis*x_factor, gf.reconstruction+proj_plot.min(), color=gauss_color[0], alpha=gauss_alpha)

        if plot_proj or plot_proj_y:
            proj = image.sum(axis=-1)
            proj_plot = (x_axis.min() +(x_axis.max()-x_axis.min()) * proj/proj.max()*0.3)*x_factor
            sp.plot(proj_plot, y_axis*y_factor, color=proj_color[1])
            if plot_gauss or plot_gauss_y:
                gf = gf_y = GaussFit(y_axis, proj_plot-proj_plot.min(), fit_const=False)
                sp.plot(gf.reconstruction+proj_plot.min(), y_axis*y_factor, color=gauss_color[1], alpha=gauss_alpha)

        if hlines is not None:
            for hline in hlines:
                sp.axhline(hline, color=hline_color)
        if vlines is not None:
            for vline in vlines:
                sp.axvline(vline, color=vline_color)

        if revert_x:
            xlim = sp.get_xlim()
            sp.set_xlim(*xlim[::-1])

        outp = {
                'gf_x': gf_x,
                'gf_y': gf_y,
                }
        return outp

def calc_rms(xx, yy):
    mean = np.sum(xx*yy)/np.sum(yy)
    rms = np.sqrt(np.sum((xx-mean)**2*yy)/np.sum(yy))
    return mean, rms

def calc_fwhm(xx, yy):
    halfmax = np.max(yy)/2
    mask = yy >= halfmax
    x1 = xx[mask][0]
    x2 = xx[mask][-1]
    return abs(x2-x1) + abs(xx[1]-xx[0])

@numba.njit
def _x_to_t_inner(image, indices, indices2, x_axis, wake_x, new_arr):
    delta_x = (x_axis[1] - x_axis[0])/2
    new_time_len = new_arr.shape[1]
    for nx, x in zip(indices, x_axis):
        indices = np.interp([x-delta_x, x+delta_x], wake_x, indices2)
        i00, i01 = sorted(indices)
        i0, i1 = int(i00), int(i01)+1
        if i0 == new_time_len-1:
            continue
        if i1 == new_time_len:
            i1 -= 1
        len_new = i1-i0+1
        weights = np.ones(len_new)
        weights[0] = i00-i0
        weights[-1] = i1 - i01
        weights /= np.sum(weights)
        new_arr[:,i0:i1+1] += np.outer(image[:,nx], weights)

