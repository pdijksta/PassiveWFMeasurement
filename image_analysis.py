import numpy as np
import matplotlib.pyplot as plt

from .gaussfit import GaussFit
from . import beam_profile
from . import myplotstyle as ms
from .logMsg import LogMsgBase

class Image(LogMsgBase):
    def __init__(self, image, x_axis, y_axis, charge=1, x_unit='m', y_unit='m', subtract_median=False, x_offset=0, xlabel='x (mm)', ylabel='y (mm)', logger=None):
        self.logger = logger
        if x_axis.size <=1:
            raise ValueError('Size of x_axis is %i' % x_axis.size)

        if image.shape[0] != y_axis.size or image.shape[1] != x_axis.size:
            raise ValueError('Wrong shapes!', image.shape, x_axis.shape, y_axis.shape)

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
        self.x_axis = x_axis - x_offset
        self.y_axis = y_axis
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.charge = charge

    def to_dict_custom(self):
        outp = {
                'image': self.image,
                'x_axis': self.x_axis,
                'y_axis': self.y_axis,
                'x_unit': self.x_unit,
                'y_unit': self.y_unit,
                'xlabel': self.xlabel,
                'ylabel': self.ylabel,
                }
        return outp

    def child(self, new_i, new_x, new_y, x_unit=None, y_unit=None, xlabel=None, ylabel=None):
        x_unit = self.x_unit if x_unit is None else x_unit
        y_unit = self.y_unit if y_unit is None else y_unit
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        return Image(new_i, new_x, new_y, self.charge, x_unit, y_unit, xlabel=xlabel, ylabel=ylabel)

    def transpose(self):
        return self.child(self.image.T.copy(), self.y_axis.copy(), self.x_axis.copy(), self.y_unit, self.x_unit, self.ylabel, self.xlabel)

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

    def fit_slice(self, rms_sigma=5, debug=False, current_cutoff=None, E_lims=None):
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

        for n_slice in range(n_slices):

            if current_cutoff is not None and current[n_slice] < current_cutoff:
                addzero()
                continue
            intensity = self.image[mask_Elim,n_slice]
            intensity = intensity - intensity.min()

            try:
                gf = GaussFit(y_axis, intensity, fit_const=True, raise_=True)
            except RuntimeError:
                addzero()
                continue

            mean_full, rms_full = calc_rms(y_axis, intensity)
            slice_full_mean.append(mean_full)
            slice_full_rms.append(rms_full**2)

            where_max = y_axis[np.argmax(gf.reconstruction)]
            mask_rms = np.logical_and(
                    y_axis > where_max - abs(gf.sigma)*rms_sigma,
                    y_axis < where_max + abs(gf.sigma)*rms_sigma)
            y_rms = y_axis[mask_rms]
            data_rms = intensity[mask_rms]
            if np.sum(data_rms) == 0:
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
            prof_y[np.logical_or(y_axis<lim1, y_axis>lim2)] = 0
            if np.all(prof_y == 0):
                slice_cut_rms.append(0)
                slice_cut_mean.append(0)
            else:
                cut_mean, cut_rms = calc_rms(y_axis, prof_y)
                slice_cut_rms.append(cut_rms**2)
                slice_cut_mean.append(cut_mean)

            # Debug bad gaussfits
            if debug:
                if rms == 0 or slice_cut_rms[-1] == 0:
                    import matplotlib.pyplot as plt
                    num = plt.gcf().number
                    plt.figure()
                    plt.suptitle('Debug')
                    sp = plt.subplot(1,1,1)
                    gf.plot_data_and_fit(sp)
                    sp.legend()
                    plt.figure(num)
                    plt.show()
                    import pdb; pdb.set_trace()

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
        return slice_dict

    def y_to_eV(self, dispersion, energy_eV, ref_y=None):
        if ref_y is None:
            ref_y = GaussFit(self.y_axis, np.sum(self.image, axis=-1)).mean
        if dispersion == 0:
            raise ValueError
        E_axis = (self.y_axis-ref_y) / dispersion * energy_eV
        return self.child(self.image, self.x_axis, E_axis, y_unit='eV', ylabel='$\Delta$ E (MeV)'), ref_y

    def x_to_t_linear(self, factor, mean_to_zero=True, current_cutoff=None):
        proj = self.image.sum(axis=0)
        if mean_to_zero:
            profile = beam_profile.BeamProfile(self.x_axis*factor, proj, 1, self.charge)
            if current_cutoff:
                profile._yy[np.abs(profile.get_current()) < current_cutoff] = 0
            refx = profile.mean()
        else:
            refx = 0
        return self.child(self.image, self.x_axis*factor-refx, self.y_axis, x_unit='s', xlabel='t (fs)')

    def x_to_t(self, wake_x, wake_time, debug=False, print_=False):
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
            if delta_cumsum == 0 and full_intensity == 0:
                new_img1[:,t_index] = 0
            else:
                new_img1[:,t_index] = self.image[:,x_index] / np.sum(self.image[:,x_index]) * delta_cumsum / full_intensity
            if np.any(np.isnan(new_img1[:,t_index])):
                import pdb; pdb.set_trace()
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

            sp = subplot(sp_ctr, title='Image projection X', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(self.x_axis*1e3, self.image.sum(axis=-2))

            sp = subplot(sp_ctr, title='Image projection T', xlabel='t [fs]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(output.x_axis*1e15, output.image.sum(axis=-2))

            sp = subplot(sp_ctr, title='Image old', xlabel='x [mm]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            self.plot_img_and_proj(sp)

            sp = subplot(sp_ctr, title='Image new', xlabel='t [fs]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            try:
                output.plot_img_and_proj(sp)
            except:
                print('Cannot plot output')

            sp = subplot(sp_ctr, title='Image new 1', xlabel='t [fs]', ylabel=' y [mm]', grid=False)
            sp_ctr += 1
            new_obj1 = self.child(new_img1, new_t_axis, self.y_axis, x_unit='s', xlabel='t (fs)')
            try:
                new_obj1.plot_img_and_proj(sp)
            except:
                print('Cannot plot new_obj1')
            #fig.savefig('/tmp/debug_fig.pdf')

        return output

    def plot_img_and_proj(self, sp, x_factor=None, y_factor=None, plot_proj=True, log=False, revert_x=False, plot_gauss=True, slice_dict=None, xlim=None, ylim=None, cmapname='hot', slice_cutoff=0, gauss_color=('orange', 'orange'), proj_color=('green', 'green'), slice_color='deepskyblue', slice_method='cut', plot_gauss_x=False, plot_gauss_y=False, plot_proj_x=False, plot_proj_y=False, gauss_alpha=None, cut_intensity_quantile=None, hlines=None, hline_color='deepskyblue', vlines=None, vline_color='deepskyblue'):

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
        else:
            log = image

        sp.imshow(log, aspect='auto', extent=extent, origin='lower', cmap=plt.get_cmap(cmapname))

        if slice_dict is not None:
            old_lim = sp.get_xlim(), sp.get_ylim()
            mask = slice_dict['slice_current'] > slice_cutoff
            xx = slice_dict['slice_x'][mask]*x_factor
            yy = slice_dict[slice_method]['mean'][mask]*y_factor
            yy_err = np.sqrt(slice_dict[slice_method]['sigma_sq'][mask])*y_factor
            sp.errorbar(xx, yy, yerr=yy_err, color=slice_color, ls='None', marker='None', lw=1)
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

