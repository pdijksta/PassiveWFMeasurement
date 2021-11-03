import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .gaussfit import GaussFit
from . import beam_profile
from . import myplotstyle as ms
from .logMsg import LogMsgBase

class Image(LogMsgBase):
    def __init__(self, image, x_axis, y_axis, x_unit='m', y_unit='m', subtract_median=False, x_offset=0, xlabel='x (mm)', ylabel='y (mm)', logger=None):

        self.logger = logger

        if x_axis.size <=1:
            raise ValueError('Size of x_axis is %i' % x_axis.size)

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            image = image[:,::-1]

        if y_axis[1] < y_axis[0]:
            y_axis = y_axis[::-1]
            image = image[::-1,:]

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

    def child(self, new_i, new_x, new_y, x_unit=None, y_unit=None, xlabel=None, ylabel=None):
        x_unit = self.x_unit if x_unit is None else x_unit
        y_unit = self.y_unit if y_unit is None else y_unit
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        return Image(new_i, new_x, new_y, x_unit, y_unit, xlabel=xlabel, ylabel=ylabel)

    def cut(self, x_min, x_max):
        x_axis = self.x_axis
        x_mask = np.logical_and(x_axis >= x_min, x_axis <= x_max)
        new_image = self.image[:,x_mask]
        new_x_axis = x_axis[x_mask]
        return self.child(new_image, new_x_axis, self.y_axis)

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
        #import pdb; pdb.set_trace()
        return output

    def fit_slice(self, smoothen_first=True, smoothen=100e-6, intensity_cutoff=None, charge=1, rms_sigma=5, noise_cut=0.1):
        y_axis = self.y_axis
        n_slices = len(self.x_axis)

        pixelsize = abs(y_axis[1] - y_axis[0])
        smoothen = smoothen/pixelsize

        slice_mean = []
        slice_sigma = []
        slice_gf = []
        slice_rms = []
        slice_mean_rms = []
        slice_cut_rms = []
        slice_cut_mean = []
        for n_slice in range(n_slices):
            intensity = self.image[:,n_slice]
            if smoothen_first:
                yy_conv = gaussian_filter1d(intensity, smoothen)
                gf0 = GaussFit(y_axis, yy_conv, fit_const=True)
                p0 = gf0.popt
            else:
                p0 = None
            try:
                gf = GaussFit(y_axis, intensity, fit_const=True, p0=p0, raise_=True)
            except RuntimeError:
                slice_mean.append(np.nan)
                slice_sigma.append(np.nan)
                slice_gf.append(None)
                slice_mean_rms.append(np.nan)
                slice_rms.append(np.nan)
                slice_cut_rms.append(np.nan)
                slice_cut_mean.append(np.nan)
            else:
                where_max = y_axis[np.argmax(intensity)]
                mask_rms = np.logical_and(
                        y_axis > where_max - abs(gf.sigma)*rms_sigma,
                        y_axis < where_max + abs(gf.sigma)*rms_sigma)
                y_rms = y_axis[mask_rms]
                data_rms = intensity[mask_rms]
                if np.sum(data_rms) == 0:
                    mean_rms, rms = 0, 0
                else:
                    mean_rms = np.sum(y_rms*data_rms)/np.sum(data_rms)
                    rms = np.sum((y_rms-mean_rms)**2*data_rms)/np.sum(data_rms)

                slice_gf.append(gf)
                slice_mean.append(gf.mean)
                slice_sigma.append(gf.sigma**2)
                slice_rms.append(rms)
                slice_mean_rms.append(mean_rms)

                intensity = intensity.copy()
                intensity[np.logical_or(y_axis<mean_rms-1.5*rms, y_axis>mean_rms+1.5*rms)]=0
                prof_y = intensity-intensity.min()
                if np.all(prof_y == 0):
                    slice_cut_rms.append(0)
                    slice_cut_mean.append(0)
                else:
                    profile = beam_profile.AnyProfile(y_axis, prof_y)
                    profile.cutoff2(noise_cut)
                    profile.crop()

                    slice_cut_rms.append(profile.rms()**2)
                    slice_cut_mean.append(profile.mean())

            # Debug bad gaussfits
            #if 101e-15 < self.x_axis[n_slice] < 104e-15:
            #if abs(gf.sigma) < 1e5:
            #    import matplotlib.pyplot as plt
            #    num = plt.gcf().number
            #    plt.figure()
            #    plt.suptitle('Debug 38 fs')
            #    sp = plt.subplot(1,1,1)
            #    gf.plot_data_and_fit(sp)
            #    sp.legend()
            #    plt.figure(num)
            #    plt.show()
            #    import pdb; pdb.set_trace()

        proj = np.sum(self.image, axis=-2)
        proj = proj / np.sum(proj) * charge
        current = proj / (self.x_axis[1] - self.x_axis[0])


        slice_dict = {
                'slice_x': self.x_axis,
                'slice_mean': np.array(slice_mean),
                'slice_sigma_sq': np.array(slice_sigma),
                'slice_rms_sq': np.array(slice_rms),
                'slice_mean_rms': np.array(slice_mean_rms),
                'slice_gf': slice_gf,
                'slice_intensity': proj,
                'slice_current': current,
                'slice_cut_rms_sq': np.array(slice_cut_rms),
                'slice_cut_mean': np.array(slice_cut_mean),
                }

        if intensity_cutoff:
            mask = proj > proj.max()*intensity_cutoff
            for key, value in slice_dict.items():
                if hasattr(value, 'shape') and value.shape == mask.shape:
                    slice_dict[key] = value[mask]
        return slice_dict

    def y_to_eV(self, dispersion, energy_eV, ref_y=None):
        if ref_y is None:
            ref_y = GaussFit(self.y_axis, np.sum(self.image, axis=-1)).mean
            #print('y_to_eV', ref_y*1e6, ' [um]')
        E_axis = (self.y_axis-ref_y) * dispersion * energy_eV
        return self.child(self.image, self.x_axis, E_axis, y_unit='eV', ylabel='$\Delta$ E (MeV)'), ref_y

    def x_to_t_linear(self, factor):
        return self.child(self.image, self.x_axis*factor, self.y_axis, x_unit='fs', xlabel='t (fs)')

    def x_to_t(self, wake_x, wake_time, debug=False, print_=False):
        if wake_time[1] < wake_time[0]:
            wake_x = wake_x[::-1]
            wake_time = wake_time[::-1]

        new_img0 = np.zeros_like(self.image)
        new_t_axis = np.linspace(wake_time.min(), wake_time.max(), self.image.shape[1])
        x_interp = np.interp(new_t_axis, wake_time, wake_x)

        to_print = []
        for t_index, (t, x) in enumerate(zip(new_t_axis, x_interp)):
            x_index = np.argmin((self.x_axis - x)**2)
            new_img0[:,t_index] = self.image[:,x_index]

            if print_:
                to_print.append('%i %i %.1f %.1f' % (t_index, x_index, t*1e15, x*1e6))
        if print_:
            print('\n'.join(to_print))

        diff_x = np.concatenate([np.diff(x_interp), [0]])

        new_img = new_img0 * np.abs(diff_x)
        new_img = new_img / new_img.sum() * self.image.sum()


        output = self.child(new_img, new_t_axis, self.y_axis, x_unit='s', xlabel='t (fs)')

        if debug:
            ms.figure('Debug x_to_t')
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
            output.plot_img_and_proj(sp)

            sp = subplot(sp_ctr, title='Image new 0', xlabel='t [fs]', ylabel=' y [mm]', grid=False)
            sp_ctr += 1
            new_obj0 = self.child(new_img0, new_t_axis, self.y_axis, x_unit='s', xlabel='t (fs)')
            new_obj0.plot_img_and_proj(sp)

        #ms.plt.show()
        #import pdb; pdb.set_trace()
        return output

    def force_projection(self, proj_x, proj):
        real_proj = np.interp(self.x_axis, proj_x, proj)
        sum_ = self.image.sum(axis=-2)
        sum_[sum_ == 0] = np.inf
        image2 = self.image / sum_ / real_proj.sum() * real_proj
        image2 = image2 / image2.sum() * self.image.sum()

        return self.child(image2, self.x_axis, self.y_axis)

    def plot_img_and_proj(self, sp, x_factor=None, y_factor=None, plot_proj=True, log=False, revert_x=False, plot_gauss=True, slice_dict=None, xlim=None, ylim=None, cmapname='hot', slice_cutoff=0, gauss_color='orange', proj_color='green', slice_color='deepskyblue', key_sigma='slice_cut_rms_sq', key_mean='slice_cut_mean'):

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
            yy = slice_dict[key_mean][mask]*y_factor
            yy_err = np.sqrt(slice_dict[key_sigma][mask])*y_factor
            sp.errorbar(xx, yy, yerr=yy_err, color=slice_color, ls='None', marker='None', lw=1)
            sp.set_xlim(*old_lim[0])
            sp.set_ylim(*old_lim[1])

        if plot_proj:
            proj = image.sum(axis=-2)
            proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*y_factor
            sp.plot(x_axis*x_factor, proj_plot, color=proj_color)
            if plot_gauss:
                gf = GaussFit(x_axis, proj_plot-proj_plot.min(), fit_const=False)
                sp.plot(x_axis*x_factor, gf.reconstruction+proj_plot.min(), color=gauss_color)

            proj = image.sum(axis=-1)
            proj_plot = (x_axis.min() +(x_axis.max()-x_axis.min()) * proj/proj.max()*0.3)*x_factor
            sp.plot(proj_plot, y_axis*y_factor, color=proj_color)
            if plot_gauss:
                gf = GaussFit(y_axis, proj_plot-proj_plot.min(), fit_const=False)
                sp.plot(gf.reconstruction+proj_plot.min(), y_axis*y_factor, color=gauss_color)

        if revert_x:
            xlim = sp.get_xlim()
            sp.set_xlim(*xlim[::-1])

def plot_slice_dict(slice_dict):
    subplot = ms.subplot_factory(3, 3)
    sp_ctr = np.inf
    for n_slice, slice_gf in enumerate(slice_dict['slice_gf']):
        slice_sigma = slice_dict['slice_sigma_sq'][n_slice]
        slice_rms = slice_dict['slice_rms_sq'][n_slice]
        slice_cut = slice_dict['slice_cut_rms_sq'][n_slice]
        if sp_ctr > 9:
            ms.figure('Investigate slice')
            sp_ctr = 1
        sp = subplot(sp_ctr, title='Slice %i, $\sigma$=%.1e, rms=%.1e, cut=%.1e' % (n_slice, slice_sigma, slice_rms, slice_cut))
        sp_ctr += 1
        slice_gf.plot_data_and_fit(sp)
        sp.legend()

