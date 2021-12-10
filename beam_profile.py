import numpy as np
from scipy.ndimage import gaussian_filter1d

from .gaussfit import GaussFit
from .logMsg import LogMsgBase
from . import blmeas
from . import config

def find_rising_flank(arr, method='Size'):
    """
    Method can be 'Length' or 'Size'
    """
    arr = arr.copy()
    #arr[arr<arr.max()*0.01] = 0
    prev_val = -np.inf
    start_index = None
    len_ctr = 0
    pairs = []
    for index, val in enumerate(arr):
        if val > prev_val:
            if start_index is None:
                start_index = index - 1
                start_val = val
            len_ctr += 1
        else:
            if start_index is not None:
                if method == 'Length':
                    pairs.append((len_ctr, start_index, index))
                elif method == 'Size':
                    pairs.append((prev_val-start_val, start_index, index))
                start_index = None
                start_val = None
            len_ctr = 0
        prev_val = val
    end_longest_streak = sorted(pairs)[-1][-1]
    return end_longest_streak

class Profile(LogMsgBase):

    def __init__(self):
        self._gf = None
        self._gf_xx = None
        self._gf_yy = None

    def expand(self, ratio):
        len_x = self._xx[-1] - self._xx[0]
        new_xx = np.linspace(self._xx[0], self._xx[-1] + ratio*len_x, int(len(self)*(1+ratio)))
        new_yy = np.interp(new_xx, self._xx, self._yy, left=0, right=0)
        self._xx = new_xx
        self._yy = new_yy

    def reshape(self, new_shape):
        _xx = np.linspace(self._xx.min(), self._xx.max(), int(new_shape))
        old_sum = np.sum(self._yy)

        _yy = np.interp(_xx, self._xx, self._yy)
        #if new_shape >= len(self._xx):
        #    _yy = np.interp(_xx, self._xx, self._yy)
        #else:
        #    _yy = np.histogram(self._xx, bins=int(new_shape), weights=self._yy)[0]

        _yy *= old_sum / _yy.sum()
        self._xx, self._yy = _xx, _yy

    def mean(self):
        return np.sum(self._xx*self._yy) / np.sum(self._yy)

    def rms(self):
        mean = self.mean()
        square = (self._xx - mean)**2
        rms = np.sqrt(np.sum(square * self._yy) / np.sum(self._yy))
        return rms

    def fwhm(self):
        def get_lim(indices):
            xx = abs_yy[indices[0]:indices[1]+1]
            yy = self._xx[indices[0]:indices[1]+1]
            sort = np.argsort(xx)
            x = np.interp(half, xx[sort], yy[sort])
            return x

        abs_yy = np.abs(self._yy)
        half = abs_yy.max()/2.
        mask_fwhm = abs_yy > half
        indices_fwhm = np.argwhere(mask_fwhm)
        index_min, index_max = indices_fwhm.min(), indices_fwhm.max()
        lims = []
        if index_min == 0:
            lims.append(self._xx[0])
        else:
            lims.append(get_lim([index_min-1, index_min]))
        if index_max == len(self._xx)-1:
            lims.append(self._xx[-1])
        else:
            lims.append(get_lim([index_max, index_max+1]))
        fwhm = abs(lims[1]-lims[0])
        return fwhm


    def cutoff(self, cutoff_factor):
        """
        Cutoff based on max value of the y array.
        """
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        abs_yy = np.abs(yy)
        yy[abs_yy<abs_yy.max()*cutoff_factor] = 0
        self._yy = yy / np.sum(yy) * old_sum

    def aggressive_cutoff(self, cutoff_factor):
        """
        Cutoff based on max value of the y array.
        Also sets to 0 all values before and after the first 0 value (from the perspective of the maximum).
        """
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        abs_yy = np.abs(yy)
        yy[abs_yy<abs_yy.max()*cutoff_factor] = 0

        index_max = np.argmax(abs_yy)
        index_arr = np.arange(len(yy))
        is0 = (yy == 0)
        zero_pos = np.logical_and(index_arr > index_max, is0)
        zero_neg = np.logical_and(index_arr < index_max, is0)
        if np.any(zero_pos):
            nearest_zero_pos = index_arr[zero_pos][0]
            yy[nearest_zero_pos:] = 0
        if np.any(zero_neg):
            nearest_zero_neg = index_arr[zero_neg][-1]
            yy[:nearest_zero_neg] = 0

        if np.sum(yy) == 0:
            self._yy = np.zeros_like(yy)
        else:
            self._yy = yy / np.sum(yy) * old_sum


    def crop(self):
        old_sum = np.sum(self._yy)
        mask = self._yy != 0
        xx_nonzero = self._xx[mask]
        new_x = np.linspace(xx_nonzero.min(), xx_nonzero.max(), len(self._xx))
        new_y = np.interp(new_x, self._xx, self._yy)
        self._xx = new_x
        self._yy = new_y / new_y.sum() * old_sum

    def __len__(self):
        return len(self._xx)

    @property
    def gaussfit(self):
        if self._gf is not None and (self._xx.min(), self._xx.max(), self._xx.sum()) == self._gf_xx and (self._yy.min(), self._yy.max(), self._yy.sum()) == self._gf_yy:
            return self._gf

        self._gf = GaussFit(self._xx, self._yy, fit_const=False)
        self._gf_xx = (self._xx.min(), self._xx.max(), self._xx.sum())
        self._gf_yy = (self._yy.min(), self._yy.max(), self._yy.sum())
        return self._gf

    @property
    def integral(self):
        return np.trapz(self._yy, self._xx)

    def smoothen(self, gauss_sigma, extend=True):
        if gauss_sigma is None or gauss_sigma == 0:
            return

        diff = self._xx[1] - self._xx[0]
        if extend:
            n_extend = int(gauss_sigma // diff * 2)
            extend0 = np.arange(self._xx[0]-(n_extend+1)*diff, self._xx[0]-0.5*diff, diff)
            extend1 = np.arange(self._xx[-1]+diff, self._xx[-1]+diff*(n_extend+0.5), diff)
            zeros0 = np.zeros_like(extend0)
            zeros1 = np.zeros_like(extend1)

            self._xx = np.concatenate([extend0, self._xx, extend1])
            self._yy = np.concatenate([zeros0, self._yy, zeros1])

        real_sigma = gauss_sigma/diff
        new_yy = gaussian_filter1d(self._yy, real_sigma)
        self._yy = new_yy

    def center(self, type_='Gauss'):
        if type_ == 'Gauss':
            self._xx = self._xx - self.gaussfit.mean
        elif type_ == 'Mean':
            self._xx = self._xx - self.mean()
        else:
            raise ValueError(type_)

    def scale_xx(self, scale_factor, keep_range=False):
        new_xx = self._xx * scale_factor
        if keep_range:
            old_sum = self._yy.sum()
            new_yy = np.interp(self._xx, new_xx, self._yy, left=0, right=0)
            self._yy = new_yy / new_yy.sum() * old_sum
        else:
            self._xx = new_xx

    def scale_yy(self, scale_factor):
        self._yy = self._yy * scale_factor

    def remove0(self):
        mask = self._yy != 0
        self._xx = self._xx[mask]
        self._yy = self._yy[mask]

    def flipx(self):
        self._xx = -self._xx[::-1]
        self._yy = self._yy[::-1]

class ScreenDistribution(Profile):
    def __init__(self, x, intensity, real_x=None, subtract_min=True, total_charge=1, meta_data=None, logger=None):
        super().__init__()
        self.logger = logger
        self._xx = x
        assert np.all(np.diff(self._xx)>=0)
        self._yy = intensity
        if subtract_min:
            self._yy = self._yy - np.min(self._yy)
        self.real_x = real_x
        self.total_charge = total_charge
        self.meta_data = meta_data
        self.normalize()

    @property
    def x(self):
        return self._xx

    @property
    def intensity(self):
        return self._yy

    def normalize(self, norm=None):
        if norm is None:
            norm = abs(self.total_charge)
        self._yy = self._yy / self.integral * norm

    def plot_standard(self, sp, show_mean=False, **kwargs):
        if self._yy[0] != 0:
            diff = self._xx[1] - self._xx[0]
            x = np.concatenate([[self._xx[0] - diff], self._xx])
            y = np.concatenate([[0.], self._yy])
        else:
            x, y = self.x, self.intensity

        if y[-1] != 0:
            diff = self.x[1] - self.x[0]
            x = np.concatenate([x, [x[-1] + diff]])
            y = np.concatenate([y, [0.]])

        factor = np.abs(self.total_charge /np.trapz(y, x))
        #try:
        #    integral = np.trapz(y*factor, x)
        #    print('total_charge', self.total_charge, 'factor', factor, 'integral', integral, 'label', kwargs['label'])
        #except:
        #    pass
        outp = sp.plot(x*1e3, y*1e9*factor, **kwargs)
        if show_mean:
            color = outp[0].get_color()
            sp.axvline(self.mean()*1e3, ls='--', color=color)
        return outp

    def to_dict_custom(self):
        return {'x': self.x,
                'intensity': self.intensity,
                'real_x': self.real_x,
                'total_charge': self.total_charge,
                'meta_data': self.meta_data
                }

    @staticmethod
    def from_dict(dict_):
        return ScreenDistribution(dict_['x'], dict_['intensity'], dict_['real_x'], total_charge=dict_['total_charge'])

class AnyProfile(Profile):
    def __init__(self, xx, yy, logger=None):
        super().__init__()
        self.logger = logger
        self._xx = xx
        self._yy = yy

    @property
    def xx(self):
        return self._xx

    @property
    def yy(self):
        return self._yy

def getScreenDistributionFromPoints(x_points, screen_bins, smoothen=0, total_charge=1):
    """
    Smoothening by applying changes to coordinate.
    Does not actually smoothen the output, just broadens it.
    """
    if smoothen:
        rand = np.random.randn(len(x_points))
        rand[rand>3] = 3
        rand[rand<-3] = -3
        x_points2 = x_points + rand*smoothen
    else:
        x_points2 = x_points
    screen_hist, bin_edges0 = np.histogram(x_points2, bins=screen_bins, density=True)
    screen_xx = (bin_edges0[1:] + bin_edges0[:-1])/2

    return ScreenDistribution(screen_xx, screen_hist, real_x=x_points, total_charge=total_charge)

class BeamProfile(Profile):
    def __init__(self, time, charge_dist, energy_eV, total_charge, logger=None):
        super().__init__()
        self.logger = logger

        if np.any(np.isnan(time)):
            raise ValueError('nans in time')
        if np.any(np.isnan(charge_dist)):
            raise ValueError('nans in charge_dist')

        self.ignore_range = None
        self._xx = time
        assert np.all(np.diff(self._xx)>=0)
        self._yy = charge_dist / charge_dist.sum() * total_charge
        self.energy_eV = energy_eV
        self.total_charge = total_charge
        self.wake_dict = {}

    def calc_wake(self, structure, gap, beam_position, wake_type):
        #print('calc_wake gap beam_offset %.2e %.2e' % (gap, beam_offset))
        if abs(beam_position) > gap/2.:
            raise ValueError('Beam offset is too large! Gap: %.2e Offset: %.2e' % (gap, beam_position))
        dict_key = gap, beam_position, structure, wake_type
        if dict_key in self.wake_dict:
            return self.wake_dict[dict_key]
        wf_dict = structure.convolve(self, gap/2., beam_position, wake_type)
        self.wake_dict[dict_key] = wf_dict
        #self.logMsg('wake_dict calculated for gap %.2f mm and beam position %.2f mm' % (gap*1e3, beam_position*1e3))
        if np.any(np.isnan(wf_dict['wake_potential'])):
            raise ValueError
        return wf_dict

    def to_dict_custom(self):
        return {'time': self.time,
                'charge_dist': self.charge_dist,
                'energy_eV': self.energy_eV,
                'total_charge': self.total_charge,
                }

    @staticmethod
    def from_dict(dict_):
        return BeamProfile(dict_['time'], dict_['charge_dist'], dict_['energy_eV'], dict_['total_charge'])

    @property
    def time(self):
        return self._xx

    @property
    def charge_dist(self):
        return self._yy

    def scale_yy(self, scale_factor):
        self.total_charge *= scale_factor
        super().scale_yy(scale_factor)

    def shift(self, center):
        if center == 'Max':
            center_index = np.argmax(np.abs(self.charge_dist))
        elif center == 'Left':
            center_index = find_rising_flank(np.abs(self.charge_dist))
        elif center == 'Right':
            center_index = len(self.charge_dist) - find_rising_flank(np.abs(self.charge_dist[::-1]))
        else:
            raise ValueError(center)

        self._xx = self._xx - self._xx[center_index]

    def plot_standard(self, sp, norm=True, center=None, center_max=False, center_float=None, **kwargs):
        """
        center can be one of 'Max', 'Left', 'Right', 'Left_fit', 'Right_fit', 'Gauss'
        """

        # Backward compatibility
        if center_max:
            center='Max'

        factor = np.sign(self.total_charge)
        if norm:
            factor *= self.total_charge/self.integral

        center_index = None
        if center is None:
            pass
        elif center == 'Max':
            center_index = np.argmax(np.abs(self.charge_dist))
        elif center == 'Left':
            center_index = find_rising_flank(self.charge_dist)
        elif center == 'Right':
            center_index = len(self.charge_dist) - find_rising_flank(self.charge_dist[::-1])
        elif center == 'Gauss':
            center_index = np.argmin((self._xx - self.gaussfit.mean)**2)
        elif center == 'Mean':
            mean = np.sum(self._xx*self._yy) / np.sum(self._yy)
            center_index = np.argmin((self._xx - mean)**2)
        else:
            raise ValueError

        if center_float is not None:
            mean = np.sum(self._xx*self._yy) / np.sum(self._yy)
            xx = (self.time - (mean - center_float))*1e15
        elif center_index is None:
            xx = self.time*1e15
        else:
            xx = (self.time - self.time[center_index])*1e15

        if self._yy[0] != 0:
            diff = xx[1] - xx[0]
            x = np.concatenate([[xx[0] - diff], xx])
            y = np.concatenate([[0.], self._yy])
        else:
            x, y = xx, self._yy

        if y[-1] != 0:
            diff = xx[1] - xx[0]
            x = np.concatenate([x, [x[-1] + diff]])
            y = np.concatenate([y, [0.]])

        return sp.plot(x, y*factor/1e3, **kwargs)


def profile_from_blmeas(file_or_dict, time_range, total_charge, energy_eV, cutoff, subtract_min=True, zero_crossing=1, len_profile=config.get_default_backward_options()['len_profile']):
    blmeas_dict = blmeas.load_avg_blmeas(file_or_dict)[zero_crossing]

    tt = blmeas_dict['time']
    if subtract_min:
        curr = blmeas_dict['current_reduced']
    else:
        curr = blmeas_dict['current']

    if tt[1] < tt[0]:
        tt = tt[::-1]
        curr = curr[::-1]
    bp0 = BeamProfile(tt, curr, energy_eV, total_charge)
    bp0.center()
    tt_new = np.linspace(-time_range/2., time_range/2., len_profile)
    curr_new = np.interp(tt_new, bp0.time, bp0.charge_dist, left=0., right=0.)
    bp = BeamProfile(tt_new, curr_new, energy_eV, total_charge)
    bp.aggressive_cutoff(cutoff)
    return bp

def get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, energy_eV, cutoff=1e-3):
    """
    cutoff can be None
    """
    time_arr = np.linspace(-tt_range/2, tt_range/2, int(tt_points))
    current_gauss = np.exp(-(time_arr-np.mean(time_arr))**2/(2*sig_t**2))

    if cutoff is not None:
        abs_c = np.abs(current_gauss)
        current_gauss[abs_c<cutoff*abs_c.max()] = 0

    return BeamProfile(time_arr, current_gauss, energy_eV, total_charge)

def get_average_profile(p_list):
    len_profile = max(len(p) for p in p_list)

    xx_list = [p._xx - p.gaussfit.mean for p in p_list]
    yy_list = [p._yy for p in p_list]

    min_profile = min(x.min() for x in xx_list)
    max_profile = max(x.max() for x in xx_list)

    xx_interp = np.linspace(min_profile, max_profile, len_profile)
    yy_interp_arr = np.zeros([len(p_list), len_profile])

    for n, (xx, yy) in enumerate(zip(xx_list, yy_list)):
        yy_interp_arr[n] = np.interp(xx_interp, xx, yy)

    yy_mean = np.mean(yy_interp_arr, axis=0)
    return xx_interp, yy_mean

