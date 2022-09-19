import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

try:
    import beam_profile
except ImportError:
    #from . import gaussfit
    from . import beam_profile

class DoublehornFit:
    def __init__(self, xx, yy, raise_=True):

        self.xx = xx
        self.yy = yy

        #self.gf = gf = gaussfit.GaussFit(xx, yy, fit_const=False)
        profile = beam_profile.AnyProfile(xx, yy)
        mean, rms = profile.mean(), profile.rms()

        mask_left = xx < mean
        if np.sum(mask_left) == 0:
            arg_left = 0
        else:
            arg_left = np.argmax(np.abs(yy[mask_left]))
        self.arg_left = arg_left
        pos_left = xx[arg_left]
        max_left = yy[arg_left]


        mask_right = xx > mean
        if np.sum(mask_right) == 0:
            arg_right = len(yy)-1
        else:
            arg_right = np.argmax(np.abs(yy[mask_right])) + np.sum(mask_right == 0)
        self.arg_right = arg_right
        pos_right = xx[arg_right]
        max_right = yy[arg_right]
        s1a = s1b = s2a = s2b = rms / 10

        mask_middle = np.logical_and(xx > pos_left, xx < pos_right)
        if np.any(mask_middle):
            const_middle = np.min(yy[mask_middle])
        else:
            const_middle = np.mean(yy)

        self.pos_right = pos_right
        self.pos_left = pos_left
        p0 = self.p0 = [s1a, s1b, s2a, s2b, const_middle, max_left, max_right]


        try:
            self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, p0=p0)
        except RuntimeError as e:
            if raise_:
                plt.figure()
                plt.plot(xx, yy, label='Input')
                plt.plot(xx, self.fit_func(xx, *p0), label='Guess')
                plt.legend()
                plt.show()
                raise
            self.popt, self.pcov = p0, np.ones([len(p0), len(p0)], float)
            print(e)
            print('Fit did not converge. Using p0 instead!')

        self.reconstruction = self.fit_func(xx, *self.popt)
        #import pdb; pdb.set_trace()
        self.s1a, self.s1b, self.s2a, self.s2b, self.const_middle, self.max_left, self.max_right = self.popt
        self.s1a = abs(self.s1a)
        self.s1b = abs(self.s1b)
        self.s2a = abs(self.s2a)
        self.s2b = abs(self.s2b)

    def fit_func(self, xx, s1a, s1b, s2a, s2b, const_middle, max_left, max_right):
        pos_left = self.pos_left
        pos_right = self.pos_right
        outp = np.zeros_like(xx)
        pos_middle = (pos_right + pos_left)/2.

        mask1a = xx <= pos_left
        outp[mask1a] = gauss(xx[mask1a], max_left, pos_left, s1a, 0)

        mask1b = np.logical_and(xx > pos_left, xx <= pos_middle)
        outp[mask1b] = gauss(xx[mask1b], max_left-const_middle, pos_left, s1b, const_middle)

        mask2a = np.logical_and(xx > pos_middle, xx <= pos_right)
        outp[mask2a] = gauss(xx[mask2a], max_right-const_middle, pos_right, s2a, const_middle)

        mask2b = xx > pos_right
        outp[mask2b] = gauss(xx[mask2b], max_right, pos_right, s2b, 0)

        return outp

    def plot_data_and_fit(self, sp):
        sp.plot(self.xx, self.yy, label='Data', marker='.')
        sp.plot(self.xx, self.reconstruction, label='Reconstruction', marker='.', ls='--')
        sp.legend()

def gauss(xx, scale, mean, sig, const=0):
    #return scale*stats.norm.pdf(xx, mean, sig)
    if sig != 0:
        return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const
    else:
        return 0

