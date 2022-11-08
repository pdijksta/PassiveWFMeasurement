import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

factor_fwhm = 2*np.sqrt(2*np.log(2))

class GaussFit:
    def __init__(self, xx, yy, print_=False, raise_=False, fit_const=True, p0=None):
        self.xx = xx
        self.yy = yy
        assert len(xx.shape) == 1
        assert len(yy.shape) == 1

        if fit_const:
            self.jacobi_arr = np.ones([len(xx), 4])
        else:
            self.jacobi_arr = np.ones([len(xx), 3])

        if p0 is None:
            if fit_const:
                const_0 = self.const_0 = min(yy[0], yy[-1])
            else:
                const_0 = self.const_0 = 0.

            if abs(np.max(yy-const_0)) > abs(np.min(yy-const_0)):
                scale_0 = self.scale_0 = np.max(yy)-const_0
                mean_0 = self.mean_0 = np.squeeze(xx[np.argmax(yy)])
            else:
                scale_0 = self.scale_0 = np.min(yy)-const_0
                mean_0 = self.mean_0 = np.squeeze(xx[np.argmin(yy)])

            mask_above_half = yy-const_0 > scale_0/2

            if np.sum(mask_above_half) > 1:
                sigma_0 = abs(xx[mask_above_half][-1] - xx[mask_above_half][0])/factor_fwhm
            else:
                sigma_0 = 1

            self.sigma_0 = sigma_0

            if fit_const:
                p0 = self.p0 = [scale_0, mean_0, sigma_0, const_0]
            else:
                p0 = self.p0 = [scale_0, mean_0, sigma_0]
        else:
            self.p0 = p0
            if fit_const:
                self.scale_0, self.mean_0, self.sigma_0, self.const_0 = p0
            else:
                self.scale_0, self.mean_0, self.sigma_0 = p0
                self.const_0 = 0

        with warnings.catch_warnings():
            warnings.simplefilter('error', category=OptimizeWarning)
            try:
                self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, p0=p0, jac=self.jacobi, maxfev=100)
            except (RuntimeError, OptimizeWarning):
                try:
                    p0[2] *= 5
                    self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, p0=p0, jac=self.jacobi, maxfev=100)
                except (RuntimeError, OptimizeWarning) as e:
                    if raise_:
                        raise
                    self.popt, self.pcov = p0, np.ones([len(p0), len(p0)], float)
                    if print_:
                        print(e)
                        print('Fit did not converge. Using p0 instead!')

        if fit_const:
            self.scale, self.mean, self.sigma, self.const = self.popt
        else:
            self.scale, self.mean, self.sigma = self.popt
            self.const = 0
        self.sigma = abs(self.sigma)
        self.reconstruction = self.fit_func(xx, *self.popt)
        self.fit_integral = self.scale*self.sigma*np.sqrt(2*np.pi)

        if print_:
            print(p0, '\t\t', self.popt)

    @staticmethod
    def fit_func(xx, scale, mean, sig, const=0):
        #return scale*stats.norm.pdf(xx, mean, sig)
        if sig != 0:
            return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const
        else:
            return 0

    def jacobi(self, xx, scale, mean, sig, const=0):
        g_minus_const = self.fit_func(xx, scale, mean, sig, 0)
        if scale == 0:
            self.jacobi_arr[:,0] = 0
        else:
            self.jacobi_arr[:,0] = g_minus_const/scale
        if sig != 0:
            self.jacobi_arr[:,1] = g_minus_const * (xx-mean)/sig**2
            self.jacobi_arr[:,2] = g_minus_const * (xx-mean)**2/sig**3
        else:
            self.jacobi_arr[:,1] = self.jacobi_arr[:,2] = np.inf

        return self.jacobi_arr

    def plot_data_and_fit(self, sp):
        sp.plot(self.xx, self.yy, label='Data', marker='.')
        xx2 = np.linspace(self.xx[0], self.xx[-1], 100)
        reconstruction2 = self.fit_func(xx2, *self.popt)
        sp.plot(xx2, reconstruction2, label='Reconstruction', marker='.', ls='--')
        sp.axhline(self.scale_0+self.const_0, label='scale0+const0', color='black')
        sp.axhline(self.scale+self.const, label='scale+const', color='black', ls='--')

        sp.axhline(self.const_0, label='const0', color='black')
        sp.axhline(self.const, label='const', color='black', ls='--')

        sp.axvline(self.mean_0, label='mean0', color='red')
        sp.axvline(self.mean, label='mean', color='red', ls='--')

        sp.axvline(self.mean_0-self.sigma_0, label='sigma0', color='green')
        sp.axvline(self.mean_0+self.sigma_0, color='green')
        sp.axvline(self.mean-self.sigma, label='sigma', color='green', ls='--')
        sp.axvline(self.mean+self.sigma, color='green', ls='--')
        sp.legend()

