import os
import numpy as np
from scipy.interpolate import griddata

correction_table = np.loadtxt(os.path.join(os.path.dirname(__file__), './data_BPMA_correction_x.txt'), delimiter=',', comments='#')
xi, yi, xc = correction_table.T*1e-3

def bpm_correction(x_in, y_in, method='table'):
    x = np.cos(np.pi/4)*x_in - np.sin(np.pi/4)*y_in
    y = np.sin(np.pi/4)*x_in + np.cos(np.pi/4)*y_in

    def poly(x, y):
        x = x*1e3
        y = y*1e3

        a = 1.946336e-5
        b = 1.361453e-3
        c = 1.00665
        d = 5.776100e-5
        e = 2.470538e-5
        f = 1.906282e-3
        g = 1.374032e-9
        h = -3.975232e-11
        i = 3.582962e-13
        x_out = a*x**5 + b*x**3 + c*x + d*x**3*y**2 + e*x*y**4 + f*x*y**2 + g*x**7*y**4 + h*x**9*y**4 + i*x**11*y**4
        return x_out/1e3

    if method == 'poly':
        x_corr = poly(x, y)
        y_corr = poly(y, x)
    elif method == 'table':
        x_corr = griddata((xi, yi), xc, (x, y), method='cubic')
        y_corr = griddata((xi, yi), xc, (y, x), method='cubic')
    else:
        raise ValueError(method)

    x_out = np.cos(np.pi/4)*x_corr + np.sin(np.pi/4)*y_corr
    y_out = - np.sin(np.pi/4)*x_corr + np.cos(np.pi/4)*y_corr
    return x_out, y_out

