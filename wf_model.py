import numpy as np
from scipy.constants import physical_constants, c
from numpy import cos, sin, tan, exp, sqrt, pi

Z0 = physical_constants['characteristic impedance of vacuum'][0]
t1 = Z0 * c / (4*pi)

def PostAramisStreaker(a):
    """
    CorrugatedStreaker with parameters specified by Paolo Craievich (personal communication)
    """
    Ls = 1.
    #delta = 250e-6 # Irrelevant
    p = 500e-6
    g = 250e-6
    w = 10e-3
    return CorrugatedStreaker(a, p, g, w, Ls)

class CorrugatedStreaker:
    """
    Following Bane, Stupakov, Zagorodnov, Analytical formulas for short bunch wakes in a flat dechirper, PRAB 19, 084401 (2016)
    DOI: 10.1103/PhysRevAccelBeams.19.084401
    p         period
    g         longitudinal gap
    w         plate width
    Ls        Length of streaker
    """
    def __init__(self, p, g, w, Ls):
        self.p = p
        self.g = g
        self.w = w
        self.Ls = Ls
        self.alpha = 1. - 0.465*sqrt(g/p) - 0.070*g/p
        self.s0d = self.s0r * (15/14)**2

    def s0r(self, a):
        return (a**2 * self.g) / (2*pi * self.alpha**2 * self.p**2)

    def wxd_lin_dipole(self, s, a, x):
        """
        Single particle wake, linear dipole approximation.
        Unit: V/m /m (offset)
        """
        t2 = pi**4 / (16*a**4)
        t3 = self.s0d
        sqr = sqrt(s/t3)
        t4 = 1 - (1 + sqr)*exp(-sqr)
        return t1*t2*t3*t4*x*self.Ls

    def s0yd(self, a, x):
        arg = pi*x/a
        return 4*self.s0r(a) * (3/2 + arg/sin(arg) - arg/(2*tan(arg)))**(-2)

    def wxd(self, s, a, x):
        t2 = pi**3 / (4*a**3)
        arg = pi*x/(2*a)
        t3 = 1./cos(arg)**2
        t4 = tan(arg)
        t5 = self.s0yd(x)
        sqr = sqrt(s/t5)
        t6 = 1 - (1 + sqr)*exp(-sqr)
        return t1 * t2 * t3 * t4 * t5 * t6 * self.Ls

    def s0yq(self, a, x):
        t0 = 4*self.s0r(a)
        theta = (pi*x)/(2*a)
        t1 = (56-cos(2*theta))/30
        t2 = (0.3+theta*sin(2*theta))/(2-cos(2*theta))
        t3 = 2*theta*tan(theta)
        return t0 * (t1 + t2 + t3)**(-2)

    def wxq(self, s, a, x):
        arg = pi*x/a
        t2 = pi**4/(16*a**4)
        t3 = 2 - cos(arg)
        t4 = 1/cos(arg/2)**4
        t5 = self.s0yq(a, x)
        sqr = sqrt(s/t5)
        t6 = 1 - (1+sqr)*exp(-sqr)
        result = t1 * t2 * t3 * t4 * t5 * t6 * self.Ls
        return result

    def s0l(self, a, x):
        arg = (pi*x)/(2*a)
        return 4*self.s0r(a) * (1. + 1./3.*cos(arg)**2 + arg*tan(arg))**(-2)

    def wld(self, s, a, x):
        t2 = pi**2/(4*a**2)
        t3 = cos((pi*x)/(2*a))**(-2)
        s0l_ = self.s0l(a, x)
        t4 = exp(-sqrt(s/s0l_))
        return t1 * t2 * t3 * t4 * self.Ls

    def generate_elegant_wf(self, filename, xx, semigap, beam_offset):
        xx -= xx.min()
        assert np.all(xx >= 0)
        if beam_offset == 0:
            w_wxd = np.zeros_like(xx)
        else:
            w_wxd = self.wxd(xx, semigap, beam_offset)
        delta_offset = 1e-6
        w_wxd2 = self.wxd(xx, semigap, beam_offset+delta_offset)
        w_wxd_deriv = (w_wxd2 - w_wxd)/delta_offset
        w_wld = self.wld(xx, semigap, beam_offset)
        tt = xx/c
        comment_str = 'semigap %.5e m ; beam_offset %.5e m ; Length %.5e m' % (semigap, beam_offset, self.Ls)
        return write_sdds(filename, tt, w_wld, w_wxd, w_wxd_deriv, comment_str)

def wf2d(s_coords, x_coords, semigap, charge, wf_func, hist_bins=(int(1e3), 100)):

    beam_hist, s_edges, x_edges = np.histogram2d(s_coords, x_coords, hist_bins)
    beam_hist *= charge / beam_hist.sum()

    s_edges = s_edges[:-1] + (s_edges[1] - s_edges[0])*0.5
    x_edges = x_edges[:-1] + (x_edges[1] - x_edges[0])*0.5

    wake2d = wf_func(s_edges[:, np.newaxis], semigap, x_edges)
    wake = np.zeros_like(s_edges)

    for n_output in range(len(wake)):
        for n2 in range(0, n_output+1):
            wake[n_output] += (beam_hist[n2,:] * wake2d[n_output-n2,:]).sum()

    wake_on_particles = np.interp(s_coords, s_edges, wake) * np.sign(charge)

    output = {
            'wake': wake,
            'beam_hist': beam_hist,
            's_bins': s_edges,
            'x_bins': x_edges,
            'spw2d': wake2d,
            'wake_on_particles': wake_on_particles,
            }
    return output

def wf2d_quad(self, s_coords, x_coords, semigap, charge, wf_func, hist_bins=(int(1e3), 100)):
    """
    Respects sign of the charge
    """

    beam_hist, s_edges, x_edges = np.histogram2d(s_coords, x_coords, hist_bins)
    beam_hist *= charge / beam_hist.sum()

    s_edges = s_edges[:-1] + (s_edges[1] - s_edges[0])*0.5
    x_edges = x_edges[:-1] + (x_edges[1] - x_edges[0])*0.5

    wake2d = wf_func(s_edges[:, np.newaxis], semigap, x_edges)
    wake = np.zeros([len(s_edges), len(x_edges)])

    for n_output in range(wake.shape[0]):
        for n2 in range(0, n_output+1):
            wake0 = beam_hist[n2,:] * wake2d[n_output-n2,:]
            c1 = wake0.sum() * x_edges
            c2 = (wake0 * x_edges).sum()
            wake[n_output,:] += c1 - c2

    wake *= np.sign(charge)

    ## A much faster interpolation over the grid than from scipy.interpolate.interp2d
    indices = []
    indices_delta = []
    delta_s = np.zeros_like(wake)
    delta_x = delta_s.copy()

    ## Derivative of wake in both directions
    delta_s[:-1,:] = wake[1:,:] - wake[:-1,:]
    delta_x[:,:-1] = wake[:,1:] - wake[:,:-1]

    ## Calculate the indices of the histogram for each point, then also add correction from first derivative
    for grid_points, points in [(s_edges, s_coords), (x_edges, x_coords)]:
        index_float = (points - grid_points[0]) / (grid_points[1] - grid_points[0])
        index = index_float.astype(int)
        indices_delta.append(index_float-index)
        np.clip(index, 0, len(grid_points)-1, out=index)
        indices.append(index)

    wake_on_particles = wake[indices[0], indices[1]]
    # Apply derivative
    correction = delta_s[indices[0], indices[1]] * indices_delta[0] + delta_x[indices[0], indices[1]] * indices_delta[1]
    wake_on_particles += correction

    output = {
            'wake': wake,
            'wake_on_particles': wake_on_particles,
            'beam_hist': beam_hist,
            's_bins': s_edges,
            'x_bins': x_edges,
            'spw2d': wake2d,
            }
    return output

def write_sdds(filename, tt, w_wld, w_wxd, w_wxd_deriv, comment_str=''):

    with open(filename, 'w') as fid:
        fid.write('SDDS1\n')
        #fid.write('&column name=z,    units=m,    type=double,    &end\n')
        fid.write('&column name=t,    units=s,    type=double,    &end\n')
        fid.write('&column name=W,    units=V/C,  type=double,    &end\n')
        fid.write('&column name=WX,   units=V/C,    type=double,    &end\n') # V/C for X_DRIVE_EXPONENT=0, otherwise V/C/m
        fid.write('&column name=DWX,   units=V/C/m,    type=double,    &end\n')
        fid.write('&data mode=ascii, &end\n')
        fid.write('! page number 1\n')
        fid.write('! %s\n' % comment_str)
        fid.write('%i\n' % len(tt))
        for t, wx, wl, dwx in zip(tt, w_wld, w_wxd, w_wxd_deriv):
            fid.write('  %12.6e  %12.6e  %12.6e  %12.6e\n' % (t, wx, wl, dwx))

    return {
            't': tt,
            'W': w_wld,
            'WX': w_wxd,
            'DWX': w_wxd_deriv,
            }

