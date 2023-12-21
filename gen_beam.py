from scipy.constants import c, e, m_e
import numpy as np

from . import beam_profile
from . import lattice

electron_mass_eV = m_e * c**2 / e

def gen_beam2D(outp, n_particles, gemit, beta, alpha, cutoff_sigma, n_mesh):
    if gemit == 0:
        outp[0] = 0
        outp[1] = 0
        return
    n_particles = int(n_particles)
    gamma = (1 + alpha**2)/beta
    sigma_x = np.sqrt(gemit*beta)
    sigma_xp = np.sqrt(gemit*gamma)
    x_arr = np.linspace(-cutoff_sigma*sigma_x, cutoff_sigma*sigma_x, n_mesh)
    x_step = x_arr[1] - x_arr[0]
    xp_arr0 = np.linspace(-cutoff_sigma*sigma_xp, cutoff_sigma*sigma_xp, n_mesh)
    xp_step = xp_arr0[1] - xp_arr0[0]
    xp_arr = xp_arr0[:,np.newaxis]
    rho = 1/(2*np.pi*gemit) * np.exp(-1/(2*gemit) * (gamma*x_arr**2 + 2*alpha * x_arr * xp_arr + beta*xp_arr**2))
    rho1 = np.reshape(rho, rho.size)
    cum0 = np.cumsum(rho1)
    cum = cum0/cum0[-1]
    randoms = np.random.rand(n_particles)
    indices = np.arange(len(cum))
    interp_indices = np.round(np.interp(randoms, cum, indices))
    xp_index = (interp_indices % n_mesh).astype(int)
    x_index = ((interp_indices - xp_index) / n_mesh).astype(int)

    outp[0] = x_arr[x_index] + (np.random.rand(n_particles)-0.5) * x_step
    outp[1] = xp_arr0[xp_index] + (np.random.rand(n_particles)-0.5) * xp_step

    outp[0] -= outp[0].mean()
    outp[1] -= outp[1].mean()

def random_gen1D(x_axis, pdf, n_particles):
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    randoms = np.random.rand(n_particles)
    x = np.interp(randoms, cdf, x_axis)
    return x

def gen_beamT(n_particles, beamProfile):
    curr = beamProfile.charge_dist
    tt = beamProfile.time
    return random_gen1D(tt, curr, n_particles)

def gen_beam6D(nemitx, nemity, betax, alphax, betay, alphay, energy_eV, beamProfile, n_particles, cutoff_sigma=5, n_mesh=500):
    outp = np.zeros([6, int(n_particles)])
    gamma = energy_eV/electron_mass_eV
    gen_beam2D(outp[0:2,:], n_particles, nemitx/gamma, betax, alphax, cutoff_sigma, n_mesh)
    gen_beam2D(outp[2:4,:], n_particles, nemity/gamma, betay, alphay, cutoff_sigma, n_mesh)
    outp[4] = gen_beamT(n_particles, beamProfile)
    outp[5] = energy_eV
    return outp

def gen_beam4D(nemitx, nemity, betax, alphax, betay, alphay, energy_eV, n_particles, cutoff_sigma=5, n_mesh=500):
    outp = np.zeros([4, int(n_particles)])
    gamma = energy_eV/electron_mass_eV
    gen_beam2D(outp[0:2,:], n_particles, nemitx/gamma, betax, alphax, cutoff_sigma, n_mesh)
    gen_beam2D(outp[2:4,:], n_particles, nemity/gamma, betay, alphay, cutoff_sigma, n_mesh)
    return outp

def particles_to_profile(time_coordinates, time_grid, total_charge, energy_eV):
    dt = time_grid[1] - time_grid[0]
    bins = np.concatenate([time_grid, [time_grid[-1]+dt]]) - dt/2.
    hist, _ = np.histogram(time_coordinates, bins=bins)
    return beam_profile.BeamProfile(time_grid, hist, energy_eV, total_charge)

class Beam:
    """
    dimensions: list or tuple of ('x', 'y', 'p', 't')
    specifications: list or tuple of dictionaries with specifications for each dimension
        'x':
    """
    def __init__(self, arr, dim_index, beamProfile, total_charge, energy_eV):
        len_arr = len(arr)
        len_dim = len(dim_index)
        if len_arr != len_dim:
            raise ValueError(len(arr), dim_index.keys())
        self.arr = arr
        self.dim_index = dim_index
        self.beamProfile = beamProfile
        self.total_charge = total_charge
        self.n_particles = len(arr[0])
        self.energy_eV = energy_eV
        self.ignore_time = True

    def to_dict_custom(self):
        outp = {
                'arr': self.arr,
                'dim_index': self.dim_index,
                'beamProfile': self.beamProfile,
                'total_charge': self.total_charge,
                'energy_eV': self.energy_eV,
                }
        return outp

    def child(self, new_arr=None):
        if new_arr is None:
            new_arr = self.arr.copy()
        return Beam(new_arr, self.dim_index, self.beamProfile, self.total_charge, self.energy_eV)

    def to_profile(self, time_grid):
        return particles_to_profile(self['t'], time_grid, self.total_charge, self.energy_eV)

    def to_screen_dist(self, screen_bins, screen_smoothen, dim):
        return beam_profile.getScreenDistributionFromPoints(self[dim], screen_bins, screen_smoothen, total_charge=self.total_charge)

    def __getitem__(self, dim):
        return self.arr[self.dim_index[dim]]

    def __setitem__(self, dim, value):
        self.arr[self.dim_index[dim]] = value


    def linear_propagate(self, matrix6D):
        """
        Gets a 6 dimensional matrix in elegant format (according to lattice.py).
        Only takes the required entries according to which dimensions are in this beam.
        """
        matrix = np.zeros([len(self.dim_index)]*2, float)
        lat_dim_index = lattice.Lattice.dim_index
        for dim_row, index_row in self.dim_index.items():
            for dim_col, index_col in self.dim_index.items():
                value = matrix6D[lat_dim_index[dim_row], lat_dim_index[dim_col]]
                matrix[index_row,index_col] = value
        # We need to ignore time because we want to keep the time frame of reference at the streaker, not at the screen.
        if 't' in self.dim_index and self.ignore_time:
            t_index = self.dim_index['t']
            matrix[t_index,:] = 0
            matrix[:, t_index] = 0
            matrix[t_index, t_index] = 1
        arr = matrix @ self.arr
        return self.child(arr)

    def new_beamProfile(self, beamProfile):
        outp = self.child()
        outp.beamProfile = beamProfile
        tt = gen_beamT(outp.arr.shape[1], beamProfile)
        outp.arr[outp.dim_index['t']] = tt
        return outp

    def update_beamProfile(self):
        self.beamProfile = particles_to_profile(self['t'], self.beamProfile.time, self.total_charge, self.beamProfile.energy_eV)
        return self.beamProfile

    def set_beamProfile(self):
        raise NotImplementedError

    @staticmethod
    def get_emittance_from_points(x, xp):
        x = x - np.mean(x)
        xp = xp - np.mean(xp)
        return np.sqrt(np.mean(x**2)*np.mean(xp**2) - np.mean(x*xp)**2)

    def get_beamsize(self, dimension):
        assert dimension in ('x', 'y', 'xp', 'yp', 't')
        arr = self['%s' % dimension]-np.mean(self['%s' % dimension])
        return np.sqrt(np.mean(arr**2))

    def get_emittance_from_beam(self, dimension, normalized=False):
        assert dimension in ('x', 'y')

        space = self['%s' % dimension]
        angle = self['%sp' % dimension]
        if normalized:
            norm_factor = np.mean(self['p'])
        else:
            norm_factor = 1
        return self.get_emittance_from_points(space, angle)*norm_factor

    def get_beta_from_beam(self, dimension):
        assert dimension in ('x', 'y')
        e = self.get_emittance_from_beam(dimension)
        beamsize_squared = self.get_beamsize(dimension)**2
        return beamsize_squared/e

    def get_gamma_from_beam(self, dimension):
        assert dimension in ('x', 'y')
        e = self.get_emittance_from_beam(dimension)
        beamsize_squared = self.get_beamsize(dimension+'p')**2
        return beamsize_squared/e

    def get_alpha_from_beam(self, dimension):
        assert dimension in ('x', 'y')
        e = self.get_emittance_from_beam(dimension)
        arr_x0 = self[dimension]
        arr_xp0 = self[dimension+'p']

        arr_x = arr_x0 - arr_x0.mean()
        arr_xp = arr_xp0 - arr_xp0.mean()

        return -np.mean(arr_x*arr_xp)/e

    def slice_beam(self, n_bins, axis='t'):
        """
        Method may be const_delta or const_size
        """
        xx = self[axis]
        xx = xx - xx.mean()

        slices = []
        xx_min, xx_max = np.min(xx), np.max(xx)
        delta = (xx_max-xx_min)/n_bins
        for n_bin in range(n_bins):
            this_mask = np.logical_and(xx_min+n_bin*delta < xx, xx_min+(n_bin+1)*delta > xx)
            new_charge = self.total_charge / len(self.arr[0]) * this_mask.sum()
            new_slice = Beam(self.arr[:,this_mask], self.dim_index, None, new_charge, self.energy_eV)
            slices.append(new_slice)
            new_slice.delta = delta

        return slices


def beam_from_spec(dimensions, specifications, n_particles, beamProfile, total_charge, energy_eV):
    """
    Generate a Gaussian beam from specifications.
    Choose which dimensions the beam should include.
    Possbible: x, y, t, delta
    x and y generate two dimensions (x, xp or y, ypr.

    Format:
        - dimensions: for example ('x','y', 't')
        - specifications: dictionary which can have the following keys:
            - nemitx, nemity, betax, betay, alphax, alphay (no explaination needed)
            - n_mesh, cutoff_sigma: for transverse gaussian coordinate generation. See other functions in this file.
        - n_particles: integer
        - beamProfile: look at beam_profile.py
        - total_charge: in Coulomb
        - energy_eV: enegy in eV
    """

    n_particles = int(n_particles)
    n_dim = 0
    if 'x' in dimensions:
        n_dim += 2
    if 'y' in dimensions:
        n_dim += 2
    if 'delta' in dimensions:
        n_dim += 1
    if 't' in dimensions:
        n_dim += 1
    arr = np.empty([n_dim, n_particles])
    n_dim = 0
    dim_index = {}
    s = specifications
    gamma_rel = energy_eV/electron_mass_eV
    for t_dim in ('x', 'y'):
        if t_dim in dimensions:
            gen_beam2D(arr[n_dim:n_dim+2], n_particles, s['nemit'+t_dim]/gamma_rel, s['beta'+t_dim], s['alpha'+t_dim], s['cutoff_sigma'], s['n_mesh'])
            dim_index[t_dim] = n_dim
            dim_index[t_dim+'p'] = n_dim+1
            n_dim += 2
    if 't' in dimensions:
        dim_index['t'] = n_dim
        arr[n_dim] = gen_beamT(n_particles, beamProfile)
        n_dim += 1
    if 'delta' in dimensions:
        if specifications['energy_chirp']:
            arr[n_dim] = (arr[dim_index['t']]-np.mean(arr[dim_index['t']]))*specifications['energy_chirp']/energy_eV
        else:
            arr[n_dim] = 0
        if specifications['energy_spread']:
            arr[n_dim] += np.random.randn(len(arr[n_dim]))*specifications['energy_spread']/energy_eV
        dim_index['delta'] = n_dim
        n_dim += 1

    beam = Beam(arr, dim_index, beamProfile, total_charge, energy_eV)
    beam.specifications = specifications
    return beam

