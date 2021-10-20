from scipy.constants import c, e, m_e
import numpy as np

from . import beam_profile

electron_mass_eV = m_e * c**2 / e

def gen_beam2D(outp, n_particles, gemit, beta, alpha, cutoff_sigma, n_mesh):
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

def gen_beamT(n_particles, beamProfile):
    curr = beamProfile.current
    tt = beamProfile.time
    integrated_curr = np.cumsum(curr)
    integrated_curr /= integrated_curr[-1]
    randoms = np.random.rand(n_particles)
    interp_tt = np.interp(randoms, integrated_curr, tt)
    interp_tt -= interp_tt.min()
    return interp_tt

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

def particles_to_profile(time_coordinates, time_grid, charge, energy_eV):
    dt = time_grid[1] - time_grid[0]
    bins = np.concatenate(time_grid, [time_grid[-1]+dt]) - dt/2.
    hist, _ = np.histogram(time_coordinates, bins=bins)
    return beam_profile.BeamProfile(time_grid, hist, energy_eV, charge)

class Beam:
    """
    dimensions: list or tuple of ('x', 'y', 'p', 't')
    specifications: list or tuple of dictionaries with specifications for each dimension
        'x':
    """
    def __init__(self, dimensions, specifications, n_particles, beamProfile, charge):
        self.charge = charge
        self.specifications = specifications
        self.n_particles = n_particles
        n_dim = 0
        if 'x' in dimensions:
            n_dim += 2
        if 'y' in dimensions:
            n_dim += 2
        if 'p' in dimensions:
            n_dim += 1
        if 't' in dimensions:
            n_dim += 1
        self.arr = arr = np.empty([n_dim, n_particles])
        n_dim = 0
        dim_index = {}
        s = specifications
        ene = specifications['energy_eV']
        gamma_rel = ene/electron_mass_eV
        for t_dim in 'x', 'y':
            if t_dim in dimensions:
                gen_beam2D(arr[n_dim:n_dim+2], n_particles, s['nemit'+t_dim]/gamma_rel, s['beta'+t_dim], s['alpha'+t_dim], s['cutoff_sigma'], s['n_mesh'])
                dim_index[t_dim] = n_dim
                n_dim += 2
        if 't' in dimensions:
            dim_index['t'] = n_dim
            self.gen_time(beamProfile)
            n_dim += 1
        if 'p' in dimensions:
            arr[n_dim] = specifications['energy_eV']
            dim_index['p'] = n_dim
            n_dim += 1

        self.dim_index = dim_index

    def gen_time(self, beamProfile):
        self.arr[self.dim_index['t']] = gen_beamT(self.n_particles, beamProfile)
        self.beamProfile = beamProfile

    def to_profile(self, time_grid):
        return particles_to_profile(self.arr[self.dim_index['t']], time_grid, self.charge)




