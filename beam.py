from scipy.constants import c, e, m_e
import numpy as np

electron_mass_eV = m_e * c**2 / e

def gen_twoD(outp, n_particles, gemit, beta, alpha, cutoff_sigma, n_mesh):
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

def gen_beam6D(nemitx, nemity, betax, alphax, betay, alphay, p_central, beamProfile, n_particles, cutoff_sigma=5, n_mesh=500):
    outp = np.zeros([6, int(n_particles)])
    gen_twoD(outp[0:2,:], n_particles, nemitx/p_central, betax, alphax, cutoff_sigma, n_mesh)
    gen_twoD(outp[2:4,:], n_particles, nemity/p_central, betay, alphay, cutoff_sigma, n_mesh)

    curr = beamProfile.current
    tt = beamProfile.time
    integrated_curr = np.cumsum(curr)
    integrated_curr /= integrated_curr[-1]
    randoms = np.random.rand(n_particles)
    interp_tt = np.interp(randoms, integrated_curr, tt)
    interp_tt -= interp_tt.min()
    outp[4] = interp_tt
    outp[5] = p_central
    return outp

def gen_beam4D(nemitx, nemity, betax, alphax, betay, alphay, p_central, n_particles, cutoff_sigma=5, n_mesh=500):
    outp = np.zeros([4, int(n_particles)])
    gen_twoD(outp[0:2,:], n_particles, nemitx/p_central, betax, alphax, cutoff_sigma, n_mesh)
    gen_twoD(outp[2:4,:], n_particles, nemity/p_central, betay, alphay, cutoff_sigma, n_mesh)
    return outp

