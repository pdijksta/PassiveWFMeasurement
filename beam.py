import numpy as np

def gen_twoD(outp, n_particles, gemit, beta, alpha, cutoff_sigma, n_mesh):
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
    cum = np.cumsum(rho1) * xp_step * x_step
    cum /= cum.max()
    randoms = np.random.rand(int(n_particles))
    indices = np.arange(len(cum))
    interp_indices = np.round(np.interp(randoms, cum, indices))
    xp_index = (interp_indices % n_mesh).astype(int)
    x_index = ((interp_indices - xp_index) / n_mesh).astype(int)

    outp[0] = x_arr[x_index] + (np.random.rand(int(n_particles))-0.5) * x_step
    outp[1] = xp_arr0[x_index] + (np.random.rand(int(n_particles))-0.5) * xp_step

def gen_beam6D(nemitx, nemity, betax, alphax, betay, alphay, p_central, beamProfile, n_particles, cutoff_sigma=5, n_mesh=200):
    outp = np.zeros([6, int(n_particles)])
    gen_twoD(outp[0:2,:], n_particles, nemitx/p_central, betax, alphax, p_central, cutoff_sigma, n_mesh)
    gen_twoD(outp[2:4,:], n_particles, nemity/p_central, betay, alphay, p_central, cutoff_sigma, n_mesh)

    outp[5] = p_central





if __name__ == '__main__':
    nemitx = 300e-9
    nemity = 300e-9
    betax = 10
    alphax = 1
    betay = 5
    alphay = -.5
    p_central = 6e9/511e3
    rms_bunch_duration = 50e-15
    n_particles = 100e3
    cutoff_sigma = 5
    gen_beam6D(nemitx, nemity, betax, alphax, betay, alphay, p_central, rms_bunch_duration, n_particles, cutoff_sima=cutoff_sigma)



