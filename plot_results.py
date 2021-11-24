import numpy as np
import matplotlib.pyplot as plt
from . import myplotstyle as ms
from . import config
from . import data_loader
from . import image_analysis
from . import beam_profile

class dummy_plot:
    def __init__(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        class dummy_return:
            def get_color(self):
                return None
        return [dummy_return(),]

    def dummy(self, *args, **kwargs):
        pass

    errorbar = plot
    legend = dummy
    axhline = dummy
    axvline = dummy
    grid = dummy
    set_title = dummy
    set_xlabel = dummy
    set_ylabel = dummy
    clear = dummy
    scatter = dummy
    set_ylim = dummy
    set_xlim = dummy
    set_yticklabels = dummy
    set_yticks = dummy
    imshow = dummy

def plot_gap_reconstruction(gap_recon_dict, plot_handles=None, figsize=None, exclude_gap_ctrs=()):
    if plot_handles is None:
        fig, plot_handles = gap_recon_figure(figsize=figsize)
    (sp_rms, sp_overview, sp_std, sp_fit, sp_distances) = plot_handles

    gap_arr = gap_recon_dict['gap_arr']
    all_rms_arr = gap_recon_dict['all_rms']
    lin_fit = gap_recon_dict['lin_fit']
    lin_fit_const = gap_recon_dict['lin_fit_const']
    gap0 = gap_recon_dict['gap0']

    beam_positions = gap_recon_dict['final_beam_positions']

    for gap_ctr in list(range(len(gap_arr)))[::-1]:
        if gap_ctr in exclude_gap_ctrs:
            continue
        gap = gap_arr[gap_ctr]
        distance_arr = gap/2. - np.abs(beam_positions)
        d_arr2 = distance_arr - distance_arr.min()
        sort = np.argsort(d_arr2)
        _label = '%i' % round((gap-gap0)*1e6)
        #sp_centroid.plot(d_arr2, centroid_arr, label=_label)
        rms_arr = all_rms_arr[gap_ctr]
        #color = ms.colorprog(gap_ctr, gap_arr)
        color= sp_rms.plot(d_arr2[sort]*1e6, rms_arr[sort]*1e15, label=_label, marker='.')[0].get_color()
        fit_yy = lin_fit_const[gap_ctr] + lin_fit[gap_ctr]*d_arr2[sort]
        sp_rms.plot(d_arr2[sort]*1e6, fit_yy*1e15, color=color, ls='--')


    sp_overview.errorbar(gap_arr*1e3, all_rms_arr.mean(axis=-1)*1e15, yerr=all_rms_arr.std(axis=-1)*1e15)
    sp_std.plot(gap_arr*1e3, all_rms_arr.std(axis=-1)/all_rms_arr.mean(axis=-1), marker='.')
    sp_fit.plot(gap_arr*1e3, lin_fit*1e15/1e6, marker='.')

    sp_fit.axhline(0, color='black', ls='--')
    sp_fit.axvline(gap_recon_dict['gap']*1e3, color='black', ls='--')

    sp_rms.legend(title='$\Delta$g ($\mu$m)')

    mask_pos = beam_positions > 0
    mask_neg = beam_positions < 0

    distances = gap_recon_dict['gap']/2. - np.abs(beam_positions)
    best_index = gap_recon_dict['best_index']
    for mask, label in [(mask_pos, 'Positive'), (mask_neg, 'Negative')]:
        sp_distances.plot(distances[mask]*1e3, all_rms_arr[best_index][mask]*1e15, label=label)

    sp_distances.legend()

def plot_reconstruction(gauss_dicts, plot_handles=None, blmeas_profile=None, max_distance=350e-6, type_='centroid', figsize=None):
    center = 'Mean'
    if plot_handles is None:
        fig, (sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg) = gauss_recon_figure(figsize)
    else:
        fig, (sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg) = plot_handles

    if len(gauss_dicts) == 0:
        raise ValueError

    for gauss_dict in gauss_dicts:
        beam_position = gauss_dict['beam_position']

        sp_screen = sp_screen_pos if beam_position > 0 else sp_screen_neg
        sp_profile = sp_profile_pos if beam_position > 0 else sp_profile_neg

        semigap = gauss_dict['gap']/2.
        distance = semigap-abs(beam_position)
        if max_distance and distance > max_distance:
            continue

        rec_profile = gauss_dict['reconstructed_profile']
        label = '%i' % (round(rec_profile.rms()*1e15))
        rec_profile.plot_standard(sp_profile, label=label, center=center)

        meas_screen = gauss_dict['meas_screen_raw']
        rec_screen = gauss_dict['reconstructed_screen']
        label = '%i' % round(distance*1e6)
        color = meas_screen.plot_standard(sp_screen, label=label)[0].get_color()
        rec_screen.plot_standard(sp_screen, ls='--', color=color)

    if blmeas_profile is not None:
        for _sp in sp_profile_pos, sp_profile_neg:
            blmeas_profile.plot_standard(_sp, color='black', center=center, ls='--', label='%i' % round(blmeas_profile.rms()*1e15))

    for _sp in sp_screen_pos, sp_screen_neg:
        _sp.legend(title='d ($\mu$m)')

    for _sp in sp_profile_pos, sp_profile_neg:
        _sp.legend(title='rms (fs)')

def gauss_recon_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Current reconstruction')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 2, grid=False)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+4)))
    clear_gauss_recon(*plot_handles)
    return fig, plot_handles

def clear_gauss_recon(sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg):
    for sp, title, xlabel, ylabel in [
            (sp_screen_pos, 'Screen profile (+)', 'x (mm)', config.rho_label),
            (sp_screen_neg, 'Screen profile (-)', 'x (mm)', config.rho_label),
            (sp_profile_pos, 'Beam current (+)', 't (fs)', 'Current (kA)'),
            (sp_profile_neg, 'Beam current (-)', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def structure_fit_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker center calibration')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 3)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+6)))
    clear_structure_fit(*plot_handles)
    return fig, plot_handles

def clear_structure_fit(sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Centroid shift', 'Streaker center (mm)', 'Beam X centroid (mm)'),
            (sp_sizes, 'Size increase', 'Streaker center (mm)', 'Beam X rms (mm)'),
            (sp_center2, 'Centroid shift', 'Distance from jaw ($\mu$m)', 'Beam X centroid (mm)'),
            (sp_sizes2, 'Size increase', 'Distance from jaw ($\mu$m)', 'Beam X rms (mm)'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            (sp_current, 'Beam current', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def gap_recon_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker gap reconstruction')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 3, grid=False)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+5)))
    clear_gap_recon(*plot_handles)
    return fig, plot_handles

def clear_gap_recon(sp_rms, sp_overview, sp_std, sp_fit, sp_distances):
    for sp, title, xlabel, ylabel in [
            (sp_rms, 'Rms bunch duration', '$\Delta$d ($\mu$m)', 'I rms (fs)'),
            (sp_overview, 'Rms bunch duration', 'Gap (mm)', 'rms (fs)'),
            (sp_std, 'Relative beamsized error', 'Gap (mm)', r'$\Delta \tau / \tau$'),
            (sp_fit, 'Fit coefficient', 'Gap (mm)', r'$\Delta \tau / \Delta$Gap (fs/$\mu$m)'),
            (sp_distances, 'Final distances', 'd (mm)', 'I rms (fs)')
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def plot_structure_position0_fit(fit_dicts, plot_handles=None, figsize=None, blmeas_profile=None, sim_screens=None):

    if plot_handles is None:
        fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = structure_fit_figure(figsize)
    else:
        (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = plot_handles

    fit_dict_centroid = fit_dicts['centroid']
    raw_struct_positions = fit_dict_centroid['raw_struct_positions']
    fit_dict_rms = fit_dicts['beamsize']
    forward_propagate_blmeas = (blmeas_profile is not None)
    centroids = fit_dict_centroid['centroids']
    centroids_std = fit_dict_centroid['centroids_std']
    rms = fit_dict_centroid['rms']
    rms_std = fit_dict_centroid['rms_std']
    meas_screens = fit_dict_centroid['meas_screens']

    rms_sim = np.zeros(len(raw_struct_positions))
    centroid_sim = np.zeros(len(raw_struct_positions))
    if sim_screens is not None:
        for n_proj, (meas_screen, position) in enumerate(zip(meas_screens, raw_struct_positions)):
            color = ms.colorprog(n_proj, raw_struct_positions)
            meas_screen.plot_standard(sp_proj, label='%.2f mm' % (position*1e3), color=color)
            sim_screen = sim_screens[n_proj]
            sim_screen.plot_standard(sp_proj, color=color, ls='--')
            centroid_sim[n_proj] = sim_screen.mean()
            rms_sim[n_proj] = sim_screen.rms()

    if forward_propagate_blmeas:
        blmeas_profile.plot_standard(sp_current, color='black', ls='--')

    for fit_dict, sp1, sp2, yy, yy_err, yy_sim in [
            (fit_dict_centroid, sp_center, sp_center2, centroids, centroids_std, centroid_sim),
            (fit_dict_rms, sp_sizes, sp_sizes2, rms, rms_std, rms_sim),
            ]:

        xx_fit = fit_dict['xx_fit']
        reconstruction = fit_dict['reconstruction']


        sp1.errorbar(raw_struct_positions*1e3, yy*1e3, yerr=yy_err*1e3, label='Data', ls='None', marker='o')
        sp1.plot(xx_fit*1e3, reconstruction*1e3, label='Fit')
        if sim_screens is not None:
            sp1.plot(raw_struct_positions*1e3, yy_sim*1e3, label='Simulated', marker='.', ls='None')
        title = sp1.get_title()
        sp1.set_title('%s; Gap=%.2f mm' % (title, fit_dict['gap_fit']*1e3), fontsize=config.fontsize)

        structure_position0 = fit_dict['structure_position0']
        beam_positions = -(raw_struct_positions - structure_position0)
        beam_positions_fit = -(xx_fit - structure_position0)
        gap = fit_dict['gap_fit']
        distances = gap/2. - np.abs(beam_positions)
        distances_fit = gap/2. - np.abs(beam_positions_fit)

        mask_pos = np.logical_and(beam_positions > 0, raw_struct_positions != 0)
        mask_neg = np.logical_and(beam_positions < 0, raw_struct_positions != 0)
        for mask2, label in [(mask_pos, 'Positive'), (mask_neg, 'Negative')]:
            sp2.errorbar(distances[mask2]*1e6, np.abs(yy[mask2])*1e3, yerr=yy_err[mask2]*1e3, label=label, marker='o', ls='None')
        lims = sp2.get_xlim()
        mask_fit = np.logical_and(distances_fit*1e6 > lims[0], distances_fit*1e6 < lims[1])
        xx_fit2 = distances_fit[mask_fit]
        yy_fit2 = reconstruction[mask_fit]
        sp2.plot(xx_fit2*1e6, np.abs(yy_fit2)*1e3, label='Fit')

        if sim_screens is not None:
            plot2_sim = []
            for mask in mask_pos, mask_neg:
                plot2_sim.extend([(a, np.abs(b)) for a, b in zip(xx_fit2[mask], yy_sim[mask])])
            plot2_sim.sort()
            xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
            xx_plot_sim = np.array(xx_plot_sim)
            yy_plot_sim = np.array(yy_plot_sim)
            sp2.plot(xx_plot_sim*1e6, yy_plot_sim*1e3, label='Simulated', ls='None', marker='o')

        title = sp2.get_title()
        sp2.set_title('%s; Center=%i $\mu$m' % (title, round(fit_dict['structure_position0']*1e6)), fontsize=config.fontsize)
        sp1.legend()
        sp2.legend()

def calib_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 6.4]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Gap and structure position calibration')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ny, nx = 2, 3
    subplot = ms.subplot_factory(ny, nx, grid=False)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+6)))
    clear_calib(*plot_handles)
    return fig, plot_handles


def clear_calib(sp_raw, sp_heat, sp_heat_rms, sp_heat_diff, sp_comb, sp_final):
    for sp, title, xlabel, ylabel in [
            (sp_raw, 'Raw data', '$\Delta$ g ($\mu$ m)', 'rms duration (fs)'),
            (sp_heat, 'Fit', '$\Delta$ g ($\mu$m)', '$\Delta$ center ($\mu$m)'),
            (sp_heat_rms, 'Rms duration', '$\Delta$ g ($\mu$m)', '$\Delta$ center ($\mu$m)'),
            (sp_heat_diff, 'Rms duration difference', '$\Delta$ g ($\mu$m)', '$\Delta$ center ($\mu$m)'),
            (sp_comb, 'Target', '$\Delta$ g ($\mu$m)', '$\Delta$ center ($\mu$m)'),
            (sp_final, 'New fits', 'distances ($\mu$m)', 'rms duration (fs)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def plot_calib(calib_dict, fig=None, plot_handles=None):
    delta_gap_range = calib_dict['delta_gap_range']
    delta_structure0_range = calib_dict['delta_structure0_range']
    fit_coefficients2 = calib_dict['fit_coefficients2']
    mean_rms_arr = calib_dict['mean_rms']
    diff_sides = calib_dict['diff_sides']
    argmin = calib_dict['best_index']
    distance_rms_arr = calib_dict['distance_rms_arr']
    beam_positions = calib_dict['beam_positions']

    if fig is None:
        fig, plot_handles = calib_figure()
    sp_raw, sp_heat, sp_heat_rms, sp_heat_diff, sp_comb, sp_final = plot_handles

    for n in range(len(distance_rms_arr)):
        distance_arr = distance_rms_arr[n,:,0]
        rms_arr = distance_rms_arr[n,:,1]
        distance_plot = distance_arr - distance_arr.min()
        sort = np.argsort(distance_plot)
        label = '%.2f' % (beam_positions[n]*1e3)
        sp_raw.plot(distance_plot[sort]*1e6, rms_arr[sort]*1e15, label=label, marker='.')

    x_axis = delta_gap_range
    y_axis = delta_structure0_range
    x_factor = y_factor = 1e6
    extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[-1]*y_factor, y_axis[0]*y_factor]
    #extent = None
    plot = sp_heat.imshow(fit_coefficients2, cmap='hot', extent=extent, aspect='auto')
    fig.colorbar(plot, label='Fit coefficint (arb. units)', ax=sp_heat)

    plot = sp_heat_rms.imshow(mean_rms_arr*1e15, cmap='hot', extent=extent, aspect='auto')
    fig.colorbar(plot, label='Profile rms (fs)', ax=sp_heat_rms)


    plot = sp_heat_diff.imshow(diff_sides*1e15, cmap='hot', extent=extent, aspect='auto')
    fig.colorbar(plot, label='Profile rms delta (fs)', ax=sp_heat_diff)

    combined_target = calib_dict['combined_target']
    plot = sp_comb.imshow(np.sqrt(combined_target), cmap='hot', extent=extent, aspect='auto')
    fig.colorbar(plot, label='Optimization function (arb. units)', ax=sp_comb)

    n12_pairs = []
    for n1 in [0, len(delta_structure0_range)-1]:
        for n2 in [0, len(delta_gap_range)-1]:
            n12_pairs.append([n1, n2])
    n12_pairs.append(argmin)
    mask_pos, mask_neg = beam_positions > 0, beam_positions < 0

    for n1, n2 in n12_pairs:
        delta_structure0 = delta_structure0_range[n1]
        delta_gap = delta_gap_range[n2]
        fit_dict = calib_dict['all_fit_dicts'][n1][n2]
        new_distances = fit_dict['new_distances']
        new_rms = fit_dict['new_rms']
        label = '%i / %i / %i' % (round(delta_structure0*1e6), delta_gap*1e6, new_rms.mean()*1e15)
        color = sp_final.plot(new_distances[mask_pos]*1e6, new_rms[mask_pos]*1e15, label=label, ls='None', marker='.')[0].get_color()
        sp_final.plot(new_distances[mask_neg]*1e6, new_rms[mask_neg]*1e15, color=color, ls='None', marker='o')
        xx_fit = np.array(new_distances)
        yy_fit = fit_dict['fit'](xx_fit)
        sp_final.plot(xx_fit*1e6, yy_fit*1e15, color=color, ls='dotted')

    sp_raw.legend(title='Beam position (mm)')
    sp_final.legend(title='$\Delta p_0$ ($\mu$m) / $\Delta$g ($\mu$m) / rms (fs)')

def resolution_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Screen center calibration')
    fig.subplots_adjust(hspace=0.35)
    subplot = ms.subplot_factory(1, 1)
    sp_res = subplot(1)
    sp_curr = sp_res.twinx()
    clear_resolution_figure(sp_curr, sp_res)
    return fig, (sp_curr, sp_res)

def clear_resolution_figure(sp_curr, sp_res):
    for sp, title, xlabel, ylabel in [
            (sp_res, 'Expected resolution', 't (fs)', 'R (fs)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp_res.grid(True)
    sp_curr.set_ylabel('I (kA)')

def screen_calibration_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Screen center calibration')
    fig.subplots_adjust(hspace=0.35)
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 1)

    sp_proj = subplot(sp_ctr, sciy=True)
    sp_ctr += 1
    clear_screen_calibration(sp_proj)
    return fig, (sp_proj, )

def clear_screen_calibration(sp_proj):
    for sp, title, xlabel, ylabel in [
            (sp_proj, 'Unstreaked beam', 'x (mm)', config.rho_label),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def reconstruction_figure(figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Current reconstruction')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,2)
    subplots = [subplot(sp_ctr) for sp_ctr in range(1, 1+4)]
    clear_reconstruction(*subplots)
    return fig, subplots

def clear_reconstruction(sp_screen, sp_profile, sp_opt, sp_moments):
    for sp, title, xlabel, ylabel in [
            (sp_screen, 'Screen', 'x (mm)', config.rho_label),
            (sp_profile, 'Profile', 't (fs)', 'Current (kA)'),
            (sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
            (sp_moments, 'Moments', 'Gaussian $\sigma$ (fs)', r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

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

def lasing_figure(figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Lasing reconstruction')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(3,3)
    subplots = [subplot(sp_ctr) for sp_ctr in range(1, 1+9)]
    clear_lasing_figure(*subplots)
    return fig, subplots

def clear_lasing_figure(sp_image_on, sp_image_on2, sp_image_off, sp_slice_mean, sp_slice_sigma, sp_current, sp_lasing_loss, sp_lasing_spread, sp_orbit):

    for sp, title, xlabel, ylabel in [
            (sp_image_on, 'Lasing On', 'x (mm)', 'y (mm)'),
            (sp_image_on2, 'Lasing On', 't (fs)', '$\Delta E$ (MeV)'),
            (sp_image_off, 'Lasing Off', 't (fs)', '$\Delta E$ (MeV)'),
            (sp_slice_mean, 'Energy loss', 't (fs)', '$\Delta E$ (MeV)'),
            (sp_slice_sigma, 'Energy spread increase', 't (fs)', 'Energy spread (MeV)'),
            (sp_current, 'Current profile', 't (fs)', 'Current (kA)'),
            (sp_lasing_loss, 'Energy loss power profile', 't (fs)', 'Power (GW)'),
            (sp_lasing_spread, 'Energy spread power profile', 't (fs)', 'Power (GW)'),
            (sp_orbit, 'Orbit jitter', r'Screen $\left|\langle x \rangle\right|$ (mm)', '$\Delta$d ($\mu$m)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def plot_rec_gauss(gauss_dict, plot_handles=None, blmeas_profiles=None, do_plot=True, figsize=None, both_zero_crossings=True, skip_indices=()):

    opt_func_screens = gauss_dict['opt_func_screens']
    opt_func_profiles = gauss_dict['opt_func_profiles']
    opt_func_sigmas = np.array(gauss_dict['opt_func_sigmas'])
    meas_screen = gauss_dict['meas_screen']
    gauss_sigma = gauss_dict['gauss_sigma']

    if plot_handles is None:
        fig, (sp_screen, sp_profile, sp_opt, sp_moments) = reconstruction_figure(figsize)
        plt.suptitle('Optimization')
    else:
        sp_screen, sp_profile, sp_opt, sp_moments = plot_handles

    rms_arr = np.zeros(len(opt_func_screens))
    centroid_arr = rms_arr.copy()

    for opt_ctr, (screen, profile, sigma) in enumerate(zip(opt_func_screens, opt_func_profiles, opt_func_sigmas)):
        if opt_ctr in skip_indices:
            continue
        if opt_ctr == gauss_dict['best_index']:
            lw = 3.
        else:
            lw = None
        screen.plot_standard(sp_screen, label='%i' % round(sigma*1e15), lw=lw)
        profile.plot_standard(sp_profile, label='%i' % round(profile.rms()*1e15), center='Mean', lw=lw)
        rms_arr[opt_ctr] = screen.rms()
        centroid_arr[opt_ctr] = screen.mean()

    meas_screen.plot_standard(sp_screen, color='black', ls='--', label=r'$\rho(x)$')

    #best_screen.plot_standard(sp_screen, color='red', lw=3, label='Final')
    #best_profile.plot_standard(sp_profile, color='red', lw=3, label='Final', center='Mean')

    if blmeas_profiles is not None:
        for blmeas_profile, ls, zero_crossing in zip(blmeas_profiles, ['--', 'dotted'], [1, 2]):
            blmeas_profile.plot_standard(sp_profile, ls=ls, color='black', label='%i' % round(blmeas_profile.rms()*1e15), center='Mean')
            if not both_zero_crossings:
                break

    color = sp_moments.plot(opt_func_sigmas*1e15, np.abs(centroid_arr)*1e3, marker='.', label='Reconstructed centroid')[0].get_color()
    sp_moments.axhline(np.abs(meas_screen.mean())*1e3, label='Measured centroid', color=color, ls='--')
    color = sp_moments.plot(opt_func_sigmas*1e15, rms_arr*1e3, marker='.', label='Reconstructed rms')[0].get_color()
    sp_moments.axhline(meas_screen.rms()*1e3, label='Measured rms', color=color, ls='--')

    sp_moments.legend()
    sp_screen.legend(title='Gaussian $\sigma$ (fs)', fontsize=config.fontsize)
    sp_profile.legend(title='rms (fs)', fontsize=config.fontsize)

    for sp_ in sp_opt, sp_moments:
        sp_.axvline(gauss_sigma*1e15, color='black')

    return gauss_dict

def plot_simple_daq(data_dict, dim):
    images = data_dict['pyscan_result']['image'].astype(np.float64).squeeze()
    x_axis = data_dict['pyscan_result']['x_axis_m'].astype(np.float64).squeeze()
    y_axis = data_dict['pyscan_result']['y_axis_m'].astype(np.float64).squeeze()
    xlabel = 'x (mm)'
    ylabel = 'y (mm)'

    if dim == 'Y':
        images = np.transpose(images, axis=(1,2))
        x_axis, y_axis = y_axis, x_axis
        xlabel, ylabel = ylabel, xlabel

    reverse_x = x_axis[1] < x_axis[0]
    reverse_y = y_axis[1] < y_axis[0]
    if reverse_x:
        x_axis = x_axis[::-1]
        images = images[:,::-1]
    if reverse_y:
        y_axis = y_axis[::-1]
        images = images[::-1,:]
    proj = images.sum(axis=-2)
    median_index = data_loader.get_median(proj, 'mean', 'index')
    image = image_analysis.Image(images[median_index], x_axis, y_axis)

    fig = ms.figure('Data acquisition')
    subplot = ms.subplot_factory(2, 2, False)
    sp_ctr = 1
    sp_img = subplot(sp_ctr, title='Median image', xlabel=xlabel, ylabel=ylabel)
    sp_ctr += 1

    sp_proj = subplot(sp_ctr, title='Projetions', xlabel=xlabel, ylabel='Intensity (arb. units)')
    sp_ctr += 1


    image.plot_img_and_proj(sp_img)

    for index in range(len(proj)):
        if index == median_index:
            color, lw = 'black', 3
        else:
            color, lw = None, None
        screen = beam_profile.ScreenDistribution(x_axis, proj[index])
        screen.plot_standard(sp_proj, color=color, lw=lw)
    return fig, (sp_img, sp_proj)

def plot_lasing(result_dict, plot_handles=None, figsize=None, n_shots=None):
    current_cutoff = result_dict['current_cutoff']
    mean_current = result_dict['mean_current']
    lasing_dict = result_dict['lasing_dict']

    mask = abs(mean_current) > current_cutoff

    if plot_handles is None:
        _, plot_handles = lasing_figure(figsize=figsize)
    sp_image_on, sp_image_on2, sp_image_off, sp_slice_mean, sp_slice_sigma, sp_current, sp_lasing_loss, sp_lasing_spread, sp_orbit = plot_handles

    for sp_image_tE, sp_image_xy, key in [
            (sp_image_on2, sp_image_on, 'images_on'),
            (sp_image_off, None, 'images_off'),
            ]:
        index_median = 0
        image_xy = result_dict[key]['raw_images'][index_median]
        image_tE = result_dict[key]['tE_images'][index_median]
        if sp_image_xy is not None:
            image_xy.plot_img_and_proj(sp_image_xy)
        image_tE.plot_img_and_proj(sp_image_tE)

    current_center = []
    for title, ls, mean_color in [('Lasing Off', None, 'black'), ('Lasing On', '--', 'red')]:
        all_slice_dict = result_dict['all_slice_dict'][title]
        mean_slice_dict = result_dict['mean_slice_dict'][title]
        for ctr in range(len(all_slice_dict['t'])):
            xx_plot = all_slice_dict['t'][ctr,mask]
            sp_slice_mean.plot(xx_plot*1e15, all_slice_dict['loss'][ctr,mask]/1e6, ls=ls)
            sp_slice_sigma.plot(xx_plot*1e15, np.sqrt(all_slice_dict['spread'][ctr,mask])/1e6, ls=ls)

        mean_mean = mean_slice_dict['loss']['mean'][mask]
        mean_std = mean_slice_dict['loss']['std'][mask]
        sigma_mean = mean_slice_dict['spread']['mean'][mask]
        sigma_std = mean_slice_dict['spread']['std'][mask]
        current_mean = mean_slice_dict['current']['mean']
        current_std = mean_slice_dict['current']['std']
        sp_slice_mean.errorbar(xx_plot*1e15, mean_mean/1e6, yerr=mean_std/1e6, color=mean_color, ls=ls, lw=3, label=title)
        _yy = np.sqrt(sigma_mean)
        _yy_err = sigma_std/(2*_yy)
        sp_slice_sigma.errorbar(xx_plot*1e15, _yy/1e6, yerr=_yy_err/1e6, color=mean_color, ls=ls, lw=3, label=title)
        sp_current.errorbar(mean_slice_dict['t']['mean']*1e15, current_mean/1e3, yerr=current_std/1e3, label=title, color=mean_color)
        current_center.append(np.sum(mean_slice_dict['t']['mean']*current_mean)/current_mean.sum())

    result_dict['images_off']['current_profile'].plot_standard(sp_current, center_float=np.mean(current_center), label='Reconstructed')
    sp_current.axhline(current_cutoff/1e3, color='black', ls='--')
    sp_current.legend()
    sp_slice_mean.legend()
    sp_slice_sigma.legend()

    for key, sp in [('all_Eloss', sp_lasing_loss), ('all_Espread', sp_lasing_spread)]:
        if n_shots is None:
            n_shots = len(lasing_dict[key])
        for n_shot, y_arr in enumerate(lasing_dict[key]):
            if n_shot > len(lasing_dict[key]) - n_shots - 1:
                sp.plot(lasing_dict['time']*1e15, y_arr/1e9, ls='--')

    for key, label, sp in [
            ('Eloss', '$\Delta E$', sp_lasing_loss),
            ('Espread', r'$\Delta \langle E^2 \rangle$', sp_lasing_spread)]:
        xx_plot = lasing_dict[key]['time']*1e15
        yy_plot = np.nanmean(lasing_dict['all_'+key], axis=0)/1e9
        yy_err = np.nanstd(lasing_dict['all_'+key], axis=0)/1e9
        sp.errorbar(xx_plot, yy_plot, yerr=yy_err, color='black')

    for label, key in [('Lasing On', 'images_on'), ('Lasing Off', 'images_off')]:
        delta_distance = result_dict[key]['delta_distances']
        mean_x = result_dict[key]['meas_screen_centroids']
        sp_orbit.scatter(mean_x*1e3, delta_distance*1e6, label=label)

    sp_orbit.legend()

def tdc_calib_figure(figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('TDC calibration')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,2)
    subplots = [subplot(sp_ctr) for sp_ctr in range(1, 1+3)]
    clear_tdc_calib_figure(*subplots)
    return fig, subplots

def clear_tdc_calib_figure(sp_profile, sp_screen, sp_wake):

    for sp, title, xlabel, ylabel in [
            (sp_profile, 'Current profile', 't (fs)', 'I (kA)'),
            (sp_screen, 'Screen distribution', 'x (mm)', config.rho_label),
            (sp_wake, 'Wake effect', 't (fs)', 'x (mm)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def plot_tdc_calibration(tdc_dict, plot_handles=None, figsize=None):
    if plot_handles is None:
        _, plot_handles = tdc_calib_figure(figsize)
    sp_profile, sp_screen, sp_wake = plot_handles

    blmeas_profile = tdc_dict['blmeas_profile']
    raw_screen = tdc_dict['meas_screen_raw']
    blmeas_profile.plot_standard(sp_profile, label='Measured')
    raw_screen.plot_standard(sp_screen, label='Measured')
    tdc_dict['forward_screen'].plot_standard(sp_screen, label='Reconstructed')
    tdc_dict['backward_dict']['profile'].plot_standard(sp_profile, label='Backward propagated')
    tdc_dict['backward_dict']['screen'].plot_standard(sp_screen, label='Backward propagation')
    wake_t = tdc_dict['backward_dict']['wake_time']
    wake_x = tdc_dict['backward_dict']['wake_x']
    sp_wake.plot(wake_x*1e3, wake_t*1e15)
    sp_profile.legend()
    sp_screen.legend()

