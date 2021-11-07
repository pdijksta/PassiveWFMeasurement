import numpy as np
import matplotlib.pyplot as plt
from . import myplotstyle as ms
from . import config

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
        if distance > max_distance:
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

def streaker_calibration_figure(figsize=None):
    if figsize is None:
        figsize = [6.4, 4.8]
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker center calibration')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 3)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+6)))
    clear_streaker_calibration(*plot_handles)
    return fig, plot_handles

def clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current):
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
        fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = streaker_calibration_figure(figsize)
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

