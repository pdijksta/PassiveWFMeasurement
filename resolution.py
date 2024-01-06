import numpy as np

def beam_to_res(forward_dict, r12, energy_eV, camera_res, bins, dim, use_other_dim, long_wake):
    beam = forward_dict['beam_at_screen']
    beamprofile = beam.beamProfile
    wake_dict = forward_dict['wake_dict_dipole']
    wake_dict_quad = forward_dict['wake_dict_quadrupole']
    wf_x = wake_dict['wake_potential'] / energy_eV * r12
    wf_t = beamprofile.time
    dxdt = np.diff(wf_x)/np.diff(wf_t)
    dx_dt_t = wf_t[:-1]
    beam_t = beam['t']
    beam_x = beam[dim] + np.random.randn(beam_t.size)*camera_res
    hist, xedges, yedges = np.histogram2d(beam_t, beam_x, bins=bins)
    t_axis = (xedges[1:] + xedges[:-1])/2.
    x_axis = (yedges[1:] + yedges[:-1])/2.
    x_axis2 = np.ones_like(hist)*x_axis
    current_t = hist.sum(axis=1)
    mean_x = np.sum(hist*x_axis, axis=1) / current_t
    mean_x2 = np.ones_like(hist)*mean_x[:,np.newaxis]
    beamsize_sq = np.sum(hist*(x_axis2 - mean_x2)**2, axis=1) / current_t
    beamsize = np.sqrt(beamsize_sq)
    streaking_strength = np.abs(np.interp(t_axis, dx_dt_t, dxdt))
    resolution = beamsize / streaking_strength
    output = {
            'time': t_axis,
            'resolution': resolution,
            'r12': r12,
            'wf_%s' % dim: wf_x,
            'center': mean_x,
            'beamsize': beamsize,
            'streaking_strength': streaking_strength,
            'beam': beam,
            'beamprofile': beamprofile,
            'wake_dict_dipole': wake_dict,
            'wake_dict_quad': wake_dict_quad,
            'forward_dict': forward_dict,
            }

    if use_other_dim:
        other_dim = 'x' if dim == 'y' else 'y'
        beam_y = beam[other_dim] + np.random.randn(beam_t.size)*camera_res
        hist, xedges, yedges = np.histogram2d(beam_t, beam_y, bins=bins)
        current_t = hist.sum(axis=1)
        y_axis = (yedges[1:] + yedges[:-1])/2.
        y_axis2 = np.ones_like(hist)*y_axis
        mean_y = np.sum(hist*y_axis, axis=1) / current_t
        mean_y2 = np.ones_like(hist)*mean_y[:,np.newaxis]
        beamsize_ysq = np.sum(hist*(y_axis2 - mean_y2)**2, axis=1) / current_t
        beamsize_y = np.sqrt(beamsize_ysq)
        output['other_beamsize'] = beamsize_y
        output['other_center'] = mean_y

        hist, xedges, yedges = np.histogram2d(beam_x, beam_y, bins=bins)
        current_t = hist.sum(axis=1)
        y_axis = (yedges[1:] + yedges[:-1])/2.
        y_axis2 = np.ones_like(hist)*y_axis
        mean_y = np.sum(hist*y_axis, axis=1) / current_t
        mean_y2 = np.ones_like(hist)*mean_y[:,np.newaxis]
        beamsize_ysq = np.sum(hist*(y_axis2 - mean_y2)**2, axis=1) / current_t
        beamsize_y = np.sqrt(beamsize_ysq)
        x_axis2 = (xedges[1:] + xedges[:-1])/2
        if np.any(np.diff(wf_x) < 0):
            output['other_time2'] = np.interp(x_axis2, wf_x[::-1], wf_t[::-1])[::-1]
            output['other_beamsize2'] = beamsize_y[::-1]
        else:
            output['other_time2'] = np.interp(x_axis2, wf_x, wf_t)
            output['other_beamsize2'] = beamsize_y

    if long_wake:
        hist, xedges, yedges = np.histogram2d(beam_t-beam_t.mean(), beam['delta'], bins=bins)
        y_axis = (yedges[1:] + yedges[:-1])/2.
        y_axis2 = np.ones_like(hist)*y_axis
        current_t = hist.sum(axis=1)
        mean_y = np.sum(hist*y_axis, axis=1) / current_t
        mean_y2 = np.ones_like(hist)*mean_y[:,np.newaxis]
        beamsize_ysq = np.sum(hist*(y_axis2 - mean_y2)**2, axis=1) / current_t
        beamsize_y = np.sqrt(beamsize_ysq)
        output['delta_beamsize'] = beamsize_y
        output['delta_center'] = mean_y
    return output


def calc_resolution(beamprofile, gap, beam_offset, tracker, bins=(150, 100), camera_res=20e-6, dim='x', use_other_dim=False, long_wake=False, add_chirp=0):
    #wf_calc = beamprofile.calc_wake(semigap*2, beam_offset, struct_length)
    beam_spec = tracker.beam_spec.copy()
    beam_spec.update(tracker.beam_optics)
    if add_chirp:
        beam_spec['energy_chirp'] = add_chirp
    tracker.beam_spec = beam_spec
    beam = tracker.gen_beam(beamprofile, other_dim=use_other_dim, delta=(long_wake or add_chirp))
    forward_dict = tracker.forward_propagate_forced(gap, beam_offset, beam, output_details=True)
    return beam_to_res(forward_dict, tracker.r12, tracker.energy_eV, camera_res, bins, dim, use_other_dim, long_wake)

def plot_resolution(res_dict, sp_current, sp_res, max_res=20e-15):
    bp = res_dict['beamprofile']
    bp.plot_standard(sp_current, color='black')
    sp_res.plot(res_dict['time']*1e15, res_dict['resolution']*1e15)
    sp_res.set_ylim(0, max_res*1e15)


def simulate_resolution_complete(tracker, beam, bins=(150,150), camera_res=0):

    old_fo = tracker.forward_options.copy()
    old_matrix = tracker.matrix.copy()
    dim = tracker.structure.dim.lower()
    other_dim = 'x' if dim == 'y' else 'y'

    try:
        tracker.forward_options['dipole_wake'] = True
        tracker.forward_options['quad_wake'] = True
        tracker.forward_options['long_wake'] = True
        tracker.forward_options['long_wake_correction'] = True
        fd_complete = tracker.forward_propagate(beam, output_details=False)

        tracker.matrix[:-1,5] = 0
        tracker.forward_options['dipole_wake'] = False
        tracker.forward_options['long_wake'] = False
        tracker.forward_options['long_wake_correction'] = False
        fd_no_disp_no_dipole = tracker.forward_propagate(beam, output_details=False)

        beam2 = beam.child()
        beam2['p'] = tracker.energy_eV
        tracker.forward_options['quad_wake'] = False
        tracker.forward_options['long_wake_correction'] = True
        fd_same_p0 = tracker.forward_propagate(beam2, output_details=False)
    finally:
        tracker.forward_options = old_fo
        tracker.matrix = old_matrix

    ## Time resolution
    beam_t = fd_complete['beam_at_screen']['t']
    beam_x = fd_complete['beam_at_screen'][dim]
    if camera_res:
        beam_x = beam_x + np.random.randn(beam_x.size)*camera_res
    hist, xedges, yedges = np.histogram2d(beam_t, beam_x, bins=bins)
    histsum = np.sum(hist, axis=1)
    t_axis = t_axis0 = (xedges[1:] + xedges[:-1])/2.
    x_axis = (yedges[1:] + yedges[:-1])/2.
    mean_x = np.sum(hist*x_axis, axis=1) / histsum
    dxdt = np.diff(mean_x) / np.diff(t_axis)
    dxdt = np.append(dxdt, dxdt[-1])


    beam_t = fd_no_disp_no_dipole['beam_at_screen']['t']
    beam_x = fd_no_disp_no_dipole['beam_at_screen'][dim]
    if camera_res:
        beam_x = beam_x + np.random.randn(beam_x.size)*camera_res
    hist, xedges, yedges = np.histogram2d(beam_t, beam_x, bins=bins)
    histsum = np.sum(hist, axis=1)
    t_axis = t_axis1 = (xedges[1:] + xedges[:-1])/2.
    assert np.all(t_axis0 == t_axis1)
    x_axis = (yedges[1:] + yedges[:-1])/2.

    mean_x = np.sum(hist*x_axis, axis=1) / histsum
    beamsize = np.sqrt(np.sum(hist*x_axis**2, axis=1) / histsum - mean_x**2)
    time_res = beamsize/dxdt

    ## Energy resolution
    #nat beam size
    beam_x = fd_no_disp_no_dipole['beam_at_screen'][dim]
    beam_y = fd_no_disp_no_dipole['beam_at_screen'][other_dim]
    if camera_res:
        beam_x = beam_x + np.random.randn(beam_x.size)*camera_res
        beam_y = beam_y + np.random.randn(beam_x.size)*camera_res
    hist, xedges, yedges = np.histogram2d(beam_x, beam_y, bins=bins)
    histsum = np.sum(hist, axis=1)
    x_axis = (xedges[1:] + xedges[:-1])/2.
    y_axis = (yedges[1:] + yedges[:-1])/2.

    mean_y = np.sum(hist*y_axis, axis=1) / histsum
    beamsize = np.sqrt(np.sum(hist*y_axis**2, axis=1) / histsum - mean_y**2)
    eres1 = beamsize/tracker.disp*tracker.energy_eV

    # energy chirp
    beam_t = fd_complete['beam_at_screen']['t']
    beam_p = fd_complete['beam_at_screen']['p']
    hist, xedges, yedges = np.histogram2d(beam_t, beam_p, bins=bins)
    histsum = np.sum(hist, axis=1)
    t_axis = t_axis2 = (xedges[1:] + xedges[:-1])/2.
    assert np.all(t_axis2 == t_axis0)
    p_axis = (yedges[1:] + yedges[:-1])/2.
    mean_p = np.sum(hist*p_axis, axis=1) / histsum
    dp_dt = np.diff(mean_p)/np.diff(t_axis)
    dp_dt = np.append(dp_dt, dp_dt[-1])
    eres2 = time_res*dp_dt

    # Induced espread
    beam_t = fd_same_p0['beam_at_screen']['t']
    beam_p = fd_same_p0['beam_at_screen']['p']
    hist, xedges, yedges = np.histogram2d(beam_t, beam_p, bins=bins)
    histsum = np.sum(hist, axis=1)
    t_axis = t_axis3 = (xedges[1:] + xedges[:-1])/2.
    assert np.all(t_axis3 == t_axis0)
    p_axis = (yedges[1:] + yedges[:-1])/2.
    mean_p = np.sum(hist*p_axis, axis=1) / histsum
    eres3 = np.sqrt(np.sum(hist*p_axis**2, axis=1) / histsum - mean_p**2)

    eres = np.sqrt(eres1**2+eres2**2+eres3**2)
    output = {
            'fd_no_disp_no_dipole': fd_no_disp_no_dipole,
            'fd_complete': fd_complete,
            'time': t_axis0,
            'time_res': time_res,
            'eres1': eres1,
            'eres2': eres2,
            'eres3': eres3,
            'eres': eres
            }
    return output

