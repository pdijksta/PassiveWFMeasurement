import numpy as np

def calc_resolution(beamprofile, gap, beam_offset, struct_length, tracker, n_streaker, bins=(150, 100), camera_res=20e-6):
    gaps = [10e-3, 10e-3]
    gaps[n_streaker] = gap
    beam_offsets = [0, 0]
    beam_offsets[n_streaker] = beam_offset
    #wf_calc = beamprofile.calc_wake(semigap*2, beam_offset, struct_length)
    forward_dict = tracker.matrix_forward(beamprofile, gaps, beam_offsets)
    beam = forward_dict['beam_at_screen']
    r12 = tracker.calcR12()[n_streaker]
    _, wf_x = beamprofile.get_x_t(gap, beam_offset, struct_length, r12)
    wf_t = beamprofile.time
    dxdt = np.diff(wf_x)/np.diff(wf_t)
    dx_dt_t = wf_t[:-1]
    beam_t = beam[-2]
    beam_x = beam[0] + np.random.randn(beam_t.size)*camera_res
    hist, xedges, yedges = np.histogram2d(beam_t-beam_t.mean(), beam_x, bins=bins)
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
            'wf_x': wf_x,
            'beamsize': beamsize,
            'streaking_strength': streaking_strength,
            'beam': beam,
            'beamprofile': beamprofile
            }
    return output

def plot_resolution(res_dict, sp_current, sp_res, max_res=20e-15):
    bp = res_dict['beamprofile']
    bp.plot_standard(sp_current, color='black')
    sp_res.plot(res_dict['time']*1e15, res_dict['resolution']*1e15)
    sp_res.set_ylim(0, max_res*1e15)


