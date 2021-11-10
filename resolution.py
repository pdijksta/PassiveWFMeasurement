import numpy as np
from . import gen_beam

def calc_resolution(beamprofile, gap, beam_offset, tracker, bins=(150, 100), camera_res=20e-6):
    #wf_calc = beamprofile.calc_wake(semigap*2, beam_offset, struct_length)
    beam_spec = tracker.beam_spec.copy()
    beam_spec.update(tracker.beam_optics)
    beam = gen_beam.beam_from_spec(['x', 't'], beam_spec, tracker.n_particles, beamprofile, tracker.total_charge, tracker.energy_eV)
    forward_dict = tracker.forward_propagate_forced(gap, beam_offset, beam, output_details=True)
    beam = forward_dict['beam_at_screen']
    r12 = tracker.r12
    wake_dict = beamprofile.calc_wake(tracker.structure, gap, beam_offset, 'Dipole')
    wf_x = wake_dict['wake_potential'] / tracker.energy_eV * r12
    wf_t = beamprofile.time
    dxdt = np.diff(wf_x)/np.diff(wf_t)
    dx_dt_t = wf_t[:-1]
    beam_t = beam['t']
    beam_x = beam['x'] + np.random.randn(beam_t.size)*camera_res
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


