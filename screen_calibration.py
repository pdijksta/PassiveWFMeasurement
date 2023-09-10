import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize

def calibrate_screen0(trans_dist, profile, tracker, smoothen, half_factor=4, sp=None):
    trans_dist = copy.deepcopy(trans_dist)
    trans_dist.smoothen(smoothen)

    method0 = tracker.find_beam_position_options['method']
    try:
        tracker.find_beam_position_options['method'] = 'beamsize'
        find_beam_position_dict = tracker.find_beam_position(tracker.beam_position, trans_dist, profile)
    #except:
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.suptitle('Trans dist')
    #    plt.plot(trans_dist._xx, trans_dist._yy)
    #    plt.figure()
    #    plt.suptitle('Current profile')
    #    plt.plot(profile._xx, profile._yy)
    #    import pdb; pdb.set_trace()
    finally:
        tracker.find_beam_position_options['method'] = method0

    def do_interp(trans_dist, half_int=None):
        indices, _ = find_peaks(trans_dist.intensity, height=0.2*trans_dist.intensity.max())
        if forward_screen.mean() > 0:
            index = indices[0]
            interp_intensity = trans_dist.intensity[:index].copy()
            interp_x = trans_dist.x[:index]
        else:
            index = indices[-1]
            interp_intensity = trans_dist.intensity[index:][::-1].copy()
            interp_x = trans_dist.x[index:][::-1]
        if half_int is None:
            half_int = trans_dist.intensity[index]/half_factor
        interp_intensity[interp_intensity < interp_intensity.max()*0.2] = 0
        half_peak_x = np.interp(half_int, interp_intensity, interp_x)
        return half_peak_x, half_int, index

    forward_screen = find_beam_position_dict['sim_screen']
    forward_screen.normalize()
    half_peak_x_sim, half_int, index_sim = do_interp(forward_screen)

    half_peak_x, _, index_meas = do_interp(trans_dist, half_int)
    delta_x = -(half_peak_x - half_peak_x_sim)

    # For debug only
    if sp is not None:
        sp.plot(forward_screen.x*1e3, forward_screen.intensity)
        sp.plot(trans_dist.x*1e3, trans_dist.intensity)
        sp.plot((trans_dist.x+delta_x)*1e3, trans_dist.intensity)
        sp.axvline(trans_dist.x[index_meas]*1e3, color='black')
        sp.axvline(half_peak_x*1e3, color='gray')
        sp.axvline(forward_screen.x[index_sim]*1e3, color='black')
        sp.axvline(half_peak_x_sim*1e3, color='gray')
    return delta_x

def new_calibrate_screen0(trans_dist, profile, tracker, smoothen, sp=None):
    trans_dist = copy.deepcopy(trans_dist)
    trans_dist.smoothen(smoothen)

    method0 = tracker.find_beam_position_options['method']
    quad0 = tracker.forward_options['quad_wake']
    try:
        tracker.find_beam_position_options['method'] = 'beamsize'
        tracker.forward_options['quad_wake'] = True
        find_beam_position_dict = tracker.find_beam_position(tracker.beam_position, trans_dist, profile)
    #except:
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.suptitle('Trans dist')
    #    plt.plot(trans_dist._xx, trans_dist._yy)
    #    plt.figure()
    #    plt.suptitle('Current profile')
    #    plt.plot(profile._xx, profile._yy)
    #    import pdb; pdb.set_trace()
    finally:
        tracker.find_beam_position_options['method'] = method0
        tracker.forward_options['quad_wake'] = quad0
    forward_screen = find_beam_position_dict['sim_screen']
    forward_screen.normalize()

    def func(shift):
        #print('%.0f' % ((shift-x0)*1e6))
        xx_new = trans_dist._xx + shift
        yy_new = np.interp(forward_screen._xx, xx_new, trans_dist._yy, left=0, right=0)
        return np.sum((yy_new - forward_screen._yy)**2)*1e2
    x0 = forward_screen.mean() - trans_dist.mean()
    outp = minimize(func, x0, options={'eps': 2e-5, 'maxiter': 20})
    delta_x = outp.x
    #print(x0, (delta_x-x0)*1e6)
    if sp is not None:
        sp.plot(forward_screen.x*1e3, forward_screen.intensity)
        sp.plot((trans_dist.x+x0)*1e3, trans_dist.intensity)
        sp.plot((trans_dist.x+delta_x)*1e3, trans_dist.intensity)

    return delta_x

