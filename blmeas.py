import numpy as np

from . import beam_profile
from . import h5_storage

def get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases, cutoff=5e-2, size=int(1e3)):
    time_arr = y_axis / calibration
    current0 = images.sum(axis=-1, dtype=np.float64)
    current = current0.reshape([current0.size//current0.shape[-1], current0.shape[-1]])
    reverse = time_arr[1] < time_arr[0]
    if reverse:
        time_arr = time_arr[::-1]
    current_profiles0 = []

    # Find out where is the head and the tail.
    # In our conventios, the head is at negative time
    if len(phases) > 1:
        dy_dphase = np.polyfit(phases, centroids, 1)[0]
        sign_dy_dt = np.sign(dy_dphase)
    else:
        print('Warning! Time orientation of bunch length measurement cannot be determined!')
        sign_dy_dt = 1

    reverse_current = sign_dy_dt == -1

    if reverse != reverse_current:
        current = current[:,::-1]

    for curr in current:
        bp = beam_profile.BeamProfile(time_arr, curr, 1, 1)
        current_profiles0.append(bp)

    for bp in current_profiles0:
        bp._yy = bp._yy - bp._yy.min()
        bp.reshape(size)
        bp.aggressive_cutoff(cutoff)
        bp.crop()
        bp.reshape(size)
        bp.center('Mean')

    squares_mat = np.zeros([len(current_profiles0)]*2, float)

    for n_row in range(len(squares_mat)):
        for n_col in range(n_row):
            bp1 = current_profiles0[n_row]
            bp2 = current_profiles0[n_col]
            minus = bp1.charge_dist - bp2.charge_dist
            squares_mat[n_row,n_col] = squares_mat[n_col,n_row] = np.sum(minus**2)

    squares = squares_mat.sum(axis=1)
    n_best = np.argmin(squares)

    #import myplotstyle as ms
    #fignum = ms.plt.gcf().number
    #ms.figure('Debug')
    #sp = ms.plt.subplot(1,1,1)
    #for curr in current_profiles0:
    #    curr.plot_standard(sp)
    #current_profiles0[n_best].plot_standard(sp, color='black', lw=3)
    #ms.show()
    #import pdb; pdb.set_trace()
    #ms.plt.figure(fignum)

    return time_arr, current[n_best], current

def load_avg_blmeas_new(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)

    outp = {}
    calibration = blmeas_dict['Processed data']['Calibration'] * 1e-6/1e-15

    zc_strings = ['']
    if 'Beam images 2' in blmeas_dict['Processed data']:
        zc_strings.append(' 2')
    for n_zero_crossing, zc_string in enumerate(zc_strings, 1):
        images = blmeas_dict['Processed data']['Beam images'+zc_string]
        x_axis = blmeas_dict['Processed data']['x axis'+zc_string]*1e-6
        y_axis = blmeas_dict['Processed data']['y axis'+zc_string]*1e-6
        phases = blmeas_dict['Processed data']['Phase'+zc_string]
        centroids = blmeas_dict['Processed data']['Beam centroids'+zc_string]

        time_arr, curr_best, all_current = get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases)

        outp[n_zero_crossing] = {
                'time': time_arr,
                'current': curr_best,
                'current_reduced': curr_best - curr_best.min(),
                'all_current': all_current,
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis]
                }
    return outp

def load_avg_blmeas_old(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)
    outp = {}
    calibration = blmeas_dict['Meta_data']['Calibration factor'] * 1e-6/1e-15
    zc_strings = ['']
    if 'Beam images 2' in blmeas_dict['Raw_data']:
        zc_strings.append(' 2')

    for n_zero_crossing, zc_string in enumerate(zc_strings, 1):
        images = blmeas_dict['Raw_data']['Beam images'+zc_string]
        x_axis = blmeas_dict['Raw_data']['xAxis'+zc_string.replace(' ','')]*1e-6
        y_axis = blmeas_dict['Raw_data']['yAxis'+zc_string.replace(' ','')]*1e-6
        centroids = blmeas_dict['Raw_data']['Beam centroids'+zc_string].mean(axis=1)
        phases = np.arange(len(centroids))

        time_arr, curr_best, all_current = get_average_blmeas_profile(images, x_axis, y_axis, calibration, centroids, phases)

        outp[n_zero_crossing] = {
                'time': time_arr,
                'current': curr_best,
                'current_reduced': curr_best - curr_best.min(),
                'all_current': all_current,
                'all_current_reduced': all_current - all_current.min(axis=1)[:,np.newaxis]
                }
    return outp

def load_avg_blmeas(file_or_dict):
    if type(file_or_dict) is dict:
        blmeas_dict = file_or_dict
    else:
        blmeas_dict = h5_storage.loadH5Recursive(file_or_dict)

    if 'Raw_data' in blmeas_dict:
        return load_avg_blmeas_old(blmeas_dict)
    else:
        return load_avg_blmeas_new(blmeas_dict)

