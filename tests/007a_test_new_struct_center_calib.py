import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

gap_recon_dict = h5_storage.loadH5Recursive('./gap_recon_dict.h5')


ms.figure('Gap reconstruction improvements')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_raw = subplot(sp_ctr, title='Raw data', xlabel='$\Delta$ d ($\mu$ m)', ylabel='rms duration (fs)')
sp_ctr += 1

distance_rms_arr = gap_recon_dict['distance_rms_arr']

for n in range(len(distance_rms_arr)):
    distance_arr = distance_rms_arr[n,:,0]
    rms_arr = distance_rms_arr[n,:,1]

    distance_plot = distance_arr - distance_arr.min()

    sort = np.argsort(distance_plot)

    label = '%i $\mu$m' % (gap_recon_dict['final_beam_positions'][n]*1e6)
    sp_raw.plot(distance_plot[sort]*1e6, rms_arr[sort]*1e15, label=label, marker='.')




ms.show()

