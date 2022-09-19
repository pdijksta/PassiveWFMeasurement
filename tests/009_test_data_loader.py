import numpy as np; np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.data_loader as data_loader
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

data_dict = h5_storage.loadH5Recursive('./data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5')
pyscan_result = data_dict['pyscan_result']

x_axis = pyscan_result['x_axis_m'].astype(np.float64)
images = pyscan_result['image'].astype(np.float64)

projx = images.sum(axis=-2)
median_proj, _ = data_loader.get_median(projx, 'mean', 'proj')


fig = ms.figure('Test data loader')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_proj = subplot(sp_ctr)
sp_ctr += 1

for proj in projx:
    sp_proj.plot(proj)

sp_proj.plot(median_proj, color='black', lw=3.)

ms.show()

