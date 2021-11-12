import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

from PassiveWFMeasurement import h5_storage
from PassiveWFMeasurement import image_analysis
from PassiveWFMeasurement import beam_profile
from PassiveWFMeasurement import myplotstyle as ms
from PassiveWFMeasurement import gaussfit

ms.closeall()

example_file = '/sf/data/measurements/2021/11/12/20211112_200247_SATBD02-DSCR050_camera_snapshot.h5'

example_data = h5_storage.loadH5Recursive(example_file)['camera1']

raw_image = example_data['image'].astype(np.float64).squeeze()
x_axis = example_data['x_axis'].astype(np.float64).squeeze()*1e-6
y_axis = example_data['y_axis'].astype(np.float64).squeeze()*1e-6

raw_image -= np.median(raw_image)
raw_image[raw_image < 0] = 0

reverse_x = x_axis[1] < x_axis[0]
reverse_y = y_axis[1] < y_axis[0]
if reverse_x:
    x_axis = x_axis[::-1]
    raw_image = raw_image[:,::-1]
if reverse_y:
    y_axis = y_axis[::-1]
    raw_image = raw_image[::-1,:]

energy_eV = 3.17e9
charge = 200e-12
y_cutoff = 0.15
x_cutoff = 0.15


image = image_analysis.Image(raw_image, x_axis, y_axis)

projY = raw_image.sum(axis=-1)
projY -= projY.min()
yProfile = beam_profile.AnyProfile(y_axis, projY)

projX = raw_image.sum(axis=0)
projX -= projX.min()
xProfile = beam_profile.AnyProfile(x_axis, projX)

yProfile.aggressive_cutoff(y_cutoff)
yProfile.crop()

xProfile.aggressive_cutoff(x_cutoff)
xProfile.crop()

img_cut1 = image.cut(xProfile.xx[0], xProfile.xx[-1])
img_cut = img_cut1.cutY(yProfile.xx[0], yProfile.xx[-1])


fig = ms.figure('Test tilted reconstruction')
fig.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2, 3, grid=False)
sp_ctr = 1

sp_img = subplot(sp_ctr, title='Image', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

img_cut.plot_img_and_proj(sp_img)

sp_x = subplot(sp_ctr, title='Projection in X', xlabel='x (mm)', ylabel='Intensity (arb. units)')
sp_ctr += 1

sp_x.plot(xProfile.xx*1e3, xProfile.yy*1e3)

sp_y = subplot(sp_ctr, title='Projection in Y', xlabel='y (mm)', ylabel='Intensity (arb. units)')
sp_ctr += 1

sp_y.plot(yProfile.xx*1e3, yProfile.yy*1e3)

mask_y = np.logical_and(img_cut.y_axis >= yProfile.xx.min(), img_cut.y_axis <= yProfile.xx.max())

xx = img_cut.x_axis
centroids = []

for nx in range(len(img_cut.x_axis)):
    arr = img_cut.image[mask_y, nx]
    gfY = gaussfit.GaussFit(img_cut.y_axis, arr)
    centroids.append(gfY.mean)

centroids = np.array(centroids)

sp_centroid = subplot(sp_ctr, title='X to Y', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

sp_centroid.plot(xx*1e3, centroids*1e3, label='Initial')

gf_projY = gaussfit.GaussFit(y_axis, projY)
mean_y = gf_projY.mean
where_mean = int(np.argmin((mean_y - img_cut.y_axis)**2).squeeze())

centroids2 = centroids.copy()
for y_index in range(where_mean, len(centroids)-1):
    old_val = centroids2[y_index]
    new_val = centroids2[y_index+1]
    max_val = max(old_val, new_val)
    centroids2[y_index+1] = max_val
for y_index in range(where_mean, 0, -1):
    old_val = centroids2[y_index]
    new_val = centroids2[y_index-1]
    centroids2[y_index-1] = min(old_val, new_val)

sp_centroid.plot(xx*1e3, centroids2*1e3, label='Corrected')


sp_centroid.legend()

if np.all(np.diff(centroids2) <= 0):
    centroids2 = centroids2[::-1]
    xx = xx[::-1]

rel_time = (centroids2 - centroids2[0]) / (centroids2[-1] - centroids2[0])

sp_rel_time = subplot(sp_ctr, title='X to relative time', xlabel='x (mm)', ylabel='Relative time')
sp_ctr += 1

sp_rel_time.plot(xx*1e3, rel_time)

img_converted = img_cut.x_to_t(xx, rel_time)

sp_converted = subplot(sp_ctr, title='Converted image', xlabel='Relative time', ylabel='y (mm)')
sp_ctr += 1

for y_index in range(len(img_converted.y_axis)):
    old_sum = img_cut.image[y_index,:].sum()
    img_converted.image[y_index,:] *= old_sum / img_converted.image[y_index,:].sum()

img_converted.image[:,-2:] = 0
img_converted.plot_img_and_proj(sp_converted)


# DAQ 12.11.2021 approx 21:40
# 1. lasing Off
# 2. tail lasing 34 uJ
# 3. head lasing 82.5 uJ
# 4. both lasing 117 uJ (gas detector 87 uJ)



ms.show()

