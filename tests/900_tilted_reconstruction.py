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

print(xProfile.xx.min(), xProfile.yy.min())
xProfile.aggressive_cutoff(x_cutoff)
xProfile.crop()
print(xProfile.xx.min(), xProfile.yy.min())

img_cut1 = image.cut(xProfile.xx[0], xProfile.xx[-1])
img_cut = img_cut1.cutY(yProfile.xx[0], yProfile.xx[-1])


fig = ms.figure('Test tilted reconstruction')
fig.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2, 2, grid=False)
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

sp_centroid = subplot(sp_ctr, title='X to Y', xlabel='x (mm)', ylabel='y(mm)')
sp_ctr += 1

sp_centroid.plot(xx*1e3, centroids*1e3)



ms.show()

