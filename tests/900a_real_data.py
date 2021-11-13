import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.image_analysis as image_analysis
import PassiveWFMeasurement.beam_profile as beam_profile
from PassiveWFMeasurement import myplotstyle as ms

ms.closeall()

def raw_data_to_img(raw_image, x_axis, y_axis):
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

    image = image_analysis.Image(raw_image, x_axis, y_axis)
    return image

def cutoff_image(raw_image, x_cutoff, y_cutoff):
    projY = raw_image.image.sum(axis=-1)
    projY -= projY.min()
    yProfile = beam_profile.AnyProfile(raw_image.y_axis, projY)

    projX = raw_image.image.sum(axis=0)
    projX -= projX.min()
    xProfile = beam_profile.AnyProfile(raw_image.x_axis, projX)

    yProfile.aggressive_cutoff(y_cutoff)
    yProfile.crop()

    xProfile.aggressive_cutoff(x_cutoff)
    xProfile.crop()

    img_cut1 = raw_image.cut(xProfile.xx[0], xProfile.xx[-1])
    img_cut = img_cut1.cutY(yProfile.xx[0], yProfile.xx[-1])

    x_mean = xProfile.mean()
    xProfile._xx = xProfile._xx - x_mean
    img_cut.x_axis = img_cut.x_axis - x_mean

    y_mean = yProfile.mean()
    yProfile._xx = yProfile._xx - y_mean
    img_cut.y_axis = img_cut.y_axis - y_mean

    return img_cut

def cut_imageX(raw_image, x_len, y_cutoff):
    projY = raw_image.image.sum(axis=-1)
    projY -= projY.min()
    yProfile = beam_profile.AnyProfile(raw_image.y_axis, projY)

    projX = raw_image.image.sum(axis=0)
    projX -= projX.min()
    xProfile = beam_profile.AnyProfile(raw_image.x_axis, projX)
    xProfile.aggressive_cutoff(0.15)
    x_mean = xProfile.mean()

    yProfile.aggressive_cutoff(y_cutoff)
    yProfile.crop()

    img_cut1 = raw_image.cutY(yProfile.xx[0], yProfile.xx[-1])
    img_cut1.x_axis = img_cut1.x_axis - x_mean
    mask_x = np.ones_like(img_cut1.x_axis, dtype=bool)
    zero_left = (len(mask_x) - x_len)//2
    zero_right = len(mask_x) - x_len - zero_left
    mask_x[:zero_left] = 0
    mask_x[-zero_right:] = 0
    assert mask_x.sum() == x_len

    img_cut1.x_axis = img_cut1.x_axis[mask_x]
    img_cut1.image = img_cut1.image[:,mask_x]
    return img_cut1

def center_image(image):
    projY = image.image.sum(axis=-1)
    projY -= projY.min()
    yProfile = beam_profile.AnyProfile(image.y_axis, projY)
    yProfile.aggressive_cutoff(0.15)

    projX = image.image.sum(axis=0)
    projX -= projX.min()
    xProfile = beam_profile.AnyProfile(image.x_axis, projX)
    xProfile.aggressive_cutoff(0.15)

    return image.child(image.image, image.x_axis-xProfile.mean(), image.y_axis-yProfile.mean())

def y_to_t(img):
    _y = img.y_axis
    rel_time = (_y - _y[0]) / (_y[-1] - _y[0])
    return img.child(img.image, img.x_axis, rel_time)

def get_img(dict_, index):
    image_data = dict_['image'].astype(np.float64).squeeze()[index]
    x_axis = dict_['x_axis_m'].astype(np.float64).squeeze()
    y_axis = dict_['y_axis_m'].astype(np.float64).squeeze()
    return raw_data_to_img(image_data, x_axis, y_axis)



lasing_off_file = '/sf/data/measurements/2021/11/12/2021_11_12-23_22_11_Lasing_False_SATBD02-DSCR050.h5'
lasing_on_file = '/sf/data/measurements/2021/11/12/2021_11_12-23_15_06_Lasing_True_SATBD02-DSCR050.h5'

lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)['pyscan_result']
lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)['pyscan_result']


energy_eV = 3.17e9
charge = 200e-12
y_cutoff = 0.15
x_cutoff = 0.15

raw_img = get_img(lasing_off_dict, 0)
img_cut0 = cutoff_image(raw_img, x_cutoff, y_cutoff)
#img_cut1 = img_cut0.cut(-0.1e-3, 1000)
img_cut1 = img_cut0
img_cut = y_to_t(img_cut1)

fig = ms.figure('Test tilted reconstruction')
fig.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2, 3, grid=False)
sp_ctr = 1

sp_img = subplot(sp_ctr, title='Calib image (x,t)', xlabel='x (mm)', ylabel='t (arb. units)')
sp_ctr += 1

img_cut.plot_img_and_proj(sp_img, y_factor=1, plot_gauss=False, log=False)

def x_to_t(img, calib_image):
    new_image = np.zeros_like(img.image)
    int_image = img.image.astype(int)*10
    x_hist_bins = np.linspace(0, 1, len(calib_image.x_axis)+1)
    new_x = np.linspace(0, 1, new_image.shape[1])
    for x_index, x in enumerate(img.x_axis):
        #x_index2 = np.argmin(x-calib_image.x_axis
        y_dist = calib_image.image[:,x_index].copy()
        y_dist[y_dist < 0.5 * y_dist.max()] = 0
        cdf = np.cumsum(y_dist)
        cdf /= cdf.max()
        for y_index in range(len(img.y_axis)):
            randoms = np.random.rand(int_image[y_index, x_index])
            new_t = np.interp(randoms, cdf, calib_image.y_axis)
            add, _ = np.histogram(new_t, bins=x_hist_bins)
            new_image[y_index,:] += add
    new_image = new_image / new_image.sum() * calib_image.image.sum()
    new_x = np.linspace(0, 1, new_image.shape[1])

    img_converted = img.child(new_image, new_x, img.y_axis)
    return img_converted

    #if x_index == len(img_cut.x_axis)//2:
    #    sp = subplot(sp_ctr)
    #    sp_ctr += 1
    #    sp.plot(cdf)
    #    sp = subplot(sp_ctr)
    #    sp_ctr += 1
    #    sp.plot(new_x, add)
    #    sp = subplot(sp_ctr)
    #    sp_ctr += 1
    #    sp.plot(img_cut.y_axis, y_dist)


sp_img_converted = subplot(sp_ctr, title='Calib image (t,t)', xlabel='t (arb. units)', ylabel='y (mm)')
sp_ctr += 1

img_converted = x_to_t(img_cut, img_cut)

img_converted.plot_img_and_proj(sp_img_converted, x_factor=1, plot_gauss=False, log=False)

raw_on = get_img(lasing_on_dict, 0)
img_on = cut_imageX(raw_on, len(img_converted.x_axis), 0.15)

img_on_converted = x_to_t(img_on, img_cut)

sp_raw_on = subplot(sp_ctr, title='Lasing (x,y)', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

raw_on.plot_img_and_proj(sp_raw_on, plot_gauss=False)



sp_on = subplot(sp_ctr, title='Lasing  cut (x,y)', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

img_on.plot_img_and_proj(sp_on, plot_gauss=False)

sp_on_converted = subplot(sp_ctr, title='Image on', xlabel='t (arb. units)', ylabel='y (mm)')
sp_ctr += 1

img_on_converted.plot_img_and_proj(sp_on_converted, x_factor=1, plot_gauss=False)

ms.show()

