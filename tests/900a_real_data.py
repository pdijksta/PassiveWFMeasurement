import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.image_analysis as image_analysis
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.myplotstyle as ms

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
    x_mean = xProfile.gaussfit.mean

    yProfile.aggressive_cutoff(y_cutoff)
    yProfile.crop()

    img_cut1 = raw_image.cutY(yProfile.xx[0], yProfile.xx[-1])
    old_x_axis = img_cut1.x_axis
    img_cut1.x_axis = old_x_axis - x_mean
    mask_x = np.zeros_like(img_cut1.x_axis, dtype=bool)
    where0 = int(np.argmin(img_cut1.x_axis**2).squeeze())

    one_left = x_len//2
    one_right = x_len - one_left

    mask_x[where0-one_left:where0+one_right] = 1

    assert mask_x.sum() == x_len

    img_cut1.x_axis = img_cut1.x_axis[mask_x]
    img_cut1.image = img_cut1.image[:,mask_x]
    #import pdb; pdb.set_trace()
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


dir_ = '/sf/data/measurements/2021/11/12/'
dir_ = '/mnt/data/data_2021-11-12/'

lasing_off_file = dir_+'2021_11_12-23_22_11_Lasing_False_SATBD02-DSCR050.h5'
lasing_on_file = dir_+'2021_11_12-23_15_06_Lasing_True_SATBD02-DSCR050.h5'

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
subplot = ms.subplot_factory(2, 4, grid=False)
sp_ctr = 1

sp_img = subplot(sp_ctr, title='Calib image (x,y)', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

raw_img.plot_img_and_proj(sp_img)

sp_img = subplot(sp_ctr, title='Calib image (x,t)', xlabel='x (mm)', ylabel='t (arb. units)')
sp_ctr += 1


img_cut.plot_img_and_proj(sp_img, y_factor=1, plot_gauss=False, log=False)

sp_img_converted = subplot(sp_ctr, title='Calib image (t,y)', xlabel='t (arb. units)', ylabel='y (mm)')
sp_ctr += 1

sp_espread = subplot(sp_ctr, title='Energy spread', xlabel='t (arb. units)', ylabel='Slice rms size (mm)')
sp_ctr += 1

img_converted = x_to_t(img_cut, img_cut)

img_converted2 = img_converted.child(img_converted.image, img_converted.x_axis, img_cut0.y_axis)

img_converted2.plot_img_and_proj(sp_img_converted, x_factor=1, plot_gauss=False, log=False)

raw_on = get_img(lasing_on_dict, 0)
img_on = cut_imageX(raw_on, len(img_converted.x_axis), 0.15)

img_on_converted = x_to_t(img_on, img_cut)

sp_raw_on = subplot(sp_ctr, title='Lasing (x,y)', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

raw_on.plot_img_and_proj(sp_raw_on, plot_gauss=False)


sp_on = subplot(sp_ctr, title='Lasing  cut (x,y)', xlabel='x (mm)', ylabel='y (mm)')
sp_ctr += 1

img_on.plot_img_and_proj(sp_on, plot_gauss=True)

sp_on_converted = subplot(sp_ctr, title='Image on', xlabel='t (arb. units)', ylabel='y (mm)')
sp_ctr += 1

img_on_converted.plot_img_and_proj(sp_on_converted, x_factor=1, plot_gauss=False)

def get_espread(image):
    yy = image.y_axis
    im = image.image
    divisor = np.sum(im, axis=0)
    divisor[divisor==0] = np.inf
    mean = np.sum(im*yy[:,np.newaxis], axis=0) / divisor
    std = np.sqrt(np.sum(im*(yy[:,np.newaxis]-mean)**2, axis=0) / divisor)
    return std

espread_on = get_espread(img_on_converted)
espread_off = get_espread(img_converted2)

sp_espread.plot(img_on_converted.x_axis, espread_on*1e3, label='Lasing on')
sp_espread.plot(img_converted2.x_axis, espread_off*1e3, label='Lasing off')

sp_espread.legend()

ms.show()

