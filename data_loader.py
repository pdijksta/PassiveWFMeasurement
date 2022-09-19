import numpy as np
from . import gaussfit
from . import h5_storage
from . import config
from . import image_analysis

default_cutoff = config.get_default_forward_options()['screen_cutoff']

def get_median(projx, method, output):
    """
    From list of projections, return the median one
    Methods: gf_mean, gf_sigma, mean, rms
    Output: proj, index
    """
    x_axis = np.arange(len(projx[0]))
    all_mean = []
    for proj in projx:
        proj = proj - proj.min()
        if method == 'gf_mean':
            gf = gaussfit.GaussFit(x_axis, proj)
            all_mean.append(gf.mean)
        elif method == 'gf_sigma':
            gf = gaussfit.GaussFit(x_axis, proj)
            all_mean.append(gf.sigma)
        elif method == 'mean':
            mean = np.nansum(x_axis*proj) / np.nansum(proj)
            all_mean.append(mean)
        elif method == 'std':
            mean = np.nansum(x_axis*proj) / np.nansum(proj)
            rms = np.sqrt(np.nansum((x_axis-mean)**2 * proj) / np.nansum(proj))
            all_mean.append(rms)
        else:
            raise ValueError(method)
    index_median = np.argsort(all_mean)[len(all_mean)//2]
    projx_median = projx[index_median]
    if output == 'proj':
        return projx_median, index_median
    elif output == 'index':
        return index_median

def screen_data_to_median(pyscan_result, dim, output='data'):
    """
    dim: 'X' or 'Y'
    Returns: x_axis, proj, charge
    """
    if dim == 'X':
        x_axis = pyscan_result['x_axis_m'].astype(np.float64)
        projx = pyscan_result['image'].sum(axis=-2, dtype=np.float64)
    elif dim == 'Y':
        x_axis = pyscan_result['y_axis_m'].astype(np.float64)
        projx = pyscan_result['image'].sum(axis=-1, dtype=np.float64)

    for pv in config.beamline_chargepv.values():
        if pv in pyscan_result:
            charge = pyscan_result[pv].mean()*1e-12
            break
    else:
        charge = None
    if output == 'data':
        proj, index = get_median(projx, 'mean', 'proj')
        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            proj = proj[::-1]
        return x_axis, proj, charge, index
    else:
        return get_median(projx, 'mean', 'index')

class DataLoaderMultiPosition:
    def add_data(self, positions, images, x_axis_m, y_axis_m, meta_data, screen_name, position_key):
        self.single_position_data = []
        self.positions = positions
        for ctr, position in enumerate(self.positions):
            new_meta_data = meta_data.copy()
            new_meta_data[position_key] = position*1e3
            sp = DataLoaderSinglePosition(images[ctr], x_axis_m, y_axis_m, new_meta_data, screen_name)
            self.single_position_data.append(sp)

class DataLoaderSinglePosition:
    def __init__(self, images, x_axis_m, y_axis_m, meta_data, screen_name):
        self.images = images
        self.x_axis_m = x_axis_m
        self.y_axis_m = y_axis_m
        self.projx = np.sum(self.images, axis=1, dtype=np.float64)
        self.projy = np.sum(self.images, axis=2, dtype=np.float64)

    def noise_cut(self, subtract_quantile, max_quantile):
        old_images = self.images
        new_images = np.zeros_like(old_images)
        for ctr in range(len(new_images)):
            if subtract_quantile is not None:
                new_image = old_images[ctr] - np.quantile(old_images[ctr], subtract_quantile)
            else:
                new_image = old_images[ctr]
            if max_quantile is None:
                max_ = None
            else:
                max_ = np.quantile(new_image, max_quantile)
            new_images[ctr] = np.clip(new_image, 0, max_)
        self.images = new_images

    def get_median_index(self, dimension):
        if dimension in ('X', 'x'):
            return get_median(self.projx, 'mean', 'index')
        elif dimension in ('Y', 'y'):
            return get_median(self.projy, 'mean', 'index')

    def get_median_image(self, dimension, **kwargs):
        index = self.get_median_index(dimension)
        img = image_analysis.Image(self.images[index].astype(np.float64), self.x_axis_m, self.y_axis_m, **kwargs)
        return img

class PSICalibrationData(DataLoaderMultiPosition):
    def __init__(self, raw_data):
        positions = raw_data['streaker_offsets']
        x_axis_m = raw_data['pyscan_result']['x_axis_m']
        y_axis_m = raw_data['pyscan_result']['y_axis_m']
        images = raw_data['pyscan_result']['image']
        meta_data = raw_data['meta_data_begin']
        screen_name = raw_data['screen']
        position_key = raw_data['streaker'] + ':CENTER'
        assert position_key in meta_data
        self.add_data(positions, images, x_axis_m, y_axis_m, meta_data, screen_name, position_key)

class PSISinglePositionData(DataLoaderSinglePosition):
    def __init__(self, file_or_dict, screen_name):
        if type(file_or_dict) is dict:
            raw_data = file_or_dict
        else:
            raw_data = h5_storage.loadH5Recursive(file_or_dict)

        x_axis_m = raw_data['pyscan_result']['x_axis_m']
        y_axis_m = raw_data['pyscan_result']['y_axis_m']
        images = raw_data['pyscan_result']['image']
        meta_data = raw_data['meta_data_begin']

        DataLoaderSinglePosition.__init__(self, images, x_axis_m, y_axis_m, meta_data, screen_name)

