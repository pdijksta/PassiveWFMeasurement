import numpy as np
from . import gaussfit
from . import config

default_cutoff = config.get_default_forward_options()['screen_cutoff']

def get_median(projx, method, output, cutoff=default_cutoff):
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
            mean = np.sum(x_axis*proj) / np.sum(proj)
            all_mean.append(mean)
        elif method == 'std':
            mean = np.sum(x_axis*proj) / np.sum(proj)
            rms = np.sqrt(np.sum((x_axis-mean)**2 * proj) / np.sum(proj))
            all_mean.append(rms)
        else:
            raise ValueError(method)

    index_median = np.argsort(all_mean)[len(all_mean)//2]
    projx_median = projx[index_median]

    if output == 'proj':
        return projx_median
    elif output == 'index':
        return index_median

def screen_data_to_median(pyscan_result):
    x_axis = pyscan_result['x_axis_m'].astype(np.float64)
    projx = pyscan_result['image'].astype(np.float64).sum(axis=-2)
    proj = get_median(projx, 'mean', 'proj')

    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        proj = proj[::-1]
    return x_axis, proj

