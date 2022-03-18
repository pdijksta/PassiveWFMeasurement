import numpy as np
from . import gaussfit
from . import config

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
        return projx_median
    elif output == 'index':
        return index_median

def screen_data_to_median(pyscan_result, dim):
    """
    dim: 'X' or 'Y'
    Returns: x_axis, proj, charge
    """
    if dim == 'X':
        x_axis = pyscan_result['x_axis_m'].astype(np.float64)
        projx = pyscan_result['image'].astype(np.float64).sum(axis=-2)
    elif dim == 'Y':
        x_axis = pyscan_result['y_axis_m'].astype(np.float64)
        projx = pyscan_result['image'].astype(np.float64).sum(axis=-1)

    for pv in config.beamline_chargepv.values():
        if pv in pyscan_result:
            charge = pyscan_result[pv].mean()*1e-12
            break
    else:
        charge = None
    proj = get_median(projx, 'mean', 'proj')
    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        proj = proj[::-1]
    return x_axis, proj, charge

