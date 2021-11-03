import numpy as np
from . import gaussfit

def get_median(projx, method='gf_mean', output='proj'):
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

    #import matplotlib.pyplot as plt
    #plt.figure()
    #for proj in projx:
    #    plt.plot(proj)
    #plt.plot(projx_median, color='black', lw=3)
    #plt.show()
    #import pdb; pdb.set_trace()

    if output == 'proj':
        return projx_median
    elif output == 'index':
        return index_median

