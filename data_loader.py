import itertools
import numpy as np

from . import gaussfit
from . import lattice
from . import beam_profile
from . import h5_storage
from . import config
from . import image_analysis
from . import myplotstyle as ms

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
    Returns: x_axis, proj, charge, index
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

def load_lasing_result(h5_file):
    result_dict = h5_storage.loadH5Recursive(h5_file)
    for key1, key2 in itertools.product(
            ['images_on', 'images_off'],
            ['raw_images', 'tE_images'],
            ):
        d_old = result_dict[key1][key2]
        list_new = [image_analysis.Image(**d_old['list_entry_%i' % x]) for x in range(len(d_old))]
        result_dict[key1][key2] = list_new
    return result_dict

def file_or_dict_to_dict(file_or_dict):
    if type(file_or_dict) is dict:
        return file_or_dict
    else:
        return h5_storage.loadH5Recursive(file_or_dict)

def beamline_from_structure(structure):
    for beamline, structures in config.structure_names.items():
        if structure in structures:
            return beamline
    else:
        raise ValueError('Structure not found!')

def eval_psi_meta_data(meta_data, structure):
    beamline = beamline_from_structure(structure)
    lat = lattice.get_beamline_lattice(beamline, meta_data)
    energy_eV = meta_data[config.beamline_energypv[beamline]]*1e6
    charge = meta_data[config.beamline_chargepv[beamline]]*1e-12
    structure_position = meta_data[structure+':CENTER']*1e-3
    structure_gap = meta_data[structure+':GAP']*1e-3
    return {
            'beamline': beamline,
            'lat': lat,
            'energy_eV': energy_eV,
            'charge': charge,
            'structure_position': structure_position,
            'structure_gap': structure_gap,
            'structure': structure,
            }

def dlsp_to_meta_dict(dlsp):
    structure = dlsp.structure
    beamline = beamline_from_structure(structure)
    lat = dlsp.lat
    energy_eV = dlsp.energy_eV
    charge = dlsp.charge
    structure_position = dlsp.structure_position
    structure_gap = dlsp.structure_gap
    return {
            'beamline': beamline,
            'lat': lat,
            'energy_eV': np.median(energy_eV),
            'charge': np.median(charge),
            'structure_position': structure_position,
            'structure_gap': structure_gap,
            'structure': structure,
            }


class DataLoaderBase:
    def __init__(self):
        self.images = []
        self.sd_dict = {}

    def prepare_data(self):
        x_axis_m = self.x_axis_m
        y_axis_m = self.y_axis_m
        images = self.image_data
        cutX = self.data_loader_options['cutX']
        cutY = self.data_loader_options['cutY']
        subtract_quantile = self.data_loader_options['subtract_quantile']
        subtract_absolute = self.data_loader_options['subtract_absolute']
        void_cutoff = self.data_loader_options['void_cutoff']

        if subtract_quantile and subtract_absolute:
            raise ValueError('Cannot specify both absolute and quantile subtractions!')

        def _cutX(lim1, lim2):
            mask = np.logical_and(x_axis_m >= lim1, x_axis_m <= lim2)
            x_axis_m2 = x_axis_m[mask]
            images2 = images[:,:,mask]
            return x_axis_m2, images2

        def _cutY(lim1, lim2):
            mask = np.logical_and(y_axis_m >= lim1, y_axis_m <= lim2)
            y_axis_m2 = y_axis_m[mask]
            images2 = images[:,mask,:]
            return y_axis_m2, images2

        if cutX is not None:
            x_axis_m, images = _cutX(*cutX)
        if cutY is not None:
            y_axis_m, images = _cutY(*cutY)

        if subtract_absolute:
            np.clip(images-subtract_absolute, 0, None, out=images)
        if subtract_quantile:
            for image in images:
                np.clip(image - np.quantile(image, subtract_quantile), 0, None, out=image)

        if void_cutoff is not None and any(void_cutoff):
            image_sum = images.sum(axis=0)
            profX = beam_profile.AnyProfile(x_axis_m, np.sum(image_sum, axis=0))
            profX.aggressive_cutoff(void_cutoff[0])
            profX.crop()
            x_axis_m, images = _cutX(profX.xx[0], profX.xx[-1])
            profY = beam_profile.AnyProfile(y_axis_m, np.sum(image_sum, axis=1))
            profY.aggressive_cutoff(void_cutoff[1])
            profY.crop()
            y_axis_m, images = _cutY(profY.xx[0], profY.xx[-1])

        self.projx = np.sum(images, axis=1, dtype=np.float64)
        self.projy = np.sum(images, axis=2, dtype=np.float64)

        self.image_data = images
        self.x_axis_m = x_axis_m
        self.y_axis_m = y_axis_m

    def shift_axis(self, dimension, value):
        if dimension == 'X':
            self.x_axis_m = self.x_axis_m - value
        if dimension == 'Y':
            self.y_axis_m = self.y_axis_m - value

    def init_images(self, **kwargs):
        self.images = []
        for img, charge, energy_eV in zip(self.image_data, self.charge, self.energy_eV):
            self.images.append(image_analysis.Image(img, self.x_axis_m, self.y_axis_m, charge, energy_eV, **kwargs))

    def init_screen_distributions(self, dimension):
        self.sd_dict[dimension] = {'sd': [], 'mean': [], 'rms': []}
        for img in self.images:
            sd = img.get_screen_dist(dimension)
            if self.data_loader_options['screen_cutoff_relative']:
                nn = self.data_loader_options['screen_cutoff_edge_points']
                edge1 = np.max(sd.intensity[:nn])
                edge2 = np.max(sd.intensity[-nn:])
                cutoff = min(edge1, edge2)*self.data_loader_options['screen_cutoff_relative_factor']
                cutoff_factor = cutoff/sd.intensity.max()
                sd.aggressive_cutoff(cutoff_factor)

            sd.aggressive_cutoff(self.data_loader_options['screen_cutoff'])
            self.sd_dict[dimension]['sd'].append(sd)
            self.sd_dict[dimension]['mean'].append(sd.mean())
            self.sd_dict[dimension]['rms'].append(sd.rms())
        self.sd_dict[dimension]['mean'] = np.array(self.sd_dict[dimension]['mean'])
        self.sd_dict[dimension]['rms'] = np.array(self.sd_dict[dimension]['rms'])

    def plot_all(self, ny, nx, title=None, only_one_fig=False, figsize=(12,10), plot_kwargs={}, subplots_adjust_kwargs={}):
        fig_ctr = 0
        sp_ctr = np.inf
        figs = []
        all_sps = []
        for n_image, image in enumerate(self.images):
            if sp_ctr > nx*ny:
                if fig_ctr > 0 and only_one_fig:
                    break
                fig = ms.figure(title, figsize=figsize)
                if subplots_adjust_kwargs:
                    fig.subplots_adjust(**subplots_adjust_kwargs)
                figs.append(fig)
                fig_ctr += 1
                subplot = ms.subplot_factory(ny, nx, False)
                sp_ctr = 1
                all_sps.append([])
            sp = subplot(sp_ctr, title='Image %i' % n_image, xlabel='x (mm)', ylabel='y (mm)')
            sp_ctr += 1
            image.plot_img_and_proj(sp, **plot_kwargs)
            all_sps[fig_ctr-1].append(sp)

        return figs, all_sps


class DataLoaderSimple(DataLoaderBase):
    def __init__(self, image_data, x_axis_m, y_axis_m, charge, energy_eV, data_loader_options):
        DataLoaderBase.__init__(self)
        self.image_data = image_data
        self.x_axis_m = x_axis_m
        self.y_axis_m = y_axis_m
        self.charge = charge
        self.energy_eV = energy_eV
        self.data_loader_options = data_loader_options


class DataLoaderSinglePosition(DataLoaderBase):
    def __init__(self, structure, structure_gap, structure_position, images, x_axis_m, y_axis_m, lat, charge, energy_eV, screen_name, data_loader_options):
        DataLoaderBase.__init__(self)
        self.data_loader_options = data_loader_options
        self.structure = structure
        self.structure_gap = structure_gap
        self.structure_position = structure_position
        self.lat = lat
        self.charge = charge if hasattr(charge, '__len__') else [charge]*len(images)
        self.energy_eV = energy_eV if hasattr(energy_eV, '__len__') else [energy_eV]*len(images)

        self.image_data = images.astype(np.float64).copy()
        self.x_axis_m = x_axis_m.astype(np.float64).copy()
        self.y_axis_m = y_axis_m.astype(np.float64).copy()
        self.screen_name = screen_name
        self.prepare_data()

    def get_median_index(self, dimension):
        if dimension in ('X', 'x'):
            return get_median(self.projx, 'mean', 'index')
        elif dimension in ('Y', 'y'):
            return get_median(self.projy, 'mean', 'index')

    def get_median_image(self, dimension, **kwargs):
        index = self.get_median_index(dimension)
        img = image_analysis.Image(self.image_data[index], self.x_axis_m, self.y_axis_m, self.charge[index], self.energy_eV[index], **kwargs)
        return img


class DataLoaderMultiPosition:
    def init(self, structure, structure_gap, positions, images, x_axis_m, y_axis_m, lat, charge, energy_eV, screen_name, data_loader_options, streaking_direction):
        self.structure = structure
        self.screen_name = screen_name
        self.single_position_data = []
        self.positions = positions
        self.streaking_direction = streaking_direction
        self.data_loader_options = data_loader_options
        self.charge = charge
        self.energy_eV = energy_eV
        self.structure_gap = structure_gap
        for ctr, position in enumerate(self.positions):
            sp = DataLoaderSinglePosition(structure, structure_gap, position, images[ctr], x_axis_m, y_axis_m, lat, charge, energy_eV, screen_name, data_loader_options)
            self.single_position_data.append(sp)

    def init_images(self):
        for sp in self.single_position_data:
            sp.init_images()

    def init_sds(self):
        for sp in self.single_position_data:
            sp.init_screen_distributions(self.streaking_direction)

    def adjust_screen0(self):
        zero_positions = self.data_loader_options['zero_positions']
        multi_zero_position = self.data_loader_options['multi_zero_position']
        if multi_zero_position is not None and zero_positions is None:
            index_zero = list(self.positions).index(multi_zero_position)
            sp = self.single_position_data[index_zero]
            means = []
            sp.init_images()
            sp.init_screen_distributions(self.streaking_direction)
            sds = sp.sd_dict[self.streaking_direction]['sd']
            for sd in sds:
                means.append(sd.mean())
            delta = np.median(means)
        for ctr, sp in enumerate(self.single_position_data):
            if zero_positions is not None:
                sp.shift_axis(self.streaking_direction, zero_positions[ctr])
            elif multi_zero_position is not None:
                sp.shift_axis(self.streaking_direction, delta)

    def boolean_filter(self, filter):
        assert len(filter) == len(self.positions)
        indices = np.arange(len(filter), dtype=int)[filter]
        outp = DataLoaderMultiPosition()
        outp.positions = np.take(self.positions, indices)
        outp.single_position_data = list(np.take(self.single_position_data, indices))
        outp.structure = self.structure
        outp.screen_name = self.screen_name
        outp.streaking_direction = self.streaking_direction
        outp.structure_gap = self.structure_gap
        outp.charge = self.charge
        outp.energy_eV = self.energy_eV
        return outp


class PSICalibrationData(DataLoaderMultiPosition):
    def __init__(self, raw_data, data_loader_options):
        positions = raw_data['streaker_offsets']
        x_axis_m = raw_data['pyscan_result']['x_axis_m']
        y_axis_m = raw_data['pyscan_result']['y_axis_m']
        images = raw_data['pyscan_result']['image']
        meta_data = raw_data['meta_data_begin']
        screen_name = raw_data['screen']
        structure = raw_data['structure']
        structure_gap = meta_data[structure + ':GAP']*1e-3
        streaking_direction = config.structure_dimensions[raw_data['structure']]
        meta_dict = eval_psi_meta_data(meta_data, structure)
        self.init(structure, structure_gap, positions, images, x_axis_m, y_axis_m, meta_dict['lat'], meta_dict['charge'], meta_dict['energy_eV'], screen_name, data_loader_options, streaking_direction)
        self.adjust_screen0()


class PSISinglePositionData(DataLoaderSinglePosition):
    def __init__(self, file_or_dict, screen_name, data_loader_options, structure=None):
        raw_data = file_or_dict_to_dict(file_or_dict)
        x_axis_m = raw_data['pyscan_result']['x_axis_m']
        y_axis_m = raw_data['pyscan_result']['y_axis_m']
        images = raw_data['pyscan_result']['image']
        if structure is None:
            structure = raw_data['structure']
        meta_dict = eval_psi_meta_data(raw_data['meta_data_begin'], structure)

        DataLoaderSinglePosition.__init__(self, structure, meta_dict['structure_gap'], meta_dict['structure_position'], images, x_axis_m, y_axis_m, meta_dict['lat'], meta_dict['charge'], meta_dict['energy_eV'], screen_name, data_loader_options)


class PSILinearData(DataLoaderSinglePosition):
    pass

