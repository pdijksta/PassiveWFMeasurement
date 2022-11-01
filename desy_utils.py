import numpy as np
from scipy.signal import find_peaks

from . import beam_profile
from . import lasing
from . import tracking
from . import calibration
from . import data_loader
from . import config
from . import logMsg

# From pptx by Nina
matrix_0304 = np.array([
     [0.446543, 2.455919, 0.000000, 0.000000, 0.000000, 0.304000,],
     [0.191976, 3.295264, 0.000000, 0.000000, 0.000000, 0.057857,],
     [0.000000, 0.000000, 1.508069, -39.583671, 0.000000, 0.000000,],
     [0.000000, 0.000000, -0.068349, 2.457107, 0.000000, 0.000000,],
     [0.032525, 0.859669, 0.000000, 0.000000, 1.000000, -0.000199,],
     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000,],
     ])

default_optics = {
        'betax': 34.509,
        'alphax': 1.068,
        'betay': 54.205,
        'alphay': -1.864,
        }

def prepare_image_data(image_data):
    new_images = np.zeros_like(image_data, dtype=float)
    for n_image, image in enumerate(image_data):
        new_image = image - np.quantile(image, 0.5)
        new_image = np.clip(new_image, 0, None)
        new_images[n_image] = new_image
    return new_images


class Xfel_data(logMsg.LogMsgBase):
    def __init__(self, identifier, filename_or_data, charge, energy_eV, pixelsize, init_distance, optics_at_streaker=default_optics, matrix=matrix_0304, gap=20e-3, profile=None, logger=None):

        self.identifier = identifier
        self.logger = logger
        self.beamline = beamline = 'SASE2'
        self.structure_name = 'SASE2'
        self.screen_name = 'SASE2'
        self.matrix = matrix
        self.optics_at_streaker = optics_at_streaker
        self.charge = charge
        self.rec_obj = None

        if type(filename_or_data) in (dict, np.lib.npyio.NpzFile):
            data = filename_or_data
        else:
            data = np.load(filename_or_data)
        self.raw_data = data

        self.profile = profile
        if profile is None:
            crisp_intensity = np.nan_to_num(data['crisp'][1])
            if np.any(crisp_intensity):
                crisp_profile = beam_profile.BeamProfile(data['crisp'][0]*1e-15, crisp_intensity, energy_eV, charge)
                self.profile = crisp_profile
            else:
                self.logMsg('All zeros in crisp intensity')

        x_axis_m = np.arange(0, data['images'][0].shape[1], dtype=float)*pixelsize
        y_axis_m = np.arange(0, data['images'][0].shape[0], dtype=float)*pixelsize
        new_images = prepare_image_data(data['images'])
        self.data = data = {
                'pyscan_result': {
                    'x_axis_m': x_axis_m,
                    'y_axis_m': y_axis_m,
                    'image': new_images[:,::-1],
                    },
                'meta_data_begin': {
                    '%s:ENERGY' % beamline: energy_eV/1e6,
                    '%s:GAP' % beamline: gap*1e3,
                    '%s:CENTER' % beamline: -(gap/2-init_distance)*1e3,
                    '%s:CHARGE' % beamline: charge*1e12,
                    },
                }
        self.meta_data = data['meta_data_end'] = data['meta_data_begin']

        forward_options = config.get_default_forward_options()
        forward_options['screen_smoothen'] = 20e-6
        backward_options = config.get_default_backward_options()
        reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
        reconstruct_gauss_options['gauss_profile_t_range'] = 800e-15
        beam_spec = config.get_default_beam_spec()
        beam_spec['nemitx'] = beam_spec['nemity'] = 500e-9
        find_beam_position_options = config.get_default_find_beam_position_options()
        beam_optics = self.optics_at_streaker

        calib0 = calibration.StructureCalibration(self.structure_name, 0, 0, 0)
        tracker = tracking.Tracker(self.beamline, self.screen_name, self.structure_name, self.meta_data, calib0, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, gen_lat=False, matrix=self.matrix, logger=self.logger)
        tracker.optics_at_streaker = beam_optics
        self.tracker = tracker

    def limit_images(self, limit):
        self.data['pyscan_result']['image'] = self.data['pyscan_result']['image'][:limit]

    def calibrate_screen0(self, sp=None, profile=None, backup_profile=None):
        if profile is None:
            profile = self.profile
        if profile is None:
            profile = backup_profile
        # This function will not work if the streaking is on the other side, for example if the R34 changes sign
        tracker = self.tracker
        axis, proj, _, index = data_loader.screen_data_to_median(self.data['pyscan_result'], self.tracker.structure.dim)
        trans_dist = self.get_median_sd()
        trans_dist.smoothen(20e-6)

        method0 = tracker.find_beam_position_options['method']
        #position_explore0 = tracker.find_beam_position_options['position_explore']
        try:
            tracker.find_beam_position_options['method'] = 'beamsize'
            #tracker.find_beam_position_options['position_explore'] = 100e-6
            find_beam_position_dict = tracker.find_beam_position(tracker.beam_position, trans_dist, profile)
        finally:
            #tracker.find_beam_position_options['position_explore'] = position_explore0
            tracker.find_beam_position_options['method'] = method0

        def do_interp(trans_dist, half_int=None):
            indices, _ = find_peaks(trans_dist.intensity, height=0.2*trans_dist.intensity.max())
            if forward_screen.mean() > 0:
                index = indices[0]
                interp_intensity = trans_dist.intensity[:index].copy()
                interp_x = trans_dist.x[:index]
            else:
                index = indices[-1]
                interp_intensity = trans_dist.intensity[index:][::-1].copy()
                interp_x = trans_dist.x[index:][::-1]
            if half_int is None:
                half_int = trans_dist.intensity[index]/2
            interp_intensity[interp_intensity < interp_intensity.max()*0.2] = 0
            half_peak_x = np.interp(half_int, interp_intensity, interp_x)
            return half_peak_x, half_int, index


        forward_screen = find_beam_position_dict['sim_screen']
        forward_screen.normalize()
        half_peak_x_sim, half_int, index_sim = do_interp(forward_screen)

        half_peak_x, _, index_meas = do_interp(trans_dist, half_int)

        delta_x = -(half_peak_x - half_peak_x_sim)
        self.data['pyscan_result']['%s_axis_m' % tracker.structure.dim.lower()] += delta_x
        self.logMsg('Shifted axis by %.3f mm' % (delta_x*1e3))

        # For debug only
        if sp is not None:
            sp.plot(forward_screen.x*1e3, forward_screen.intensity)
            sp.plot(trans_dist.x*1e3, trans_dist.intensity)
            sp.plot((trans_dist.x+delta_x)*1e3, trans_dist.intensity)
            sp.axvline(trans_dist.x[index_meas]*1e3, color='black')
            sp.axvline(half_peak_x*1e3, color='gray')
            sp.axvline(forward_screen.x[index_sim]*1e3, color='black')
            sp.axvline(half_peak_x_sim*1e3, color='gray')
        #import matplotlib.pyplot as plt
        #plt.show()
        #import pdb; pdb.set_trace()

    def get_median_sd(self):
        axis, proj, _, index = data_loader.screen_data_to_median(self.data['pyscan_result'], self.tracker.structure.dim)
        trans_dist = beam_profile.ScreenDistribution(axis, proj, total_charge=self.charge)
        trans_dist.aggressive_cutoff(self.tracker.forward_options['screen_cutoff'])
        trans_dist.crop()
        trans_dist.normalize()
        self.median_index = index
        return trans_dist

    def calibrate_distance(self, ref_profile=None):
        tracker = self.tracker
        if ref_profile is None:
            ref_profile = self.profile

        trans_dist = self.get_median_sd()
        exp0 = tracker.find_beam_position_options['position_explore']
        #tracker.find_beam_position_options['position_explore'] = 100e-6
        find_beam_position_dict = tracker.find_beam_position(tracker.beam_position, trans_dist, ref_profile)
        delta_position = find_beam_position_dict['delta_position']
        if round(abs(delta_position*1e6)) == round(abs(tracker.find_beam_position_options['position_explore']*1e6)):
            raise ValueError('Beam position could not be found. Delta position: %0.f um' % (delta_position*1e6))
        new_beam_position = find_beam_position_dict['beam_position']
        distance = tracker.structure_gap/2. - abs(new_beam_position)
        self.meta_data[self.beamline+':CENTER'] = -(tracker.structure_gap/2 - distance)*1e3
        self.tracker.meta_data = self.meta_data
        self.logMsg('Distance calibrated to %.0f um. Change by %.0f um' % (distance*1e6, delta_position*1e6))
        tracker.find_beam_position_options['position_explore'] = exp0
        return distance

    def init_images(self, lasing_options=None):
        if lasing_options is None:
            lasing_options = config.get_default_lasing_options()
        lasing_options['subtract_quantile'] = 0
        self.rec_obj = lasing.LasingReconstructionImages(self.identifier, self.tracker, lasing_options)
        self.rec_obj.add_dict(self.data)

    def get_images(self, lasing_options=None):
        if self.rec_obj is None:
            self.init_images(lasing_options=lasing_options)
        self.rec_obj.process_data()
        return self.rec_obj

    def cut_axes(self, cutX=None, cutY=None):
        x_axis_m = self.data['pyscan_result']['x_axis_m']
        y_axis_m = self.data['pyscan_result']['y_axis_m']
        new_images = self.data['pyscan_result']['image']
        if cutX:
            mask_x = np.logical_and(x_axis_m >= cutX[0], x_axis_m <= cutX[1])
            x_axis_m = x_axis_m[mask_x]
            new_images = new_images[:,:,mask_x]
        if cutY:
            mask_y = np.logical_and(y_axis_m >= cutY[0], y_axis_m <= cutY[1])
            y_axis_m = y_axis_m[mask_y]
            new_images = new_images[:,mask_y,:]
        self.data['pyscan_result']['x_axis_m'] = x_axis_m
        self.data['pyscan_result']['y_axis_m'] = y_axis_m
        self.data['pyscan_result']['image'] = new_images

