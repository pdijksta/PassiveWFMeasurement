import numpy as np
from scipy.signal import find_peaks

from . import beam_profile
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

class Xfel_data:
    def __init__(self, filename_or_data, charge, energy_eV, pixelsize, optics_at_streaker=default_optics, matrix=matrix_0304, gap=20e-3, distance=500e-6, profile=None, logger=None):
        if type(filename_or_data) in (dict, np.lib.npyio.NpzFile):
            data = filename_or_data
        else:
            data = np.load(filename_or_data)

        self.beamline = beamline = 'SASE2'
        self.structure_name = 'SASE2'
        self.screen_name = 'SASE2'
        self.matrix = matrix
        self.optics_at_streaker = optics_at_streaker
        self.logger = logger
        self.charge = charge

        if profile is None:
            crisp_intensity = data['crisp'][1]
            if np.any(np.isnan(crisp_intensity)):
                raise ValueError('NaNs in crisp intensity')
            if not np.any(crisp_intensity):
                raise ValueError('All zeros in crisp intensity')
            crisp_profile = beam_profile.BeamProfile(data['crisp'][0]*1e-15, crisp_intensity, energy_eV, charge)
        self.profile = crisp_profile


        new_images = np.zeros_like(data['images'])
        for n_image, image in enumerate(data['images']):
            new_image = image - np.quantile(image, 0.5)
            new_image = np.clip(new_image, 0, None)
            new_images[n_image] = new_image

        x_axis_m = np.arange(0, data['images'][0].shape[1], dtype=float)*pixelsize
        y_axis_m = np.arange(0, data['images'][0].shape[0], dtype=float)*pixelsize
        self.data = data = {
                'pyscan_result': {
                    'x_axis_m': x_axis_m,
                    'y_axis_m': y_axis_m,
                    'image': data['images'],
                    },
                'meta_data_begin': {
                    '%s:ENERGY' % beamline: energy_eV/1e6,
                    '%s:GAP' % beamline: gap*1e3,
                    '%s:CENTER' % beamline: (gap/2-distance)*1e3,
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
        tracker = tracking.Tracker(self.beamline, self.screen_name, self.structure_name, self.meta_data, calib0, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, gen_lat=False, matrix=self.matrix)
        tracker.optics_at_streaker = beam_optics
        self.tracker = tracker

    def calibrate_axis(self, sp=None, profile=None, backup_profile=None):
        if profile is None:
            profile = self.profile
        if profile is None:
            profile = backup_profile
        # This function will not work if the streaking is on the other side, for example if the R34 changes sign
        tracker = self.tracker
        axis, proj, _, index = data_loader.screen_data_to_median(self.data['pyscan_result'], self.tracker.structure.dim)
        trans_dist = beam_profile.ScreenDistribution(axis, proj, total_charge=self.charge)
        trans_dist.aggressive_cutoff(0.02)
        trans_dist.crop()
        trans_dist.normalize()

        method0 = tracker.find_beam_position_options['method']
        position_explore0 = tracker.find_beam_position_options['position_explore']
        try:
            tracker.find_beam_position_options['method'] = 'beamsize'
            tracker.find_beam_position_options['position_explore'] = 100e-6
            find_beam_position_dict = tracker.find_beam_position(tracker.beam_position, trans_dist, profile)
        finally:
            tracker.find_beam_position_options['position_explore'] = position_explore0
            tracker.find_beam_position_options['method'] = method0

        forward_screen = find_beam_position_dict['sim_screen']
        forward_screen.normalize()
        indices, _ = find_peaks(forward_screen.intensity, height=0.2*forward_screen.intensity.max())
        half_int = forward_screen.intensity[indices[0]]/2
        half_peak_x_sim = np.interp(half_int, forward_screen.intensity[:indices[0]], forward_screen.x[:indices[0]])

        indices, _ = find_peaks(trans_dist.intensity, height=0.2*trans_dist.intensity.max())
        half_peak_x = np.interp(half_int, trans_dist.intensity[:indices[0]], trans_dist.x[:indices[0]])
        #print(half_int, half_peak_x_sim, half_peak_x)

        delta_x = -(half_peak_x - half_peak_x_sim)
        self.data['pyscan_result']['%s_axis_m' % tracker.structure.dim.lower()] += delta_x
        logMsg.logMsg('Shifted axis by %.3f mm' % (delta_x*1e3), logger=self.logger)

        # For debug only
        if sp is not None:
            sp.plot(forward_screen.x*1e3, forward_screen.intensity)
            sp.plot(trans_dist.x*1e3, trans_dist.intensity)
            sp.plot((trans_dist.x+delta_x)*1e3, trans_dist.intensity)


