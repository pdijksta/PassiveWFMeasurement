import time
import bisect
import copy
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
        self.gap = gap

        if type(filename_or_data) in (dict, np.lib.npyio.NpzFile):
            data = filename_or_data
        else:
            data = np.load(filename_or_data)
        self.raw_data = data

        self.profile = profile
        if profile is None:
            crisp_intensity = np.nan_to_num(data['crisp'][1])
            crisp_intensity2 = np.nan_to_num(data['crisp_all'][0][1])
            if np.any(crisp_intensity):
                crisp_profile = beam_profile.BeamProfile(data['crisp'][0]*1e-15, crisp_intensity, energy_eV, charge)
                self.profile = crisp_profile
            elif np.any(crisp_intensity2):
                crisp_profile = beam_profile.BeamProfile(data['crisp_all'][0][0]*1e-15, crisp_intensity2, energy_eV, charge)
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
                    '%s:GAP' % beamline: self.gap*1e3,
                    '%s:CENTER' % beamline: 0,
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
        self.set_distance(init_distance)

    def set_distance(self, distance):
        self.data['meta_data_begin']['%s:CENTER' % self.beamline] = -(self.gap/2-distance)*1e3
        self.tracker.meta_data = self.data['meta_data_begin']

    def limit_images(self, limit):
        if limit != np.inf:
            self.data['pyscan_result']['image'] = self.data['pyscan_result']['image'][:limit]

    def calibrate_screen0(self, sp=None, profile=None, backup_profile=None):
        half_factor = 4
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
        #except:
        #    import pdb; pdb.set_trace()
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
                half_int = trans_dist.intensity[index]/half_factor
            interp_intensity[interp_intensity < interp_intensity.max()*0.2] = 0
            half_peak_x = np.interp(half_int, interp_intensity, interp_x)
            return half_peak_x, half_int, index

        forward_screen = find_beam_position_dict['sim_screen']
        forward_screen.normalize()
        half_peak_x_sim, half_int, index_sim = do_interp(forward_screen)

        half_peak_x, _, index_meas = do_interp(trans_dist, half_int)

        delta_x = -(half_peak_x - half_peak_x_sim)
        self.change_screen0(delta_x)
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
        return delta_x

    def change_screen0(self, delta_x):
        self.data['pyscan_result']['%s_axis_m' % self.tracker.structure.dim.lower()] += delta_x

    def screen_data_to_median(self):
        return data_loader.screen_data_to_median(self.data['pyscan_result'], self.tracker.structure.dim)

    def get_median_sd(self):
        axis, proj, _, index = self.screen_data_to_median()
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


class BumpCalibration:
    def __init__(self, ref_type, ref, ref_distance):
        self.ref_type = ref_type
        self.ref = ref
        self.ref_distance = ref_distance

    def get_distance(self, val):
        return self.ref_distance + (self.ref-val)

_z_corr1 = config.sase2_zpos_dict['CMY.2456.T3']
_z_corr0 = config.sase2_zpos_dict['CMY.2443.T3']
_z_bpm0 = config.sase2_zpos_dict['BPMA.2455.T3']

class XfelDistanceScan:

    bpm_names_factors = [
            ('BPMA.2455.T3', (_z_corr1-_z_corr0)/(_z_bpm0-_z_corr0)),
            ('BPMA.2467.T3', 1),
            ]

    def __init__(self, filenames, charge, energy_eV, pixelsize):
        self.filenames = self.filenames0 = filenames
        self.charge = charge
        self.energy_eV = energy_eV
        self.pixelsize = pixelsize

        self.first_images = None

        self.bumps = bumps = np.zeros(len(filenames))
        self.orbits_mean = bumps.copy()
        self.orbits_rms = bumps.copy()
        self.all_orbits = all_orbits = {bpm_name: [] for bpm_name, _ in self.bpm_names_factors}

        for ctr, filename in enumerate(self.filenames):
            data = np.load(filename)
            bump = data['bump']*1e-3
            for bpm_name, factor in self.bpm_names_factors:
                index = np.argwhere(data['orbit_list'][0,:,0] == bpm_name).squeeze()
                orbits = np.array(data['orbit_list'][:,index,2], float)*factor
                all_orbits[bpm_name].append(orbits)
                if bpm_name == 'BPMA.2455.T3':
                    bumps[ctr] = bump
                    self.orbits_mean[ctr] = np.mean(orbits)
                    self.orbits_rms[ctr] = np.std(orbits)

        for key, val in all_orbits.items():
            all_orbits[key] = np.array(val)

    def sort(self, key='bump'):
        if key == 'bump':
            sort = np.argsort(self.bumps)
        else:
            raise NotImplementedError(key)

        self.filenames = np.array(self.filenames)[sort]
        self.bumps = self.bumps[sort]
        self.orbits_mean = self.orbits_mean[sort]
        self.orbits_rms = self.orbits_rms[sort]
        if self.first_images is not None:
            self.first_images = self.first_images[sort]

        for key, val in self.all_orbits.items():
            self.all_orbits[key] = val[sort]

    def get_first_images(self):
        self.first_images = []
        for ctr, filename in enumerate(self.filenames):
            init_distance = 3.5e-3 - self.orbits_mean[ctr]
            analyzer = Xfel_data(filename, filename, self.charge, self.energy_eV, self.pixelsize, init_distance)
            analyzer.limit_images(1)
            analyzer.init_images()
            self.first_images.append(analyzer.rec_obj.images_xy[0])
        return self.first_images

    def get_crisp_distances(self):
        distances = []
        for ctr, filename in enumerate(self.filenames):
            init_distance = 3.5e-3 - self.orbits_mean[ctr]
            analyzer = Xfel_data(filename, filename, self.charge, self.energy_eV, self.pixelsize, init_distance)
            analyzer.limit_images(1)
            analyzer.calibrate_screen0()
            analyzer.tracker.find_beam_position_options['position_explore'] = 200e-6
            distances.append(analyzer.calibrate_distance())
        return np.array(distances)

class SingleSidedCalibration(logMsg.LogMsgBase):

    bpm_names_factors = [
            ('BPMA.2455.T3', (_z_corr1-_z_corr0)/(_z_bpm0-_z_corr0)),
            ('BPMA.2467.T3', 1),
            ]

    def __init__(self, pixelsize, charge, energy_eV, beamline, delta_gap_range, images_per_file=np.inf, logger=None, structure_calib_options=None):
        self.logger = logger
        self.pixelsize = pixelsize
        self.charge = charge
        self.energy_eV = energy_eV
        self.images_per_file = images_per_file
        self.beamline = beamline
        self.use_bpm = 'BPMA.2455.T3'
        if structure_calib_options is None:
            self.structure_calib_options = config.get_default_structure_calibrator_options()
            self.structure_calib_options['delta_gap_range'] = np.array([-delta_gap_range/2, delta_gap_range/2])
            self.structure_calib_options['max_iter'] = 5
            self.structure_calib_options['prec'] = 2e-6
        else:
            self.structure_calib_options = structure_calib_options

    def init_files(self, filenames, min_orbit=-np.inf, max_orbit=np.inf):
        orbits = []
        trackers = []
        raw_screens = []
        crisp_profiles = []

        init_pos = config.init_plate_pos_dict[self.beamline]

        for filename in filenames:
            analyzer = Xfel_data(filename, filename, self.charge, self.energy_eV, self.pixelsize, init_distance=0, logger=self.logger)
            analyzer.limit_images(self.images_per_file)
            data = analyzer.raw_data

            if self.images_per_file == np.inf:
                images_per_file = len(data['orbit_list'])
            else:
                images_per_file = self.images_per_file

            all_orbits = {}
            for bpm_name, factor in self.bpm_names_factors:
                index = np.argwhere(data['orbit_list'][0,:,0] == bpm_name).squeeze()
                file_orbits = (np.array(data['orbit_list'][:images_per_file,index,2], float)*factor).squeeze()
                all_orbits[bpm_name] = file_orbits
            analyzer.set_distance(init_pos-abs(file_orbits.mean()))
            analyzer.calibrate_screen0()

            dim = analyzer.tracker.structure.dim
            pyscan_result = analyzer.data['pyscan_result']
            axis = pyscan_result['%s_axis_m' % dim.lower()].astype(np.float64)
            if dim == 'X':
                proj = pyscan_result['image'].sum(axis=-2, dtype=np.float64)
            elif dim == 'Y':
                proj = pyscan_result['image'].sum(axis=-1, dtype=np.float64)

            images_per_file2 = min(images_per_file, len(proj))
            for n_image in range(images_per_file2):
                if images_per_file2 == 1:
                    orbit = all_orbits[self.use_bpm]
                else:
                    orbit = all_orbits[self.use_bpm][n_image]
                if orbit > min_orbit and orbit < max_orbit:
                    init_distance = init_pos - abs(orbit)
                    analyzer.set_distance(init_distance)
                    tracker = copy.deepcopy(analyzer.tracker)
                    raw_screen = beam_profile.ScreenDistribution(axis, proj[n_image], total_charge=self.charge)

                    orbits.append(orbit)
                    raw_screens.append(raw_screen)
                    trackers.append(tracker)
                    if n_image < len(analyzer.raw_data['crisp_all']):
                        crisp_xx, crisp_yy = analyzer.raw_data['crisp_all'][n_image]
                        if np.any(crisp_yy):
                            crisp_profile = beam_profile.BeamProfile(crisp_xx*1e-15, crisp_yy, self.energy_eV, self.charge)
                            crisp_profile.crop()
                            crisp_profile.reshape(5000)
                            crisp_profiles.append(crisp_profile)

        sort = np.argsort(orbits)
        self.orbits = np.take(orbits, sort)
        self.trackers = list(np.take(trackers, sort))
        self.raw_screens = list(np.take(raw_screens, sort))
        self.crisp_profiles = crisp_profiles

    def calibrate_plate_position(self, rms_chargecut=0):
        self.logMsg('Calibrating struct pos with %i measurements' % len(self.raw_screens))
        t0 = time.time()
        plate_positions = []
        indices = []
        fit_params = []
        fit_slopes = []
        fit_covs = []
        rms_durations = []
        rec_profiles = []
        distances = []
        prec = self.structure_calib_options['prec']
        max_iter = self.structure_calib_options['max_iter']

        def do_calc(plate_pos):
            if plate_pos in plate_positions:
                return np.nan, False
            this_profiles, this_rms_durations, this_distances = [], [], []
            for orbit, tracker, raw_screen in zip(self.orbits, self.trackers, self.raw_screens):
                distance = plate_pos - abs(orbit)
                this_distances.append(distance)
                beam_position = tracker.structure_gap/2. - distance
                gauss_dict = tracker.reconstruct_profile_Gauss_forced(tracker.structure_gap, beam_position, raw_screen)
                profile = gauss_dict['reconstructed_profile']
                this_profiles.append(profile)
                this_rms_durations.append(profile.rms_chargecut(rms_chargecut))
            fit, cov = np.polyfit(self.orbits, this_rms_durations, 1, cov=True)
            this_fit_params = np.poly1d(fit)
            this_fit_slope = this_fit_params[1]
            index = bisect.bisect(fit_slopes, this_fit_slope)
            fit_slopes.insert(index, this_fit_slope)
            fit_params.insert(index, this_fit_params)
            fit_covs.insert(index, cov)
            rms_durations.insert(index, this_rms_durations)
            rec_profiles.insert(index, this_profiles)
            plate_positions.insert(index, plate_pos)
            distances.insert(index, this_distances)

            indices.append(index)
            return True

        def get_plate_pos():
            status = (min(fit_slopes) < 0 and max(fit_slopes) > 0)
            plate_pos0 = np.interp(0, fit_slopes, plate_positions)
            plate_pos = np.round(plate_pos0/prec)*prec
            return status, plate_pos

        delta_pos_range = self.structure_calib_options['delta_gap_range']/2

        for delta_pos in delta_pos_range.min(), delta_pos_range.max():
            plate_pos0 = config.init_plate_pos_dict[self.beamline]
            plate_pos = plate_pos0 + delta_pos
            do_calc(plate_pos)

        for n_iter in range(max_iter):
            status, plate_pos = get_plate_pos()
            if status:
                status = do_calc(plate_pos)
            if not status:
                break
        n_iter += 1

        status, plate_pos = get_plate_pos()

        time_spent = time.time() - t0
        self.logMsg('Calibrated struct pos %.3f mm in %.1f s and %i iterations. Status: %i' % (plate_pos*1e3, time_spent, n_iter, status))
        outp = {
                'plate_positions': np.array(plate_positions),
                'fit_params': fit_params,
                'fit_slopes': np.array(fit_slopes),
                'fit_covs': np.array(fit_covs),
                'rms_durations': np.array(rms_durations),
                'rec_profiles': rec_profiles,
                'distances': np.array(distances),
                'status': status,
                'iterations': n_iter,
                'final_index': indices[-1],
                'final_position': plate_pos,
                'options': self.structure_calib_options,
                'orbits': self.orbits,
                'raw_screens': self.raw_screens,
                'crisp_profiles': self.crisp_profiles,
                }

        return outp

