import numpy as np
from . import beam_profile
from . import tracking
from . import calibration
from . import config

def load(filename_or_data, charge, energy_eV, pixelsize, beamline='SASE2', gap=20e-3, distance=500e-6):
    if type(filename_or_data) in (dict, np.lib.npyio.NpzFile):
        data = filename_or_data
    else:
        data = np.load(filename_or_data)

    crisp_profile = beam_profile.BeamProfile(data['crisp'][0]*1e-15, data['crisp'][1], energy_eV, charge)
    x_axis_m = np.arange(0, data['images'][0].shape[1], dtype=float)*pixelsize
    y_axis_m = np.arange(0, data['images'][0].shape[0], dtype=float)*pixelsize

    outp = {
            'crisp_profile': crisp_profile,
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
    outp['meta_data_end'] = outp['meta_data_begin']
    return outp

matrix_0304 = np.array([
     [0.446543, 2.455919, 0.000000, 0.000000, 0.000000, 0.304000,],
     [0.191976, 3.295264, 0.000000, 0.000000, 0.000000, 0.057857,],
     [0.000000, 0.000000, 1.508069, -39.583671, 0.000000, 0.000000,],
     [0.000000, 0.000000, -0.068349, 2.457107, 0.000000, 0.000000,],
     [0.032525, 0.859669, 0.000000, 0.000000, 1.000000, -0.000199,],
     [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000,],
     ])

def get_tracker(meta_data, matrix, beamline='SASE2', screen_name='SASE2', struct_name='SASE2'):

    forward_options = config.get_default_forward_options()
    forward_options['screen_smoothen'] = 20e-6
    backward_options = config.get_default_backward_options()
    reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
    reconstruct_gauss_options['gauss_profile_t_range'] = 800e-15
    beam_spec = config.get_default_beam_spec()
    beam_spec['nemitx'] = beam_spec['nemity'] = 500e-9
    beam_optics = {'betax': 50, 'betay': 50, 'alphax': 0, 'alphay': 0}
    find_beam_position_options = config.get_default_find_beam_position_options()


    calib0 = calibration.StructureCalibration(struct_name, 0, 0, 0)
    tracker = tracking.Tracker(beamline, beamline, struct_name, meta_data, calib0, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, gen_lat=False, matrix=matrix)
    tracker.optics_at_streaker = beam_optics
    return tracker

