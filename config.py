#import numpy as np
import itertools
import collections

#Important naming convention
#    beam_position: Position of the beam relative to the middle of the structure
#    structure_position: Position of the structure as read by the :CENTER PV
#
#    The beam_position is calculated as follows:
#    -(structure_position - structure_position0)
#    The sign is opposite to the structure_position!
#
#    structure_position0: calibration

structure_names = {
        'Aramis': collections.OrderedDict([
            (0, 'SARUN18-UDCP010'),
            (1, 'SARUN18-UDCP020'),
            ]),
        'Athos': collections.OrderedDict([
            (0, 'SATMA02-UDCP045'),
            ]),
        }

aramis_structure_parameters = {
        'Ls': 1,
        'p': 500e-6,
        'g': 250e-6,
        'w': 10e-3,
        }

structure_parameters = {
        'SARUN18-UDCP010': aramis_structure_parameters,
        'SARUN18-UDCP020': aramis_structure_parameters,
        'SATMA02-UDCP045': aramis_structure_parameters,
        }

beamline_quads = {
        'Aramis': [
            'SARUN18.MQUA080',
            'SARUN19.MQUA080',
            'SARUN20.MQUA080',
            'SARBD01.MQUA020',
            'SARBD02.MQUA030',
            ],
        'Athos': [
            'SATMA02.MQUA050',
            'SATMA02.MQUA070',
            'SATBD01.MQUA010',
            'SATBD01.MQUA030',
            'SATBD01.MQUA050',
            'SATBD01.MQUA070',
            'SATBD01.MQUA090',
            'SATBD02.MQUA030',
            ],
        }

beamline_chargepv = {
        'Aramis': 'SINEG01-DICT215:B1_CHARGE-OP',
        'Athos': 'SINEG01-DICT215:B2_CHARGE-OP',
        }

beamline_energypv = {
        'Aramis': 'SARBD01-MBND100:ENERGY-OP',
        'Athos': 'SATBD01-MBND200:ENERGY-OP',
        }

beamline_screens = {
        'Aramis': 'SARBD02-DSCR050',
        'Athos': 'SATBD02-DSCR050',
        }

gas_monitor_pvs = {
        'Aramis': 'SARFE10-PBPG050:PHOTON-ENERGY-PER-PULSE-AVG',
        }

default_optics = {
        'Aramis': {
            'betax': 4.968,
            'alphax': -0.563,
            'betay': 16.807,
            'alphay': 1.782,
            },
        'Athos': {
            'betax': 30.9,
            'alphax': 3.8,
            'betay': 69.4,
            'alphay': -14.3,
            },
        }

_aramis_pvs = ['SARUN%02i-DBPM070:%s1' % (i, dim) for i, dim in itertools.product(range(1, 21), ('X', 'Y'))]
_aramis_pvs += ['SARBD01-DBPM040:%s1' % dim for dim in ('X', 'Y')]
_aramis_pvs += ['SARBD02-DBPM010:%s1' % dim for dim in ('X', 'Y')]

beamline_bpm_pvs = {
        'Aramis': _aramis_pvs,
        }

beamline_charge_pvs_bsread = {
        'Aramis': ['SARBD01-DICT030:B1_CHARGE', 'SINEG01-DICT215:B1_CHARGE'],
        }

all_structures = []
for beamline, beamline_dict in structure_names.items():
    all_structures.extend([x for x in beamline_dict.values()])

#get_default_tracker_settings = lambda: {
#        'magnet_file': None,
#        'timestamp': None,
#        'struct_lengths': np.array([1., 1.]),
#        'n_particles': int(100e3),
#        'n_emittances': np.array([755, 755])*1e-9,
#        'screen_bins': 500,
#        'screen_cutoff': 2e-2,
#        'smoothen': 20e-6,
#        'profile_cutoff': 1e-2,
#        'len_screen': 2000,
#        'quad_wake': False,
#        'bp_smoothen': 1e-15,
#        'override_quad_beamsize': False,
#        'quad_x_beamsize': np.array([10., 10.])*1e-6,
#        }
#
#get_default_gauss_recon_settings = lambda: {
#        'self_consistent': True,
#        'sig_t_range': np.exp(np.linspace(np.log(7), np.log(100), 15))*1e-15,
#        'tt_halfrange': 200e-15,
#        'charge': 200e-12,
#        'method': 'centroid',
#        'delta_gap': (0., 0.)
#        }


def get_default_forward_options():
    return {
            'screen_bins': 1000,
            'screen_smoothen': 20e-6,
            'quad_wake': False,
            'screen_cutoff': 2e-2,
            'len_screen': 5000,
            }.copy()

def get_default_backward_options():
    return {
            'compensate_negative_screen': True,
            'len_profile': 5000,
            'profile_cutoff': 0.1e-2,
            'profile_smoothen': 1e-15,
            }.copy()

def get_default_reconstruct_gauss_options():
    return {
            'gauss_profile_t_range': 400e-15,
            'precision': 0.1e-15,
            'method': 'centroid',
            'sig_t_range': [7e-15, 100e-15],
            }.copy()

def get_default_beam_spec():
    outp = {
            'nemitx': 755e-9,
            'nemity': 755e-9,
            'n_mesh': 500,
            'cutoff_sigma': 5,
            }.copy()
    return outp

def get_default_structure_calibrator_options():
    return {
            'order_centroid': 2.75,
            'order_rms': 2.75,
            'fit_gap': True,
            'fit_order': False,
            'proj_cutoff': 2e-2,
            'gap_recon_precision': 1e-6,
            'gap_recon_delta': [-130e-6, 50e-6],
            }.copy()

default_n_particles = int(1e5)

tmp_elegant_dir = '~/tmp_elegant'

fontsize = 8

rho_label = r'$\rho$ (nC/m)'

