import itertools
import numpy as np

#Important naming convention
#    beam_position: Position of the beam relative to the middle of the structure
#    structure_position: Position of the structure as read by the :CENTER PV
#
#    The beam_position is calculated as follows:
#    -(structure_position - structure_position0)
#    The sign is opposite to the structure_position!
#
#    structure_position0: calibration

logfile = '/tmp/passiveWFMeasurement.log'

beamlines = [
        'Aramis',
        'Athos Pre-Undulator',
        'Athos Post-Undulator',
        'SASE2',
        'LCLS',
        ]

swissfel_beamlines = [
        'Aramis',
        'Athos Pre-Undulator',
        'Athos Post-Undulator',
        ]

structure_names = {
        'Aramis': [
             'SARUN18-UDCP020',
             'SARUN18-UDCP010',
             ],
        'Athos Post-Undulator': [
            'SATMA02-UDCP045',
            ],
        'Athos Pre-Undulator': [
            'SATDI01-UDCP100',
            'SATDI01-UDCP200',
            'SATCL02-UDCP100',
            'SATCL02-UDCP200',
            'SATCL02-UDCP300',
            'SATCL02-UDCP400',
            'SATMA01-UDCP015',
            ],
        'SASE2': [
            'SASE2',
            ],
        'LCLS': [
            'LCLS',
            ],
        }

screen_names = {
        'Aramis': [
            'SARBD02-DSCR050',
            'SARBD01-DSCR050',
            'simulation',
            ],
        'Athos Post-Undulator': [
            'SATBD02-DSCR050',
            'SATBD01-DSCR120',
            'simulation',
            ],
        'Athos Pre-Undulator': [
            'SATMA01-DSCR030',
            'SATBD02-DSCR050',
            'SATBD01-DSCR120',
            'simulation',
            ],
        'SASE2': [
            'SASE2',
            ],
        'LCLS': [
            'LCLS',
            ],
        }

aramis_structure_parameters = {
        'Ls': 1.,
        'p': 500e-6,
        'g': 250e-6,
        'w': 40e-3,
        }

xfel_structure_parameters = {
        'Ls': 5.,
        'p': 500e-6,
        'g': 250e-6,
        'w': 40e-3,
        }

lcls_structure_parameters = {
        'Ls': 2.,
        'p': 500e-6,
        'g': 250e-6,
        'w': 40e-3,
        }

structure_parameters = {
        'SARUN18-UDCP010': aramis_structure_parameters,
        'SARUN18-UDCP020': aramis_structure_parameters,
        'SATMA01-UDCP015': aramis_structure_parameters,
        'SATMA02-UDCP045': aramis_structure_parameters,
        'SATDI01-UDCP100': aramis_structure_parameters,
        'SATDI01-UDCP200': aramis_structure_parameters,
        'SATCL02-UDCP100': aramis_structure_parameters,
        'SATCL02-UDCP200': aramis_structure_parameters,
        'SATCL02-UDCP300': aramis_structure_parameters,
        'SATCL02-UDCP400': aramis_structure_parameters,
        'SASE2': xfel_structure_parameters,
        'LCLS': lcls_structure_parameters,
        }

structure_dimensions = {
        'SARUN18-UDCP010': 'X',
        'SARUN18-UDCP020': 'X',
        'SATMA01-UDCP015': 'Y',
        'SATMA02-UDCP045': 'X',
        'SATDI01-UDCP100': 'Y',
        'SATDI01-UDCP200': 'X',
        'SATCL02-UDCP100': 'Y',
        'SATCL02-UDCP200': 'X',
        'SATCL02-UDCP300': 'Y',
        'SATCL02-UDCP400': 'X',
        'SASE2': 'Y',
        'LCLS': 'X',
        }

beamline_quads = {
        'Aramis': [
            'SARUN13-MQUA080',
            'SARUN14-MQUA080',
            'SARUN15-MQUA080',
            'SARUN16-MQUA080',
            'SARUN17-MQUA080',
            'SARUN18-MQUA080',
            'SARUN19-MQUA080',
            'SARUN20-MQUA080',
            'SARBD01-MQUA020',
            'SARBD02-MQUA030',
            ],
        'Athos Post-Undulator': [
            'SATUN22-MQUA080',
            'SATMA02-MQUA010',
            'SATMA02-MQUA020',
            'SATMA02-MQUA040',
            'SATMA02-MQUA050',
            'SATMA02-MQUA070',
            'SATBD01-MQUA010',
            'SATBD01-MQUA030',
            'SATBD01-MQUA050',
            'SATBD01-MQUA070',
            'SATBD01-MQUA090',
            'SATBD02-MQUA030',
            ],
        'Athos Pre-Undulator': [
            'SATDI01-MQUA025',
            'SATDI01-MQUA080',
            'SATDI01-MQUA220',
            'SATDI01-MQUA230',
            'SATDI01-MQUA250',
            'SATDI01-MQUA260',
            'SATDI01-MQUA280',
            'SATDI01-MQUA300',
            'SATCB01-MQUA230',
            'SATCB01-MQUA430',
            'SATCL02-MQUA230',
            'SATCL02-MQUA430',
            ],
        'SASE2': [],
        'LCLS': [],
        }

beamline_undulators = {
        'Aramis': ['SARUN%02i-UIND030:K_SET' % x for x in range(3, 16)],
        'Athos Post-Undulator': ['SATUN%02i-UIND030:K_SET' % x for x in list(range(6, 14))+list(range(15, 22))],
        'Athos Pre-Undulator': [],
        'SASE2': [],
        'LCLS': [],
        }

beamline_chargepv = {
        'Aramis': 'SINEG01-DICT215:B1_CHARGE-OP',
        'Athos Post-Undulator': 'SINEG01-DICT215:B2_CHARGE-OP',
        'Athos Pre-Undulator': 'SINEG01-DICT215:B2_CHARGE-OP',
        'SASE2': 'SASE2:CHARGE',
        'LCLS': 'LCLS:CHARGE',
        }

beamline_energypv = {
        'Aramis': 'SARBD01-MBND100:ENERGY-OP',
        'Athos Post-Undulator': 'SATBD01-MBND200:ENERGY-OP',
        'Athos Pre-Undulator': 'SATBD01-MBND200:ENERGY-OP',
        'SASE2': 'SASE2:ENERGY',
        'LCLS': 'LCLS:ENERGY',
        }

gas_monitor_pvs = {
        'Aramis': 'SARFE10-PBPG050:PHOTON-ENERGY-PER-PULSE-AVG',
        'Athos Post-Undulator': 'SATFE10-PEPG046:PHOTON-ENERGY-PER-PULSE-AVG',
        'Athos Pre-Undulator': 'SATFE10-PEPG046:PHOTON-ENERGY-PER-PULSE-AVG',
        'SASE2': 'SASE2:FEL-ENERGY',
        'LCLS': 'LCLS:FEL-ENERGY',
        }

default_optics = {
        'Aramis': {
            'betax': 4.968,
            'alphax': -0.563,
            'betay': 16.807,
            'alphay': 1.782,
            },
        'Athos Post-Undulator': {
            'betax': 9.538,
            'alphax': -2.702,
            'betay': 2.857,
            'alphay': 0.819,
            },
        'Athos Pre-Undulator': { # From Eduard / Alexander
            'betax': 35.76124,
            'alphax': -1.165823,
            'betay': 24.08687,
            'alphay': 0.6483776,
            },
        'SASE2': {
            'betax': 50,
            'alphax': 0,
            'betay': 50,
            'alphay': 0,
            },
        'LCLS': {
            'betax': 4.502,
            'alphax': -0.02,
            'betay': 23.541,
            'alphay': -1.559,
            },
        }

optics_matching_points = {
        'Aramis': 'SARUN18.START',
        'Athos Post-Undulator': 'SATUN22.MQUA080.START',
        'Athos Pre-Undulator': 'SATDI01.MQUA250.START',
        'SASE2': None,
        'LCLS': None,
        }

custom_optics = {
        'Aramis': {
            'betax': 4.9,
            'alphax': 0.564,
            'betay': 15.71,
            'alphay': -1.731,
            },
        }
for _key in ['Athos Post-Undulator', 'Athos Pre-Undulator']:
    custom_optics[_key] = default_optics[_key]
del _key

custom_optics_matching_points = {
        'Aramis': 'SARUN15.MQUA080.START',
        'Athos Post-Undulator': 'SATUN22.MQUA080.START',
        'Athos Pre-Undulator': 'SATDI01.MQUA250.START',
        }

beamline_lat_files = {
        'Aramis': './elegant/Aramis.mat.h5',
        'Athos Post-Undulator': './elegant/Athos_Full.mat.h5',
        'Athos Pre-Undulator': './elegant/Athos_Full.mat.h5',
        }

def get_default_optics(beamline):
    return default_optics[beamline].copy()

def get_custom_optics(beamline):
    return custom_optics[beamline].copy()

_aramis_pvs = ['SARUN%02i-DBPM070:%s1' % (i, dim) for i, dim in itertools.product(range(1, 21), ('X', 'Y'))]
_aramis_pvs += ['SARBD01-DBPM040:%s1' % dim for dim in ('X', 'Y')]
_aramis_pvs += ['SARBD02-DBPM010:%s1' % dim for dim in ('X', 'Y')]

_pre_athos_pvs = ['SATDI01-DBPM%03i:%s1' % (num, dim) for num, dim in itertools.product([90, 210, 240, 270, 310], ['X', 'Y'])]
_pre_athos_pvs += ['SATCB01-DBPM%03i:%s1' % (num, dim) for num, dim in itertools.product([220, 420], ['X', 'Y'])]
_pre_athos_pvs += ['SATCL02-DBPM%03i:%s1' % (num, dim) for num, dim in itertools.product([220, 420], ['X', 'Y'])]
_pre_athos_pvs += ['SATMA01-DBPM%03i:%s1' % (num, dim) for num, dim in itertools.product([10, 20, 40, 110, 130, 150, 170, 240, 620], ['X', 'Y'])]

beamline_bpm_pvs = {
        'Aramis': _aramis_pvs,
        'Athos Pre-Undulator': _pre_athos_pvs,
        }

beamline_charge_pvs_bsread = {
        'Aramis': ['SARBD01-DICT030:B1_CHARGE', 'SINEG01-DICT215:B1_CHARGE'],
        }

all_structures = []
for beamline, structure_list in structure_names.items():
    all_structures.extend(structure_list)
del beamline, structure_list

default_n_particles = int(1e5)

def get_default_forward_options():
    return {
            'method': 'thicklens', # can also be thinlens or ocelot
            'ocelot_wake_step': 0.2,
            'ocelot_unit_step': 0.2,
            'ocelot_wake_sampling': 500,
            'screen_bins': 1000,
            'screen_smoothen': 20e-6,
            'dipole_wake': True,
            'quad_wake': False,
            'long_wake': False,
            'long_wake_correction': False,
            'screen_cutoff': 2e-2,
            'len_screen': 5000,
            'n_particles': default_n_particles,
            }.copy()

def get_default_backward_options():
    return {
            'compensate_negative_screen': True,
            'compensate_rms_factor': 0.4,
            'len_profile': 5000,
            'profile_cutoff': 0.1e-2,
            'profile_smoothen': 1e-15,
            }.copy()

def get_default_reconstruct_gauss_options():
    return {
            'gauss_profile_t_range': 400e-15,
            'precision': 0.1e-15,
            'method': 'centroid',
            'sig_t_range': np.array([7, 100])*1e-15,
            'max_iterations': 5,
            'init_func': 'gauss',
            }.copy()

def get_default_beam_spec():
    outp = {
            'nemitx': 250e-9,
            'nemity': 250e-9,
            'n_mesh': 500,
            'cutoff_sigma': 5,
            'energy_chirp': 0,
            'energy_spread': 0,
            }.copy()
    return outp

def get_default_structure_calibrator_options():
    outp = {
            'order_centroid': 2.75,
            'order_rms': 2.75,
            'fit_gap': True,
            'fit_order': False,
            'proj_cutoff': 2e-2,
            'delta_gap_range': np.linspace(-80, 80.01, 51)*1e-6,
            'delta_structure0_range': np.linspace(-10, 10, 21)*1e-6,
            'delta_gap_scan_n': 10,
            }.copy()
    return outp

def get_default_lasing_options():
    outp = {
            'subtract_quantile': 0.1,
            'subtract_absolute': None,
            'max_quantile': None,
            'max_absolute': None,
            'slice_factor': 3,
            'current_cutoff': 0.5e3,
            'x_conversion': 'wake',
            'x_linear_factor': 1,
            'adjust_linear_factor': False,
            'rms_sigma': 5,
            't_lims': None,
            'ref_t': None,
            'E_lims': None,
            'slice_method': 'full',
            'plot_slice_analysis': False,
            'plot_slice_analysis_save_path': None,
            'adjust_beam_position': True,
            'adjust_beam_position_backward': False,
            'self_consistent_profile': False,
            'cut_extremes': None,
            'void_cutoff': [1e-3, 1e-3],
            'x_to_t_time_factor': 5,
            'x_to_t_x_factor': 3,
            }.copy()
    return outp

def get_default_find_beam_position_options():
    outp = {
            'position_explore': 30e-6,
            'method': 'centroid',
            'precision': .5e-6,
            'max_iterations': 8,
            }.copy()
    return outp

def get_default_data_loader_options():
    outp = {
            'subtract_quantile': 0.1,
            'subtract_absolute': None,
            'max_intensity': None,
            'void_cutoff': [1e-3, 1e-3],
            'cutX': None,
            'cutY': None,
            'multi_zero_position': 0,
            'zero_positions': None,
            'screen_cutoff': 0,
            'screen_cutoff_relative': False,
            'screen_cutoff_edge_points': 10,
            'screen_cutoff_relative_factor': 2,
            }.copy()
    return outp

def get_blmeas_data_loader_options():
    outp = get_default_data_loader_options()
    outp.update({
            'subtract_quantile': 0.5,
            'subtract_absolute': None,
            'void_cutoff': [None, None],
            'cutX': None,
            'cutY': None,
            'screen_cutoff': 0,
            'screen_cutoff_relative': True,
            'screen_cutoff_edge_points': 10,
            'screen_cutoff_relative_factor': 2,
            })
    return outp

xfel_charge_pv = 'XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL'

xfel_energy_pv= {
        'preSA1': 'XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL',
        'postSA2': 'XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL',
        }


sase2_zpos_dict = {
        'CMY.2443.T3': 2443.4426,
        'BPMA.2455.T3': 2455.3481,
        'CMY.2456.T3': 2456.812,
        'BPMA.2467.T3': 2467.948,
        'CEY.2468.T3': 2468.5825,
        'BPMA.2487.T3': 2487.627,
        'CMY.2488.T3': 2488.7915,
        }

default_deltaK = 0.2

# For plots
fontsize = 8
rho_label = r'$\rho$ (nC/m)'

init_plate_pos_dict = {
        'SASE2': 3.55e-3,
        }

