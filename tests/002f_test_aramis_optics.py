import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

from PassiveWFMeasurement import optics
from PassiveWFMeasurement import lattice
from PassiveWFMeasurement import config

def calc_phase_advance(r11, r12, beta0, alpha0):
    gamma0 = (1+alpha0**2)/beta0
    beta1 = r11**2 * beta0 - 2*r11*r12*alpha0 + r12**2 * gamma0
    sin_mu = r12 / np.sqrt(beta1 * beta0)
    cos_mu = r11 * np.sqrt(beta0/beta1) - alpha0 * sin_mu
    mu = np.arctan2(sin_mu, cos_mu)
    return mu / np.pi * 180

lat = lattice.Lattice('../elegant/Aramis.mat.h5')
optics_sar15_mqua80 = {
        'betax': 4.9,
        'alphax': 0.564,
        'betay': 15.71,
        'alphay': -1.731,
        }
sar15_mqua80 = 'SARUN15.MQUA080.START'
structure = 'SARUN18.UDCP020'
screen = 'SARBD02.DSCR050'

aramis_info = (lat, optics_sar15_mqua80, sar15_mqua80, structure, screen, optics.aramis_optics, optics.aramis_quads)

lat = lattice.Lattice('../elegant/Athos_Full.mat.h5')
optics_sat22_mqua80 = {
        'betax': 9.538,
        'alphax': -2.702,
        'betay': 2.857,
        'alphay': 0.819,
        }
sat22_mqua80 = 'SATUN22.MQUA080.START'
structure = 'SATMA02.UDCP045'
screen = 'SATBD02.DSCR050'

athos_post_info = (lat, optics_sat22_mqua80, sat22_mqua80, structure, screen, optics.athos_post_undulator_optics, optics.athos_post_undulator_quads)

lat = lattice.Lattice('../elegant/Athos_Full.mat.h5')
optics_satdi01_mqua250 = config.default_optics['Athos Pre-Undulator']
satdi01_mqua250= 'SATDI01.MQUA250.START'
structure = 'SATCL02.UDCP100', 'SATCL02.UDCP200'
screen = 'SATMA01.DSCR030'

athos_pre_info = (lat, optics_satdi01_mqua250, satdi01_mqua250, structure, screen, optics.athos_pre_undulator_optics, optics.athos_pre_undulator_quads)

for beamline, (lat, optics0, pos0, _structure, screen, optics_info, quads), type_ in [
        ('Aramis', aramis_info, 0),
        ('Athos Post-Undulator', athos_post_info, 0),
        ('Athos Pre-Undulator', athos_pre_info, 1),
        ]:
    print('\n\n%s\n' % beamline)

    for struct in optics_info:
        if type_ == 0:
            identifier, structure_beta, structure_alpha, screen_beta, r12, r11, phase_advance, k1ls = struct
            dims = ('x',)
        else:
            identifier, structure_beta, structure_alpha, structure_betay, structure_alphay, screen_beta, screen_betay, r11, r12, r33, r34, phase_advance, phase_advance_y, k1ls = struct
            dims = ('x', 'y')
        for dim_ctr, dim in enumerate(dims):
            if type_ == 1:
                structure = _structure[dim_ctr]
            else:
                structure = _structure
            k1l_dict = {a: b for a, b in zip(quads, k1ls)}
            lat.generate(k1l_dict)
            optics_structure = lat.propagate_optics_dict(optics0, pos0, structure)
            optics_screen = lat.propagate_optics_dict(optics0, pos0, screen)
            mat_struct_screen = lat.get_matrix(structure, screen)
            if dim == 'x':
                phase_advance2 = calc_phase_advance(mat_struct_screen[0,0], mat_struct_screen[0,1], optics_structure['betax'], optics_structure['alphax'])
                s_beta = structure_beta
                s_alpha = structure_alpha
                sc_beta = screen_beta
                sc_beta2 = optics_screen['betax']
                sc_beta3 = optics_screen['betay']
                phase_advance0 = phase_advance
            else:
                phase_advance2 = calc_phase_advance(mat_struct_screen[2,2], mat_struct_screen[2,3], optics_structure['betay'], optics_structure['alphay'])
                s_beta = structure_betay
                s_alpha = structure_alphay
                sc_beta = screen_betay
                sc_beta2 = optics_screen['betay']
                sc_beta3 = optics_screen['betax']
                phase_advance0 = phase_advance_y
            s_gamma = (1+s_alpha**2)/s_beta
            o_beta = optics_structure['beta'+dim]
            o_alpha = optics_structure['alpha'+dim]
            o_gamma = (1+o_alpha**2)/o_beta
            mm = 0.5*(s_gamma*o_beta - 2*s_alpha*o_alpha + s_beta*o_gamma)
            print('%s%s at structure: Beta %.2f / %.2f ; Alpha %.2f / %.2f ; mm %.2f' % (identifier, dim, s_beta, o_beta, s_alpha, o_alpha, mm))
            print('%s%s at screen: Beta %.2f / %.2f ; Other dim: %.2f' % (identifier, dim, sc_beta, sc_beta2, sc_beta3))
            print('%s%s R11 %.2f / %.2f ; R12 %.2f / %.2f ; phi %.2f / %.2f' % (identifier, dim, r11, mat_struct_screen[0,0], r12, mat_struct_screen[0,1], phase_advance0, phase_advance2))

