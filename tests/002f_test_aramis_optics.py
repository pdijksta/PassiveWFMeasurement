import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

from PassiveWFMeasurement import optics
from PassiveWFMeasurement import lattice

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


for beamline, (lat, optics0, pos0, structure, screen, optics_info, quads) in [
        ('Aramis', aramis_info),
        ('Athos Post-Undulator', athos_post_info),
        ]:
    print('\n\n%s\n' % beamline)

    for identifier, structure_beta, structure_alpha, screen_beta, r12, r11, phase_advance, k1ls in optics_info:
        k1l_dict = {a: b for a, b in zip(quads, k1ls)}
        lat.generate(k1l_dict)
        optics_structure = lat.propagate_optics_dict(optics0, pos0, structure)
        optics_screen = lat.propagate_optics_dict(optics0, pos0, screen)
        mat_struct_screen = lat.get_matrix(structure, screen)
        phase_advance2 = calc_phase_advance(mat_struct_screen[0,0], mat_struct_screen[0,1], optics_structure['betax'], optics_structure['alphax'])
        print('%s at structure: Beta %.2f / %.2f ; Alpha %.2f / %.2f' % (identifier, structure_beta, optics_structure['betax'], structure_alpha, optics_structure['alphax']))
        print('%s at screen: Beta %.2f / %.2f' % (identifier, screen_beta, optics_screen['betax']))
        print('%s R11 %.2f / %.2f ; R12 %.2f / %.2f ; phi %.2f / %.2f' % (identifier, r11, mat_struct_screen[0,0], r12, mat_struct_screen[0,1], phase_advance, phase_advance2))



