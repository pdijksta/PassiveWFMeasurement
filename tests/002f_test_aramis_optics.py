import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

from PassiveWFMeasurement import optics
from PassiveWFMeasurement import lattice


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


for identifier, structure_beta, structure_alpha, screen_beta, r12, r11, phase_advance, k1ls in optics.all_optics:
    k1l_dict = {a: b for a, b in zip(optics.quadNames, k1ls)}
    lat.generate(k1l_dict)
    optics_structure = lat.propagate_optics_dict(optics_sar15_mqua80, sar15_mqua80, structure)
    optics_screen = lat.propagate_optics_dict(optics_sar15_mqua80, sar15_mqua80, screen)
    print('%s at structure: Beta %.2f / %.2f ; Alpha %.2f / %.2f' % (identifier, structure_beta, optics_structure['betax'], structure_alpha, optics_structure['alphax']))
    print('%s at screen: Beta %.2f / %.2f' % (identifier, screen_beta, optics_screen['betax']))



