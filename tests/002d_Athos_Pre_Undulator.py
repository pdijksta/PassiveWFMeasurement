import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)
from PassiveWFMeasurement import lattice

quad_k_values = [
        ('SATDI01.MQUA250', -7.487880524702719e-01),
        ('SATDI01.MQUA260', 1.437119676428307e+00),
        ('SATDI01.MQUA280', 1.888304214946000e-01),
        ('SATDI01.MQUA300', -8.016960474273966e-01),
        ]

quad_kl_dict = {a:b*0.15 for a, b in quad_k_values}

RP = 'SATCB01.MQUA230.START'
RP0 = 'SATDI01.MQUA250.START'
emit_optics = {
        'betax': 14.5804,
        'alphax': 0.75,
        'betay': 14.5804,
        'alphay': 0.75,
        }

lat_athos0 = lattice.get_beamline_lattice('Athos Pre-Undulator', quad_kl_dict)
optics_back0 = lat_athos0.propagate_optics_dict(emit_optics, RP, RP0)

lat_athos = lattice.Lattice('../elegant/Athos_Full.mat.h5')
lat_athos.generate(quad_kl_dict, True)
optics_back = lat_athos.propagate_optics_dict(emit_optics, RP, RP0)

print(RP0)
print(optics_back)

