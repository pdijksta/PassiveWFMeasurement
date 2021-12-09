import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)
from PassiveWFMeasurement import lattice

quad_k_values = [
        ('SATUN22.MQUA080', -4.352884290621318e-01),
        ('SATMA02.MQUA010', -1.171469785526617e+00),
        ('SATMA02.MQUA020', 4.007108540382343e+00),
        ('SATMA02.MQUA040', -8.509196941176438e-01),
        ('SATMA02.MQUA050', -4.279282251877575e+00),
        ('SATMA02.MQUA070', 3.057828766496955e+00),
        ]


quad_kl_dict = {a:b*0.08 for a, b in quad_k_values}

RP = 'SATBD01.MQUA010.START'
RP0 = 'SATUN22.MQUA080.START'
emit_optics = {
        'betax': 3.15,
        'alphax': 0.3,
        'betay': 3.15,
        'alphay': 0.3,
        }

lat_athos = lattice.Lattice('../elegant/Athos_Full.mat.h5')
#k1l_dict = {a: np.random.rand()-0.5 for a in lat_athos.quad_names}
k1l_dict = {a: -20 for a in lat_athos.quad_names}
k1l_dict.update(quad_kl_dict)
lat_athos.generate(k1l_dict)
optics_back = lat_athos.propagate_optics_dict(emit_optics, RP, RP0)

print(RP0)
print(optics_back)

