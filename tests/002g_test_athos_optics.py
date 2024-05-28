import itertools
import numpy as np
from ElegantWrapper.simulation import ElegantSimulation
from PassiveWFMeasurement import lattice

def loadSnap(filename):
    res={}
    with open(filename,'r') as fid:
        lines=fid.readlines()
        for line in lines[1:]:
            split = line.split(',')
            pv = split[0].strip()
            val = split[1].split('"val":')[1].split('}')[0]
            if '.' in val:
                res[pv]=float(val)
            else:
                res[pv]= val
    return res

snapshot = loadSnap('./SF_settings_20240528_113951.snap')
lat = lattice.Lattice('./Athos_ref0.mat.h5')

quad_names = lat.quad_names
quad_k1l_dict = {}
for quad_name0 in quad_names:
    if 'MQUP' in quad_name0:
        k1l = 0
    else:
        quad_name = quad_name0.replace('.Q', '')
        ch = quad_name.replace('.', '-')+':K1L-SET'
        k1l = snapshot[ch]
    quad_k1l_dict[quad_name0] = k1l

lat.generate(quad_k1l_dict, assert0=True)
sim = ElegantSimulation('./elegant_test/Athos.ele')

mat = sim.mat

def rot_mtx(angle):
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([[cs, 0., sn, 0., 0., 0.],
                     [0., cs, 0., sn, 0., 0.],
                     [-sn, 0., cs, 0., 0., 0.],
                     [0., -sn, 0., cs, 0., 0.],
                     [0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 1.]])






for index, (name, type_) in enumerate(zip(mat['ElementName'], mat['ElementType'])):
    if type_ == 'QUAD':
        mat_ele66 = np.zeros([6, 6])
        for i1, i2 in itertools.product(range(6), repeat=2):
            mat_ele66[i1,i2] = mat['R%i%i' % (i1+1,i2+1)][index]

        index = list(lat.element_names).index(name)
        mat_lat66 = lat.single_matrices[index]
        mat_lat = mat_lat66[:4,:4]
        mat_ele = mat_ele66[:4,:4]


        #print(name)
        #print(mat_ele)
        #print(mat_lat)
        #print()
        #mat_ele_rot = (rot_mtx(-np.pi/4) @ mat_ele66 @ rot_mtx(np.pi/4))[:4,:4]
        #mat_lat_rot = (rot_mtx(-np.pi/4) @ mat_lat66 @ rot_mtx(np.pi/4))[:4,:4]
        #print(mat_ele_rot)
        #print(mat_lat_rot)

        if 'MQSK' in name:
            k1l = lat.quad_k1l_dict[name]
            k1 = lat.quad_k1_dict[name]
            if k1 == 0:
                continue
            length = k1l/k1
            mat2 = lattice.transferMatrixQuad66(length, k1)
            mat_rot = (rot_mtx(-np.pi/4) @ mat2 @ rot_mtx(np.pi/4))[:4,:4]

            equal = not np.any(np.abs(mat_ele-mat_lat) > 1e-3)
            if not equal:
                print(name)
                print(mat_ele)
                print(mat_lat)
                print(mat_rot)
                import pdb; pdb.set_trace()
        #print(mat_ele[:4,:4])
        #print(mat_lat[:4,:4])

