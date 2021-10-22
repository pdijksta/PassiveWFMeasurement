import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)


import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.beam as beam


beamline = 'Aramis'
screen_name = 'SARBD02-DSCR050'
structure_name = 'SARUN18-UDCP020'
file_ = './data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5'
dict_ = h5_storage.loadH5Recursive(file_)
meta_data = dict_['meta_data_begin']

delta_gap = -70e-6
structure_center = 360e-6
screen_center = 1044e-6


tracker = tracking.Tracker('Aramis', screen_name, structure_name, meta_data, delta_gap, structure_center, screen_center)

#beam_obj = beam.beam_from_spec('x'

tracker.forward_propagate()


