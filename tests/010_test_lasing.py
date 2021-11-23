import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.lasing as lasing
import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.config as config
import PassiveWFMeasurement.plot_results as plot_results
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

lasing_on_dict = h5_storage.loadH5Recursive('./data/2021_10_18-15_13_42_Lasing_True_SARBD02-DSCR050.h5')
lasing_off_dict = h5_storage.loadH5Recursive('./data/2021_10_18-15_11_55_Lasing_False_SARBD02-DSCR050.h5')

structure_name = 'SARUN18-UDCP020'
screen_center = 1046e-6
delta_gap = -45e-6
structure_position0 = 361e-6
pulse_energy = 180e-6

lasing_options = config.get_default_lasing_options()
calib = calibration.StructureCalibration(structure_name, screen_center, delta_gap, structure_position0)

tracker = tracking.get_default_tracker('Aramis', structure_name, lasing_on_dict['meta_data_begin'], calib)
las_rec_images = {}

for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
    rec_obj = lasing.LasingReconstructionImages(tracker, lasing_options)
    rec_obj.add_dict(data_dict)
    if main_ctr == 1:
        rec_obj.profile = las_rec_images['Lasing Off'].profile
        rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
    rec_obj.process_data()
    las_rec_images[title] = rec_obj
    #rec_obj.plot_images('raw', title)
    #rec_obj.plot_images('tE', title)

las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=0.5e3)
plot_results.plot_lasing(las_rec.get_result_dict())

ms.show()

