import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.resolution as resolution
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.tracking as tracking
import PassiveWFMeasurement.h5_storage as h5_storage
import PassiveWFMeasurement.calibration as calibration
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()

sig_t = 30e-15
tt_range = 400e-15
tt_points = 1000
total_charge = 200e-12
energy_eV = 6e9
gap = 10e-3
beam_offset = 4.7e-3
beamline = 'Aramis'
structure_name = 'SARUN18-UDCP020'

data = h5_storage.loadH5Recursive('./data/2021_05_19-14_59_24_Lasing_True_SARBD02-DSCR050.h5')
meta_data = data['meta_data_begin']
calib = calibration.StructureCalibration(structure_name, 900e-6, 0, 0)

tracker = tracking.get_default_tracker(beamline, structure_name, meta_data, calib)
tracker.forward_options['quaad_wake'] = True
bp = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, total_charge, energy_eV)
res_dict = resolution.calc_resolution(bp, gap, beam_offset, tracker)


ms.figure('Test resolution')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_current = subplot(sp_ctr, title='Current profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1
sp_res = subplot(sp_ctr, title='Resolution', xlabel='t (fs)', ylabel='R (fs)')
sp_ctr += 1

resolution.plot_resolution(res_dict, sp_current, sp_res)


ms.show()
