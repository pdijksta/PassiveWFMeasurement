import numpy as np
from scipy.constants import c
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)


import PassiveWFMeasurement.wf_model as wf_model
import PassiveWFMeasurement.beam_profile as beam_profile
import WakefieldAnalysis.wf_model as wf_model_old
import WakefieldAnalysis.image_and_profile as iap
import PassiveWFMeasurement.myplotstyle as ms

ms.closeall()


s_arr = np.linspace(0, 20e-6, 1000)
semigap = 5e-3
beam_position = 4.7e-3

spw_old = wf_model_old.wxq(s_arr, semigap, beam_position)
struct_new = wf_model.get_structure('SARUN18-UDCP020')

spw_new = struct_new.wxq(s_arr/c, semigap, beam_position)

ms.figure('Compare quad wake')
sp_ctr = 1

subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_wxq = subplot(sp_ctr, title='Wake function', xlabel='s ($\mu$m)', ylabel='wake function (MV/nC/m')
sp_ctr += 1

sp_wxq.plot(s_arr*1e6, spw_old/1e15, label='Old')
sp_wxq.plot(s_arr*1e6, spw_new/1e15, label='New')

sp_wxq.legend()

gauss_args_new = [20e-15, 400e-15, 2000, 180e-12, 6e9]
gauss_args_old = gauss_args_new[:]
gauss_args_old[1] /= 2
bp_new = beam_profile.get_gaussian_profile(*gauss_args_new)
bp_old = iap.get_gaussian_profile(*gauss_args_old)

wake_dict_old = bp_old.calc_wake(semigap*2, beam_position, 1)
wake_dict_new = bp_new.calc_wake(struct_new, semigap*2, beam_position, 'Quadrupole')

wake_potential_old = wake_dict_old['quadrupole']['wake_potential']
wake_potential_new = wake_dict_new['wake_potential']

sp_bp = subplot(sp_ctr, title='Profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

bp_old.plot_standard(sp_bp, label='Old')
bp_new.plot_standard(sp_bp, label='New')

sp_bp.legend()


sp_pot = subplot(sp_ctr, title='Wake potential', xlabel='t (fs)', ylabel='W [MV/m /$\mu$m]')
sp_ctr += 1

sp_pot.plot(bp_old.time-bp_old.time[0], wake_potential_old, label='Old')
sp_pot.plot(bp_new.time-bp_new.time[0], wake_potential_new, label='New')

sp_pot.legend()






ms.show()


