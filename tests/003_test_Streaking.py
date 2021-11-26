from scipy.constants import c
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.wf_model as wf_model
import PassiveWFMeasurement.beam_profile as beam_profile
import PassiveWFMeasurement.myplotstyle as ms
import WakefieldAnalysis.wf_model as wf_model_old

ms.closeall()


gap = 10e-3
beam_position = 4.7e-3
sig_t = 20e-15
tt_range = 200e-15
tt_points = 1e3
charge = 200e-12
energy_eV = 6e9

bp = beam_profile.get_gaussian_profile(sig_t, tt_range, tt_points, charge, energy_eV)

structure = wf_model.get_structure('SARUN18-UDCP020')

wake_dict = structure.convolve(bp, gap/2., beam_position, 'Dipole')

tt = wake_dict['wake_time']
wake = wake_dict['wake_potential']
spw = wake_dict['spw']


xx = bp.time * c
xx -= xx.min()
charge_profile = bp.charge_dist
calc = wf_model_old.WakeFieldCalculator(xx, charge_profile, energy_eV, 1)
wf_dict = calc.calc_all(gap/2., 1, beam_position)

tt_old = wf_dict['input']['charge_xx']/c
wake_old = wf_dict['dipole']['wake_potential']
spw_old = wf_dict['dipole']['single_particle_wake']


fig = ms.figure('Comparison of wakefields')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_current = subplot(sp_ctr, title='Current', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

bp.plot_standard(sp_current)

sp_spw = subplot(sp_ctr, title='Single particle wake', xlabel='t (fs)', ylabel='W (V/(m C))')
sp_ctr += 1

sp_w = subplot(sp_ctr, title='wake potential', xlabel='t (fs)', ylabel='W (V/m)')
sp_ctr += 1


for label, spw, wake_potential in [('New', spw, wake), ('Old', spw_old, wake_old)]:
    sp_spw.plot(tt, spw, label=label)
    sp_w.plot(tt, wake_potential, label=label)

sp_w.legend()
sp_spw.legend()



ms.show()

