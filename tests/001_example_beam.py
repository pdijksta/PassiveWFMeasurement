import numpy as np
import argparse
import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.beam as beam
import PassiveWFMeasurement.beam_profile as beam_profile
import WakefieldAnalysis.elegant_matrix as elegant_matrix

import myplotstyle as ms

parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--savefig', type=str)
args = parser.parse_args()



ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

nemitx = nemity = 300e-9
betax = 20
betay = 30
alphax = -0.5
alphay = 0.5
energy_eV = 6e9
p_central = energy_eV/511e3
n_particles = int(1e5)
cutoff_sigma = 5
n_mesh = 500
sigma_t = 10e-15
tt_range = 200e-15
tt_points = 1000
charge = -200e-12

gauss_bp = beam_profile.get_gaussian_profile(sigma_t, tt_range, tt_points, charge, energy_eV)
beam = beam.gen_beam6D(nemitx, nemity, betax, alphax, betay, alphay, energy_eV, gauss_bp, n_particles, cutoff_sigma, n_mesh)
beam_old_w = elegant_matrix.gen_beam(nemitx, nemity, alphax, betax, alphay, betay, p_central, sigma_t, n_particles)[0]
beam_old = np.zeros_like(beam)

beam_old[0] = beam_old_w['x']
beam_old[1] = beam_old_w['xp']
beam_old[2] = beam_old_w['y']
beam_old[3] = beam_old_w['yp']
beam_old[4] = beam_old_w['dt']
beam_old[5] = beam_old_w['p']

for dim in range(6):
    for b, key in [(beam_old, 'Elegant'), (beam, 'Python')]:
        print(dim, key, b[dim].mean(), b[dim].std())


ms.figure('', figsize=(8,7))
subplot = ms.subplot_factory(2,2, grid=False)

sp_ctr = 1

sps = []
for b, title in [(beam, 'Python'), (beam_old, 'Elegant')]:

    sp = subplot(sp_ctr, grid=False, title=title, xlabel='x ($\mu$m)', ylabel='x\' ($\mu$rad)')
    sp_ctr += 1
    sps.append(sp)

    sp.hist2d(b[0]*1e6, b[1]*1e6, bins=(100,100))

xlims = [max(x.get_xlim()[0] for x in sps), min(x.get_xlim()[1] for x in sps)]
ylims = [max(x.get_ylim()[0] for x in sps), min(x.get_ylim()[1] for x in sps)]
for sp in sps:
    sp.set_xlim(*xlims)
    sp.set_ylim(*ylims)


sp_x = subplot(sp_ctr, grid=False, title='Horizontal profile', xlabel='x ($\mu$m)', ylabel=r'$\rho$ (pC/$\mu$m)')
sp_ctr += 1


sp_t = subplot(sp_ctr, grid=False, title='Time profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1


bin_edges_dict = {}
for b, title in [(beam, 'Python'), (beam_old, 'Elegant')]:

    for index, sp, x_factor, y_factor in [(0, sp_x, 1e6, 1e12/1e6*abs(charge)), (4, sp_t, 1e15, abs(charge)/1e3)]:
        tt = b[index] - b[index].mean()

        if title == 'Python':
            bins = 200
        else:
            bins = bin_edges_dict[index]
        hist, bin_edges = np.histogram(tt, bins, density=True)
        if title == 'Python':
            bin_edges_dict[index] = bin_edges
        sp.plot(bin_edges[:-1]*x_factor, hist*y_factor, label=title)

sp.legend(loc='upper right')

hspace, wspace = 0.40, 0.35
if args.savefig:
    ms.saveall(args.savefig, hspace, wspace, ending='.png')

if not args.noshow:
    ms.show()

