import numpy as np
from scipy.constants import c

from . import lattice

from ocelot.cpbd.wake3D import WakeTableDechirperOffAxis, Wake
from ocelot.cpbd.elements import Drift, Marker
from ocelot.cpbd.optics import SecondTM, Navigator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import ParticleArray
from ocelot.cpbd.track import track

def forward_propagate_ocelot(tracker, beam, plot_details=False, output_details=False):
    self = tracker
    wake_step = self.forward_options['ocelot_wake_step']
    unit_step = self.forward_options['ocelot_unit_step']
    wake_sampling = self.forward_options['ocelot_wake_sampling']

    # OCELOT needs beam before the structure, not at the center
    drift_neg = lattice.transferMatrixDrift66(-0.5*self.structure.Ls)
    beam_before_streaker = beam.linear_propagate(drift_neg)

    # Init waketable
    gap = self.structure_gap
    beam_position = self.beam_position
    b = gap/2. - beam_position
    a = gap/2.
    width = self.structure.w
    t = self.structure.g
    p = self.structure.p
    length = self.structure.Ls
    sigma = beam.beamProfile.rms() * c # around 5 um
    # Ocelot uses orientation of corrugated plate. This code uses streaking direction
    orient = {'X': 'vert', 'Y': 'horz'}[self.structure.dim]

    wake_table = WakeTableDechirperOffAxis(b, a, width, t, p, length, sigma, orient)
    wake = Wake()
    wake.w_sampling = wake_sampling
    wake.wake_table = wake_table
    wake.step = wake_step

    # Generate OCELOT lattice
    w_start = Marker()
    w_stop= Marker()
    D01m = Drift(l=self.structure.Ls)

    lat_list = (w_start, D01m, w_stop)
    lat = MagneticLattice(lat_list, method={"global": SecondTM})

    rparticles = np.zeros([6, self.n_particles], float)
    rparticles[0] = beam_before_streaker['x'] #+ tracker.beam_position
    rparticles[1] = beam_before_streaker['xp']
    rparticles[2] = beam_before_streaker['y']
    rparticles[3] = beam_before_streaker['yp']
    rparticles[4] = beam_before_streaker['t']*(c)
    rparticles[5] = 0

    particle_array = ParticleArray()
    particle_array.rparticles = rparticles
    particle_array.s = 1
    particle_array.E = self.energy_eV/1e9
    particle_array.q_array = np.ones(self.n_particles) * self.total_charge / self.n_particles

    # Add wake effect to lattice
    navi = Navigator(lat)
    navi.add_physics_proc(wake, w_start, w_stop)
    navi.unit_step = unit_step

    # Forward track with OCELOT
    _, pa = track(lat, particle_array, navi)
    beam_arr = pa.rparticles
    beam_arr[4] /= c
    beam0 = beam.child(beam_arr)
    beam0.dim_index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3, 't': 4, 'delta': 5}
    beam_after_streaker = beam0.linear_propagate(drift_neg)

    beam_at_screen = beam_after_streaker.linear_propagate(self.matrix)
    screen = self._beam2screen(beam_at_screen)
    outp_dict = {'screen': screen}
    if output_details:
        outp_dict.update({
            'beam': beam,
            'beam_after_streaker': beam_after_streaker,
            'beam_at_screen': beam_at_screen,
            'transport_matrix': self.matrix,
            })

    return outp_dict

