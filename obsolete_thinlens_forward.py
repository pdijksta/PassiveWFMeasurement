import numpy as np

from . import myplotstyle as ms

def forward_propagate_thinlens(self, beam, plot_details=False, output_details=False):
    """
    beam: must correspond to middle of structure
    """

    wake_time = beam.beamProfile.time
    energy_eV = beam.energy_eV
    wake_dict_dipole = self.calc_wake(beam.beamProfile, 'Dipole')
    delta_xp_dipole = wake_dict_dipole['wake_potential']/energy_eV
    delta_xp_coords_dip = np.interp(beam['t'], wake_time, delta_xp_dipole)
    quad_wake = self.forward_options['quad_wake']
    long_wake = self.forward_options['long_wake']
    dim = self.structure.dim.lower()
    other_dim = 'x' if dim == 'y' else 'y'

    if quad_wake:
        wake_dict_quadrupole = self.calc_wake(beam.beamProfile, 'Quadrupole')
        delta_xp_quadrupole = wake_dict_quadrupole['wake_potential']/energy_eV
        delta_xp_interp = np.interp(beam['t'], wake_time, delta_xp_quadrupole)
        delta_xp_coords_quad = delta_xp_interp*(beam[dim]-beam[dim].mean())
        if other_dim in beam.dim_index:
            delta_yp_coords_quad = -delta_xp_interp*(beam[other_dim]-beam[other_dim].mean())
    else:
        wake_dict_quadrupole = None
        delta_xp_quadrupole = 0.
        delta_xp_coords_quad = 0.

    beam_after_streaker = beam.child()

    if long_wake:
        wake_dict_long = self.calc_wake(beam.beamProfile, 'Longitudinal')
        delta_p = wake_dict_long['wake_potential']/energy_eV
        delta_p_interp = np.interp(beam['t'], wake_time, delta_p)
        beam_after_streaker['delta'] += delta_p_interp

    beam_after_streaker[dim+'p'] += delta_xp_coords_dip
    beam_after_streaker[dim+'p'] += delta_xp_coords_quad
    if quad_wake and other_dim in beam.dim_index:
        beam_after_streaker[other_dim+'p'] += delta_yp_coords_quad

    beam_at_screen = beam_after_streaker.linear_propagate(self.matrix)
    screen = self._beam2screen(beam_at_screen)
    outp_dict = {'screen': screen}
    if output_details:
        outp_dict.update({
            'beam': beam,
            'beam_after_streaker': beam_after_streaker,
            'beam_at_screen': beam_at_screen,
            'wake_dict_dipole': wake_dict_dipole,
            'wake_dict_quadrupole': wake_dict_quadrupole,
            'transport_matrix': self.matrix,
            })

    if plot_details:
        if ms.plt.get_fignums():
            fig_number = ms.plt.gcf().number
        else:
            fig_number = None
        ms.figure('Forward details')
        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1
        sp_profile = subplot(sp_ctr, xlabel='t (fs)', ylabel='I (kA)')
        sp_ctr += 1
        beam.beamProfile.plot_standard(sp_profile, center='Mean')

        sp_screen = subplot(sp_ctr, title='Screen dist', xlabel='x (mm)', ylabel='Intensity (arb. units)')
        sp_ctr += 1
        screen.plot_standard(sp_screen)

        sp_wake = subplot(sp_ctr, title='Wake', xlabel='t (fs)', ylabel='Wake effect [mrad]')
        sp_ctr += 1
        sp_wake.plot(wake_time*1e15, delta_xp_dipole*1e3)

        if fig_number is not None:
            ms.plt.figure(fig_number)

    #self.logMsg('Forward propagated profile with rms %.1f fs to screen with %.1f um mean' % (beam.beamProfile.rms()*1e15, screen.mean()*1e6))
    return outp_dict

