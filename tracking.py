import numpy as np
#from . import beam
from . import lattice
from . import wf_model

class Tracker:
    """
    Parameters:
        beamline - Can be "Aramis" or "Athos".
        screen_name - For example "SARBD02-DSCR050".
        structure_name - For example "SARUN18-UDCP020".
        meta_data - Look at daq.py for format.
        beam_parameters - Look at config.py for format.
        delta_gap - Calibrated gap.
        structure_center - Structure position for which the beam is centered inside.
        screen_center - Position of the unstreaked beam on the screen.
    """
    def __init__(self, beamline, screen_name, structure_name, meta_data, calibration, forward_options, backward_options):
        self.forward_options = forward_options
        self.backward_options = backward_options
        self.meta_data = meta_data
        self.structure_name = structure_name
        self.screen_name = screen_name
        self.beamline = beamline
        self.calibration = calibration
        self.structure_center = calibration.structure_center
        self.screen_center = calibration.screen_center
        self.lat = lattice.get_beamline_lattice(beamline, meta_data)

        self.matrix = self.lat.get_matrix(structure_name.replace('-', '.'), screen_name.replace('-', '.'))
        self.r12 = self.matrix[0,1]
        self.disp = self.matrix[2,5]
        self.energy_eV = meta_data[screen_name+':ENERGY-OP']*1e6
        self.delta_gap = calibration.delta_gap
        self.structure_position = meta_data[structure_name+':CENTER']*1e-3
        self.structure_gap0 = meta_data[structure_name+':GAP']*1e-3
        self.structure_gap = self.structure_gap0 + self.delta_gap
        self.beam_position = -(self.structure_position - self.structure_center)
        self.structure = wf_model.get_structure(structure_name)

    def forward_propagate(self, beam):
        """
        beam: must be beam corresponding to beginning of self.lat
        """
        mat = self.lat.get_matrix(self.lat.element_names[0].replace('-', '.'), self.structure_name.replace('-', '.'))
        beam.linear_propagate(mat)
        wake_time = beam.beamProfile.time
        energy_eV = beam.energy_eV
        wake_dict_dipole = self.structure.convolve(beam.beamProfile, self.structure_gap/2., self.beam_position, 'Dipole')
        delta_xp_dipole = wake_dict_dipole['wake_potential']/energy_eV
        delta_xp_coords_dip = np.interp(beam['t'], wake_time, delta_xp_dipole)
        quad_wake = self.forward_options['quad_wake']
        if quad_wake:
            wake_dict_quadrupole = self.structure.convolve(beam.beamProfile, self.structure_gap/2., self.beam_position, 'Quadrupole')
            delta_xp_quadrupole = wake_dict_quadrupole['wake_potential']/energy_eV
            delta_xp_coords_quad = np.interp(beam['t'], wake_time, delta_xp_quadrupole)
        else:
            delta_xp_quadrupole = 0.
            delta_xp_coords_quad = 0.

        beam['xp'] += delta_xp_coords_dip + delta_xp_coords_quad

        beam.linear_propagate(self.matrix)
        screen = beam.to_screen_dist(self.forward_options['screen_bins'], 0)
        screen.smoothen(self.forward_options['screen_smoothen'])
        outp_dict = {
                'beam': beam,
                'screen': screen,
                }
        return outp_dict

    def backward_propagate(self, screenDist, beamProfile):
        pass

        """
        screen = copy.deepcopy(screen)
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        charge = wake_effect['charge']

        diff_x = np.diff(wake_x)
        if np.all(diff_x <= 0):
            wake_x = wake_x[::-1]
            q_wake_x = q_wake_x[::-1]
            wake_time = wake_time[::-1]
        elif np.all(diff_x >= 0):
            pass
        else:
            raise ValueError('Wake x is not monotonous')

        if abs(wake_x.max()) > abs(wake_x.min()):
            mask_negative = screen.x < 0
        else:
            mask_negative = screen.x > 0

        if self.compensate_negative_screen and np.any(mask_negative):
            x_positive = -screen.x[mask_negative][::-1]
            y_positive = screen.intensity[mask_negative][::-1]
            if np.all(np.diff(x_positive) < 0):
                x_positive = x_positive[::-1]
                y_positive = y_positive[::-1]
            positive_interp = np.interp(screen.x, x_positive, y_positive, left=0, right=0)
            screen_intensity = screen.intensity + positive_interp
            screen_intensity[mask_negative] = 0
            screen._yy = screen_intensity

        screen.reshape(self.n_particles)
        t_interp0 = np.interp(screen.x, wake_x, wake_time)
        charge_interp, hist_edges = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
        charge_interp[0] = 0
        charge_interp[-1] = 0
        t_interp = (hist_edges[1:] + hist_edges[:-1])/2.

        try:
            if np.any(np.diff(t_interp) < 0):
                t_interp = t_interp[::-1]
                charge_interp = charge_interp[::-1]
            assert np.all(np.diff(t_interp) >= 0)
            bp = BeamProfile(t_interp, charge_interp, self.energy_eV, charge)
        except (ValueError, AssertionError) as e:
            print(e)
            ms.figure('')
            self.set_bs_at_streaker()
            subplot = ms.subplot_factory(2,2)
            sp = subplot(1, title='Wake', xlabel='t', ylabel='$\Delta$ x')
            sp.plot(wake_time, wake_x, label='Dipole')
            sp.plot(wake_time, q_wake_x*self.bs_at_streaker[n_streaker], label='Quad')
            sp.legend()
            sp = subplot(2, title='Screen', xlabel='x')
            sp.plot(screen.x, screen.intensity)

            sp = subplot(3, title='Current profile', xlabel='time', ylabel='Current')
            sp.plot(t_interp, charge_interp)
            plt.show()
            raise
        bp.reshape(self.len_screen)
        bp.cutoff2(self.profile_cutoff)
        bp.crop()
        if np.any(np.isnan(bp.time)) or np.any(np.isnan(bp.current)):
            raise ValueError('NaNs in beam profile')
        if self.bp_smoothen:
            bp.smoothen(self.bp_smoothen)
            bp.reshape(self.len_screen)


        if plot_details:
            ms.figure('track_backward')
            subplot = ms.subplot_factory(2,2)
            sp_wake = subplot(1, title='Wake effect', xlabel='t [fs]', ylabel='$\Delta$ x [mm]')
            sp_wake.plot(wake_time*1e15, wake_x*1e3)

            sp_screen = subplot(2, title='Screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            screen.plot_standard(sp_screen)

            sp_profile = subplot(3, title='Interpolated profile', xlabel='t [fs]', ylabel='Current [kA]')
            bp.plot_standard(sp_profile)

        return bp
        """


