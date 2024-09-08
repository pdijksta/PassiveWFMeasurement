import time
import copy
import numpy as np
import bisect

from . import config
from . import lattice
from . import wf_model
from . import beam_profile
from . import gen_beam
from . import myplotstyle as ms
from .logMsg import LogMsgBase
from . import obsolete_thinlens_forward

forward_ctr, backward_ctr, rec_ctr = 0, 0, 0

class Tracker(LogMsgBase):
    """
    Parameters:
        beamline - Can be "Aramis" or "Athos".
        screen_name - For example "SARBD02-DSCR050".
        structure_name - For example "SARUN18-UDCP020".
        meta_data - Look at daq.py for format.
        calib - StructureCalibration from calibration.py
        Other args: look at get_default_tracker for an example.
        force_charge: Override EPICS charge PV
        logger: from logMsg.py
    """
    def __init__(self, beamline, screen_name, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, force_charge=None, matching_point=None, logger=None, gen_lat=True, matrix=None, meta_data_type=0):

        self.gen_lat = gen_lat
        if not self.gen_lat:
            self.matrix = matrix
        self.optics_at_streaker = None
        self.logger = logger
        self.force_gap = None
        self.force_beam_position = None
        self._meta_data = None
        self.force_charge = force_charge

        self.forward_options = forward_options
        self.backward_options = backward_options
        self.reconstruct_gauss_options = reconstruct_gauss_options
        self.beam_spec = beam_spec
        self.beam_optics = beam_optics
        self.find_beam_position_options = find_beam_position_options

        self.structure_name = structure_name
        self.update_calib(calib)
        self.beamline = beamline
        self.screen_name = screen_name
        self.n_particles = forward_options['n_particles']
        if matching_point is None:
            self.matching_point = config.optics_matching_points[self.beamline]
        else:
            self.matching_point = matching_point

        self.structure = wf_model.get_structure(structure_name, self.logger)
        self.update_calib(calib)
        self.meta_data_type = meta_data_type
        if self.meta_data_type == 0:
            self.meta_data = meta_data
            if self.meta_data is not None:
                self.logMsg('Tracker initialized with gap %i um, structure_position0 %i um' % (round(self.structure_gap*1e6), round(calib.structure_position0*1e6)))
        elif self.meta_data_type == 1:
            self.lat = meta_data['lat']
            self.matrix = self.lat.get_matrix(self.structure_name.replace('-', '.'), self.screen_name.replace('-', '.'))
            self.energy_eV = meta_data['energy_eV']
            self.meta_charge = meta_data['charge']
            calib_dict = self.calib.gap_and_beam_position_from_gap0(meta_data['structure_gap'], meta_data['structure_position'])
            self.structure_gap0 = calib_dict['gap0']

    def update_calib(self, calib):
        assert self.structure_name == calib.structure_name
        self.calib = calib
        self.screen_center = calib.screen_center
        self.delta_gap = calib.delta_gap
        if self._meta_data is not None:
            self.meta_data = self._meta_data

    @property
    def structure_gap(self):
        if self.force_gap is None:
            calib_dict = self.calib.gap_and_beam_position_from_meta(self.meta_data)
            return calib_dict['gap']
        else:
            return self.force_gap

    @property
    def beam_position(self):
        if self.force_beam_position is None:
            calib_dict = self.calib.gap_and_beam_position_from_meta(self.meta_data)
            return calib_dict['beam_position']
        else:
            return self.force_beam_position

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data):
        if meta_data is None:
            return
        self._meta_data = meta_data
        if not self.gen_lat:
            self.lat = None
        else:
            self.lat = lattice.get_beamline_lattice(self.beamline, meta_data)
            self.matrix = self.lat.get_matrix(self.structure_name.replace('-', '.'), self.screen_name.replace('-', '.'))
        energy_pv = self.screen_name+':ENERGY-OP'
        if energy_pv in meta_data:
            self.energy_eV = meta_data[energy_pv]*1e6
        elif self.beamline in config.beamline_energypv:
            self.energy_eV = meta_data[config.beamline_energypv[self.beamline]]*1e6
        else:
            raise KeyError(meta_data.keys())
        calib_dict = self.calib.gap_and_beam_position_from_meta(meta_data)
        self.structure_gap0 = calib_dict['gap0']
        self.meta_charge = meta_data[config.beamline_chargepv[self.beamline]]*1e-12

    @property
    def r12(self):
        if self.structure.dim == 'X':
            return self.matrix[0,1]
        elif self.structure.dim == 'Y':
            return self.matrix[2,3]

    @property
    def disp(self):
        if self.structure.dim == 'X':
            return self.matrix[2,5]
        elif self.structure.dim == 'Y':
            return self.matrix[0,5]

    @property
    def total_charge(self):
        if self.force_charge is None:
            return self.meta_charge
        else:
            return self.force_charge

    def forward_propagate_forced(self, gap, beam_position, *args, **kwargs):
        old_gap = self.force_gap
        old_bo = self.force_beam_position
        self.force_gap = gap
        self.force_beam_position = beam_position
        try:
            outp = self.forward_propagate(*args, **kwargs)
        finally:
            self.force_gap = old_gap
            self.force_beam_position = old_bo
        return outp

    def gen_beam(self, beamProfile, other_dim=False, delta=False):
        if self.forward_options['method'] == 'ocelot':
            other_dim = delta = True
        dim = self.structure.dim.lower()
        dims = []
        if other_dim:
            if dim == 'x':
                dims.append('y')
            elif dim == 'y':
                dims.append('x')
        dims.extend([dim, 't'])
        if self.forward_options['long_wake'] or delta:
            dims.append('delta')
        beam_options = self.beam_spec.copy()
        beam_optics0 = self.beam_optics
        if self.optics_at_streaker is None:
            beam_optics = self.lat.propagate_optics_dict(beam_optics0, self.matching_point.replace('-','.'), self.structure_name.replace('-','.'))
        else:
            beam_optics = self.optics_at_streaker
        self.last_optics_at_streaker = beam_optics
        beam_options.update(beam_optics)
        beam = gen_beam.beam_from_spec(dims, beam_options, self.n_particles, beamProfile, self.total_charge, self.energy_eV)
        return beam

    def calc_wake(self, profile, type_, force_gap=None, force_beam_position=None):
        if force_gap is None:
            gap = self.structure_gap
        else:
            gap = force_gap
        if force_beam_position is None:
            beam_position = self.beam_position
        else:
            beam_position = force_beam_position
        return profile.calc_wake(self.structure, gap, beam_position, type_)

    def forward_propagate_2D(self, beam, n_splits, hist_bins_2d, output_details=False, beam_positions=None):
        dim = self.structure.dim.lower()
        quad_wake = self.forward_options['quad_wake']
        gap = self.structure_gap
        struct_length = self.structure.Ls
        half_length_drift = lattice.transferMatrixDrift66(struct_length/(2.*n_splits))
        negative_drift = lattice.transferMatrixDrift66(-struct_length/2.)

        if beam_positions is None:
            beam_positions = [self.beam_position]*n_splits

        beam_now = beam.linear_propagate(negative_drift)

        outp_split = {
                'beams_in_structure': [],
                'wake2d_dicts_dipole': [],
                'wake2d_dicts_quad': [],
                }

        for n_split, beam_position in enumerate(beam_positions):
            beam_now = beam_now.linear_propagate(half_length_drift)
            x_coords = beam_now[dim]
            t_coords = beam_now['t'] - beam_now['t'].min()
            dict_dipole_2d = wf_model.wf2d(t_coords, x_coords+beam_position, gap/2., self.total_charge, self.structure.wxd, hist_bins_2d)
            delta_xp_dipole = dict_dipole_2d['wake_on_particles']/self.energy_eV

            if quad_wake:
                dict_quad_2d = wf_model.wf2d_quad(t_coords, x_coords+beam_position, gap/2., self.total_charge, self.structure.wxq, hist_bins_2d)
                delta_xp_quad = dict_quad_2d['wake_on_particles']/self.energy_eV
                #sum_good = np.sum(np.sign(delta_xp_quad) == np.sign(x_coords))
                #sum_bad = np.sum(np.sign(delta_xp_quad) != np.sign(x_coords))
                #print(sum_good, sum_bad)
            else:
                dict_quad_2d = None
                delta_xp_quad = 0
            beam_now[dim+'p'] += (delta_xp_dipole + delta_xp_quad)/n_splits
            outp_split['beams_in_structure'].append(beam_now)
            outp_split['wake2d_dicts_dipole'].append(dict_dipole_2d)
            outp_split['wake2d_dicts_quad'].append(dict_dipole_2d)
            beam_now = beam_now.linear_propagate(half_length_drift)
        beam_after_streaker = beam_now.linear_propagate(negative_drift)

        beam_at_screen = beam_after_streaker.linear_propagate(self.matrix)
        screen = self._beam2screen(beam_at_screen)

        outp = {'screen': screen}

        if output_details:
            outp.update({
                'beam': beam,
                'beam_after_streaker': beam_after_streaker,
                'beam_at_screen': beam_at_screen,
                'splits': outp_split,
                })

        return outp

    def _beam2screen(self, beam):
        screen = beam.to_screen_dist(self.forward_options['screen_bins'], 0, self.structure.dim.lower())
        screen.smoothen(self.forward_options['screen_smoothen'])
        screen.aggressive_cutoff(self.forward_options['screen_cutoff'])
        screen.crop()
        screen.reshape(self.forward_options['len_screen'])
        return screen

    def forward_propagate(self, beam, plot_details=False, output_details=False, **kwargs):
        global forward_ctr
        forward_ctr += 1
        method = self.forward_options['method']
        if method == 'thicklens':
            return self.forward_propagate_thicklens(beam, plot_details, output_details, **kwargs)
        elif method == 'thinlens':
            return obsolete_thinlens_forward.forward_propagate_thinlens(self, beam, plot_details, output_details, **kwargs)
        elif method == 'ocelot':
            from . import ocelot_forward
            return ocelot_forward.forward_propagate_ocelot(self, beam, plot_details, output_details, **kwargs)

    def forward_propagate_thicklens(self, beam, plot_details=False, output_details=False):
        """
        beam: must correspond to middle of structure
        """

        beam0 = beam
        wake_time = beam.beamProfile.time
        dim = self.structure.dim.lower()
        other_dim = 'x' if dim == 'y' else 'y'
        mean_energy = beam.energy_eV

        if 'delta' in beam.dim_index:
            energy_eV = beam.energy_eV*(1+beam['delta'])
        else:
            energy_eV = beam.energy_eV

        if self.forward_options['dipole_wake']:
            wake_dict_dipole = self.calc_wake(beam.beamProfile, 'Dipole')
            delta_xp_coords_dip = np.interp(beam['t'], wake_time, wake_dict_dipole['wake_potential'])/energy_eV
        else:
            wake_dict_dipole = None
            delta_xp_coords_dip = 0

        if self.forward_options['quad_wake']:
            wake_dict_quadrupole = self.calc_wake(beam.beamProfile, 'Quadrupole')
            eff_quad_pot_interp = np.interp(beam['t'], wake_time, wake_dict_quadrupole['wake_potential'])/energy_eV
            k1 = {'y': 1, 'x': -1}[dim]*eff_quad_pot_interp/self.structure.Ls
            quad_matrix = lattice.transferMatrixQuad66_arr(self.structure.Ls, k1) # 6x6xN matrix

            halfmat = lattice.transferMatrixDrift66(-self.structure.Ls/2.)
            beam_before = beam.linear_propagate(halfmat)
            beam_after = beam_before.child()
            if 'x' in beam.dim_index:
                x, xp = beam_before['x'], beam_before['xp']
                x = x - x.mean()
                xp = xp - xp.mean()
                beam_after['x'] = quad_matrix[0,0] * x + quad_matrix[0,1] * xp
                beam_after['xp'] = quad_matrix[1,0] * x + quad_matrix[1,1] * xp
            if 'y' in beam.dim_index:
                y, yp = beam_before['y'], beam_before['yp']
                y = y - y.mean()
                yp = yp - yp.mean()
                beam_after['y'] = quad_matrix[2,2] * y + quad_matrix[2,3] * yp
                beam_after['yp'] = quad_matrix[3,2] * y + quad_matrix[3,3] * yp
            beam = beam_after.linear_propagate(halfmat)
        else:
            wake_dict_quadrupole = None
            beam_before = None

        beam_after_streaker = beam.child()

        if self.forward_options['long_wake']:
            wake_dict_long = self.calc_wake(beam0.beamProfile, 'Longitudinal')
            delta_p = wake_dict_long['wake_potential']/mean_energy
            delta_p_interp = np.interp(beam0['t'], wake_time, delta_p)
        else:
            delta_p_interp = 0
            wake_dict_long = None

        if self.forward_options['long_wake_correction']:
            wake_dict_c1 = self.calc_wake(beam0.beamProfile, 'LongitudinalC1')
            wake_dict_c2 = self.calc_wake(beam0.beamProfile, 'LongitudinalC2')
            corr1 = wake_dict_c1['wake_potential']/mean_energy
            corr2 = wake_dict_c2['wake_potential']/mean_energy
            delta_dim = beam0[dim] - beam0[dim].mean()
            delta_other = beam0[other_dim] - beam0[other_dim].mean()
            corr1_interp = np.interp(beam0['t'], wake_time, corr1) * -delta_dim
            corr2_interp = np.interp(beam0['t'], wake_time, corr2) * (-delta_dim**2 + delta_other**2)/2
        else:
            corr1_interp = corr2_interp = 0
            wake_dict_c1 = wake_dict_c2 = None

        if 'delta' in beam_after_streaker.dim_index:
            beam_after_streaker['delta'] += delta_p_interp + corr1_interp + corr2_interp
        beam_after_streaker[dim+'p'] += delta_xp_coords_dip

        beam_at_screen = beam_after_streaker.linear_propagate(self.matrix)
        screen = self._beam2screen(beam_at_screen)
        outp_dict = {'screen': screen}

        if output_details:
            outp_dict.update({
                'beam': beam,
                'beam_before_streaker': beam_before,
                'beam_after_streaker': beam_after_streaker,
                'beam_at_screen': beam_at_screen,
                'wake_dict_dipole': wake_dict_dipole,
                'wake_dict_quadrupole': wake_dict_quadrupole,
                'wake_dict_long': wake_dict_long,
                'wake_dict_c1': wake_dict_c1,
                'wake_dict_c2': wake_dict_c2,
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
            sp_wake.plot(wake_time*1e15, wake_dict_dipole['wake_potential']/mean_energy*1e3)

            sp_spw = subplot(sp_ctr, title='Single particle wake', xlabel='t (fs)', ylabel='Spw')
            sp_ctr += 1
            sp_spw.plot(wake_time*1e15, wake_dict_dipole['spw'])

            if fig_number is not None:
                ms.plt.figure(fig_number)

        #self.logMsg('Forward propagated profile with rms %.1f fs to screen with %.1f um mean' % (beam.beamProfile.rms()*1e15, screen.mean()*1e6))
        return outp_dict

    def prepare_screen(self, screen):
        """
        Returns: dict with screen, mean, rms
        """
        screen0 = screen
        screen = copy.deepcopy(screen0)

        screen.aggressive_cutoff(self.forward_options['screen_cutoff'])
        screen.crop()
        mean, rms = screen.mean(), screen.rms()

        if screen.mean() > 0:
            mask_negative = screen.x < 0
        else:
            mask_negative = screen.x > 0

        compensate_negative_screen = self.backward_options['compensate_negative_screen']
        compensate_rms_factor = self.backward_options['compensate_rms_factor']

        if np.any(mask_negative):

            if compensate_negative_screen:
                x_positive = -screen.x[mask_negative][::-1]
                y_positive = screen.intensity[mask_negative][::-1]
                if np.all(np.diff(x_positive) < 0):
                    x_positive = x_positive[::-1]
                    y_positive = y_positive[::-1]
                positive_interp = np.interp(screen.x, x_positive, y_positive, left=0, right=0)
                screen_intensity = screen.intensity + positive_interp
                screen_intensity[mask_negative] = 0
                screen._yy = screen_intensity
            else:
                first_positive = np.logical_and(~mask_negative, np.abs(screen.x) < abs(rms)*compensate_rms_factor)
                screen_intensity = screen.intensity
                sum_negative = screen_intensity[mask_negative].sum()
                screen_intensity[first_positive] += sum_negative / np.sum(first_positive)
                screen_intensity[mask_negative] = 0
                screen._yy = screen_intensity

        screen.crop()
        screen.reshape(self.n_particles)

        outp = {
                'screen': screen,
                'centroid': mean,
                'rms': rms,
                }

        return outp

    def backward_propagate(self, screen, beamProfile, plot_details=False, beam_position=None):
        if self.total_charge != beamProfile.total_charge:
            raise ValueError('Charges are unequal (pC), tracker/beam_profile:', self.total_charge*1e12, beamProfile.total_charge*1e12)
        global backward_ctr
        backward_ctr += 1

        beamProfile = copy.deepcopy(beamProfile)
        beamProfile.expand(0.3)
        _beam_position = self.beam_position if beam_position is None else beam_position

        wf_dict = beamProfile.calc_wake(self.structure, self.structure_gap, _beam_position, 'Dipole')
        wake_time = wf_dict['wake_time']
        wake_deltaE = wf_dict['wake_potential']
        wake_x = wake_deltaE/self.energy_eV * self.r12

        if np.any(np.diff(wake_x) < 0):
            wake_x = wake_x[::-1]
            wake_time = wake_time[::-1]
        assert np.all(np.diff(wake_x) >= 0)
        t_interp0 = np.interp(screen.x, wake_x, wake_time)

        #print('Backward propagate')
        #print('Screen min max (um)', int(screen.x.min()*1e6), int(screen.x.max()*1e6))
        #print('Wake min max (um)', int(wake_x.min()*1e6), int(wake_x.max()*1e6))
        #print('Beam profile time (fs)', int(beamProfile.time.min()*1e15), int(beamProfile.time.max()*1e15))

        #charge_interp, hist_edges = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
        bins2 = np.concatenate([beamProfile.time, [beamProfile.time[-1] + beamProfile.time[1] - beamProfile.time[0]]])
        charge_interp, hist_edges = np.histogram(t_interp0, bins=bins2, weights=screen.intensity, density=True)
        # UNSURE
        #charge_interp[0] = 0
        #charge_interp[-1] = 0
        t_interp = (hist_edges[1:] + hist_edges[:-1])/2.

        try:
            if np.any(np.diff(t_interp) < 0):
                t_interp = t_interp[::-1]
                charge_interp = charge_interp[::-1]
            assert np.all(np.diff(t_interp) >= 0)
            bp = beam_profile.BeamProfile(t_interp, charge_interp, self.energy_eV, self.total_charge)
            bp.smoothen(self.backward_options['profile_smoothen'])
            bp.cutoff(self.backward_options['profile_cutoff'])
            bp.crop()
            bp.reshape(self.backward_options['len_profile'])
            if np.any(np.isnan(bp.charge_dist)):
                raise ValueError
        except (ValueError, AssertionError) as e:
            print(e)
            ms.figure('')
            #self.set_bs_at_streaker()
            subplot = ms.subplot_factory(2,2)
            sp = subplot(1, title='Wake', xlabel='t', ylabel='$\Delta$ x')
            sp.plot(wake_time, wake_x, label='Dipole')
            #sp.plot(wake_time, q_wake_x*self.bs_at_streaker[n_streaker], label='Quad')
            sp.legend()
            sp = subplot(2, title='Screen', xlabel='x')
            sp.plot(screen.x, screen.intensity)

            sp = subplot(3, title='Current profile', xlabel='time', ylabel='Current')
            sp.plot(t_interp, charge_interp)
            ms.plt.show()
            raise

        if plot_details:
            ms.figure('track_backward')
            subplot = ms.subplot_factory(2,2)
            sp_wake = subplot(1, title='Wake effect', xlabel='t [fs]', ylabel='$\Delta$ x [mm]')
            sp_wake.plot(wake_time*1e15, wake_x*1e3)

            sp_screen = subplot(2, title='Screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            screen.plot_standard(sp_screen)

            sp_profile = subplot(3, title='Interpolated profile', xlabel='t [fs]', ylabel='Current [kA]')
            bp.plot_standard(sp_profile, center='Mean')

        outp_dict = {
                'profile': bp,
                'screen': screen,
                'wake_time': wake_time,
                'wake_x': wake_x,
                }
        return outp_dict

    def reconstruct_profile_Gauss_forced(self, forced_gap, forced_beam_position, *args, **kwargs):
        force_gap0 = self.force_gap
        pos0 = self.force_beam_position
        try:
            self.force_gap = forced_gap
            self.force_beam_position = forced_beam_position
            outp = self.reconstruct_profile_Gauss(*args, **kwargs)
        finally:
            self.force_gap = force_gap0
            self.force_beam_position = pos0
        return outp

    def reconstruct_profile_Gauss(self, meas_screen_raw, output_details=False, plot_details=False):
        global rec_ctr
        rec_ctr += 1

        t0 = time.time()
        prec = self.reconstruct_gauss_options['precision']
        tt_range = self.reconstruct_gauss_options['gauss_profile_t_range']
        method = self.reconstruct_gauss_options['method']
        sig_t_range = self.reconstruct_gauss_options['sig_t_range']
        len_profile = self.backward_options['len_profile']
        init_func = self.reconstruct_gauss_options['init_func']

        prepare_dict = self.prepare_screen(meas_screen_raw)
        meas_screen = prepare_dict['screen']
        centroid_meas = prepare_dict['centroid']
        rms_meas = prepare_dict['rms']

        opt_func_screens = []
        opt_func_profiles0 = []
        opt_func_profiles = []
        opt_func_wake_time = []
        opt_func_wake_x = []
        gauss_profiles = []
        sig_t_list = []

        if plot_details:
            fig_number = ms.plt.gcf().number
            ms.figure('Gauss_recon')
            subplot = ms.subplot_factory(2,2)
            sp_ctr = 1
            #sp_wake = subplot(sp_ctr, title='Wake effect', xlabel='t [fs]', ylabel='$\Delta$ x [mm]')
            #sp_ctr += 1
            sp_screen = subplot(sp_ctr, title='Screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp_profile = subplot(sp_ctr, title='Interpolated profile', xlabel='t [fs]', ylabel='Current [kA]')
            sp_ctr += 1

        n_iter = 0

        if init_func == 'gauss':
            init_f = beam_profile.get_gaussian_profile
        elif init_func == 'flat':
            init_f = beam_profile.get_flat_profile
        else:
            raise ValueError

        beam0 = None

        def gaussian_baf(sig_t0):
            if prec:
                sig_t = np.round(sig_t0/prec)*prec
            else:
                sig_t = sig_t0
            #print('sig_t %.2f, %.2f' % (sig_t*1e15, sig_t0*1e15))
            if sig_t in sig_t_list:
                return 0

            bp_gauss = init_f(sig_t, float(tt_range), len_profile, float(self.total_charge), float(self.energy_eV))
            bp_back0 = self.backward_propagate(meas_screen, bp_gauss)['profile']
            back_dict1 = self.backward_propagate(meas_screen, bp_back0)
            bp_back1 = back_dict1['profile']

            nonlocal beam0
            if beam0 is None:
                beam0 = beam = self.gen_beam(bp_back1)
            else:
                beam = beam0.child()
                beam = beam.new_beamProfile(bp_back1)
            screen = self.forward_propagate(beam, plot_details=False)['screen']

            index = bisect.bisect(sig_t_list, sig_t)
            sig_t_list.insert(index, sig_t)
            opt_func_screens.insert(index, screen)
            opt_func_profiles0.insert(index, bp_back0)
            opt_func_profiles.insert(index, bp_back1)
            opt_func_wake_time.insert(index, back_dict1['wake_time'])
            opt_func_wake_x.insert(index, back_dict1['wake_x'])
            gauss_profiles.insert(index, bp_gauss)

            nonlocal n_iter
            if plot_details:
                bp_gauss.plot_standard(sp_profile, label='Gauss %i %i' % (n_iter, round(sig_t*1e15)), center='Mean')
                bp_back0.plot_standard(sp_profile, label='Back0 %i' % n_iter, center='Mean')
                bp_back1.plot_standard(sp_profile, label='Back1 %i' % n_iter, center='Mean')
                if n_iter == 0:
                    meas_screen.plot_standard(sp_screen, label='Meas', color='black')
                    meas_screen_raw.plot_standard(sp_screen, label='Meas raw', color='black', ls='--')
                    sp_screen.axvline(meas_screen.mean()*1e3, color='black', ls='--')
                color = screen.plot_standard(sp_screen, label='Rec %i' % n_iter)[0].get_color()
                sp_screen.axvline(screen.mean()*1e3, color=color, ls='--')

            n_iter += 1
            return 1

        def get_index_min(output='index'):
            sig_t_arr = np.array(sig_t_list)
            if method == 'centroid':
                vals = np.array([x.mean() for x in opt_func_screens])
                opt = centroid_meas
            elif method == 'rms' or method == 'beamsize':
                vals = np.array([x.rms() for x in opt_func_screens])
                opt = rms_meas
            else:
                raise ValueError('Method %s unknown' % method)
            index_min = np.argmin(np.abs(vals - opt))

            if output == 'index':
                return index_min.squeeze()
            elif output == 't_sig':
                m1 = np.abs(vals)<np.abs(opt)
                if not np.any(m1):
                    t_min = np.min(sig_t_arr)
                else:
                    m2 = np.abs(vals)>np.abs(opt)
                    if not np.any(m2):
                        t_min = np.max(sig_t_arr)
                    else:
                        x1 = np.max(sig_t_arr[m1])
                        x2 = np.min(sig_t_arr[m2])
                        y1 = vals[x1==sig_t_arr].squeeze()
                        y2 = vals[x2==sig_t_arr].squeeze()
                        alpha = (opt-y1)/(y2-y1)
                        t_min = x1+alpha*(x2-x1)
                return t_min

        #sig_t_arr = np.exp(np.linspace(np.log(np.min(sig_t_range)), np.log(np.max(sig_t_range)), 2))
        for sig_t in sig_t_range:
            gaussian_baf(sig_t)

        for _ in range(self.reconstruct_gauss_options['max_iterations']):
            sig_t_min = get_index_min(output='t_sig')
            gaussian_baf(sig_t_min)
        index_min = get_index_min()

        warning = False
        if index_min == 0:
            self.logMsg('Warning! index at left border (reconstruction)!', 'W')
            warning = True
        if index_min == len(sig_t_list)-1:
            self.logMsg('Warning! index at right border (reconstruction)!', 'W')
            warning = True

        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_sig_t = sig_t_list[index_min]
        best_gauss = gauss_profiles[index_min]

        distance = self.structure_gap/2. - abs(self.beam_position)
        time_needed = time.time() - t0
        self.logMsg('iterations %i, duration %i fs, charge %i pC, gap %.2f mm, beam pos %.2f mm, distance %.2f um time %.3f' % (n_iter-2, int(best_profile.rms()*1e15), int(self.total_charge*1e12), self.structure_gap*1e3, self.beam_position*1e3, distance*1e6, time_needed))

        output = {
                'gauss_sigma': best_sig_t,
                'reconstructed_screen': best_screen,
                'reconstructed_profile': best_profile,
                'best_gauss': best_gauss,
                'meas_screen': meas_screen,
                'meas_screen_raw': meas_screen_raw,
                'gap': self.structure_gap,
                'structure': self.structure_name,
                'beam_position': self.beam_position,
                'best_index': index_min,
                'warning': warning,
                'centroid_meas': centroid_meas,
                'rms_meas': rms_meas,
               }

        if output_details:
            output.update({
                'opt_func_sigmas': np.array(sig_t_list),
                'opt_func_screens': opt_func_screens,
                'opt_func_profiles0': opt_func_profiles0,
                'opt_func_profiles': opt_func_profiles,
                'opt_func_wake_time': np.array(opt_func_wake_time),
                'opt_func_wake_x': np.array(opt_func_wake_x),
                'gauss_profiles': gauss_profiles,
                })

        if plot_details:
            sp_profile.legend()
            sp_screen.legend()
            ms.plt.figure(fig_number)

        return output

    def find_beam_position(self, position0, meas_screen_raw, profile):
        position_explore = self.find_beam_position_options['position_explore']
        prec = self.find_beam_position_options['precision']
        method = self.find_beam_position_options['method']
        max_iterations = self.find_beam_position_options['max_iterations']

        beam_position_list = []
        sim_screens = []
        rms_list = []
        mean_list = []
        target_list = []
        n_iter = 0

        prepare_dict = self.prepare_screen(meas_screen_raw)
        centroid_meas = prepare_dict['centroid']
        rms_meas = prepare_dict['rms']
        if method == 'centroid':
            target_meas = centroid_meas
        elif method == 'beamsize' or method == 'rms':
            target_meas = rms_meas

        beam = self.gen_beam(profile)

        def forward(beam_position):
            beam_position = np.round(beam_position/prec)*prec
            if beam_position in beam_position_list:
                return
            nonlocal n_iter
            n_iter += 1

            sim_screen = self.forward_propagate_forced(self.structure_gap, beam_position, beam)['screen']

            index = bisect.bisect(beam_position_list, beam_position)
            beam_position_list.insert(index, beam_position)
            sim_screens.insert(index, sim_screen)
            mean, rms = sim_screen.mean(), sim_screen.rms()
            mean_list.insert(index, mean)
            rms_list.insert(index, rms)
            if method == 'centroid':
                target_list.insert(index, mean)
            elif method == 'beamsize' or method == 'rms':
                target_list.insert(index, rms)

        def get_index_min(output='index'):
            beam_offset_arr = np.array(beam_position_list)
            target_arr = np.array(target_list)
            index_min = np.argmin(np.abs(target_arr - target_meas))
            sort = np.argsort(target_arr)
            beam_position = np.interp(target_meas, target_arr[sort], beam_offset_arr[sort])
            if output == 'index':
                return index_min.squeeze()
            elif output == 'offset':
                return beam_position

        beam_offset_arr = np.linspace(position0-position_explore, position0+position_explore, 3)
        for beam_position in beam_offset_arr:
            forward(beam_position)
        for _ in range(max_iterations):
            beam_position = get_index_min(output='offset')
            forward(beam_position)
        index = get_index_min(output='index')
        beam_position = beam_position_list[index]
        delta_position = beam_position - position0
        distance = self.structure_gap/2 - abs(beam_position)

        output = {
                'sim_screens': sim_screens,
                'meas_screen': meas_screen_raw,
                'sim_screen': sim_screens[index],
                'beam_position': beam_position,
                'beam_positions': np.array(beam_position_list),
                'centroid_meas': centroid_meas,
                'rms_meas': rms_meas,
                'delta_position': delta_position,
                'gap': self.structure_gap,
                'structure_name': self.structure_name,
                'beam_offset0': position0,
                'rms_arr': np.array(rms_list),
                'mean_arr': np.array(mean_list),
                'distance': distance,
                }
        self.logMsg('Beam position found. Distance %i um. Delta: %i um. Target: %.2f mm. Result: %.2f mm. %i iterations' % (round(distance*1e6), round(delta_position*1e6), target_meas*1e3, target_list[index]*1e3, n_iter-3))
        return output

    def find_beam_position_backward(self, position0, meas_screen_raw, profile):
        position_explore = self.find_beam_position_options['position_explore']
        prec = self.find_beam_position_options['precision']
        max_iterations = self.find_beam_position_options['max_iterations']

        beam_position_list = []
        back_profile_list = []
        target_list = []
        n_iter = 0

        prepare_dict = self.prepare_screen(meas_screen_raw)
        screen = prepare_dict['screen']
        target = profile.rms()

        def backward(beam_position):
            beam_position = np.round(beam_position/prec)*prec
            if beam_position in beam_position_list:
                return
            nonlocal n_iter
            n_iter += 1
            back_profile = self.backward_propagate(screen, profile, beam_position=beam_position)['profile']

            index = bisect.bisect(beam_position_list, beam_position)
            beam_position_list.insert(index, beam_position)
            back_profile_list.insert(index, back_profile)
            target_list.insert(index, back_profile.rms())

        def get_index_min(output='index'):
            beam_offset_arr = np.array(beam_position_list)
            target_arr = np.array(target_list)
            index_min = np.argmin(np.abs(target_arr - target))
            sort = np.argsort(target_arr)
            beam_position = np.interp(target, target_arr[sort], beam_offset_arr[sort])
            if output == 'index':
                return index_min.squeeze()
            elif output == 'offset':
                return beam_position

        beam_offset_arr = np.linspace(position0-position_explore, position0+position_explore, 3)
        for beam_position in beam_offset_arr:
            backward(beam_position)
        for _ in range(max_iterations):
            beam_position = get_index_min(output='offset')
            backward(beam_position)
        index = get_index_min(output='index')
        beam_position = beam_position_list[index]
        delta_position = beam_position - position0
        distance = self.structure_gap/2 - abs(beam_position)

        output = {
                'back_profiles': back_profile_list,
                'meas_screen': meas_screen_raw,
                'back_profile': back_profile_list[index],
                'beam_position': beam_position,
                'beam_positions': np.array(beam_position_list),
                'target_rms': target,
                'delta_position': delta_position,
                'gap': self.structure_gap,
                'structure_name': self.structure_name,
                'beam_offset0': position0,
                'rms_arr': np.array(target_list),
                'distance': distance,
                }
        self.logMsg('Beam position found. Distance %i um. Delta: %i um. Target: %.1f fs. Result: %.1f fs. %i iterations' % (round(distance*1e6), round(delta_position*1e6), target*1e15, target_list[index]*1e15, n_iter-3))
        return output


def get_default_tracker(beamline, structure_name, meta_data, calib, screen, **kwargs):
    forward_options = config.get_default_forward_options()
    backward_options = config.get_default_backward_options()
    reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
    beam_spec = config.get_default_beam_spec()
    if 'beam_optics' not in kwargs:
        beam_optics = config.default_optics[beamline]
    else:
        beam_optics = kwargs.pop('beam_optics')
    find_beam_position_options = config.get_default_find_beam_position_options()
    return Tracker(beamline, screen, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, **kwargs)

