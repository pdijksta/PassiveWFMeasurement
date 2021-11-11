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

class Tracker(LogMsgBase):
    """
    Parameters:
        beamline - Can be "Aramis" or "Athos".
        screen_name - For example "SARBD02-DSCR050".
        structure_name - For example "SARUN18-UDCP020".
        meta_data - Look at daq.py for format.
        beam_parameters - Look at config.py for format.
        delta_gap - Calibrated gap.
        structure_position0 - Structure position for which the beam is centered inside.
        screen_center - Position of the unstreaked beam on the screen.
    """
    def __init__(self, beamline, screen_name, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, force_charge=None, n_particles=config.default_n_particles, logger=None):

        self.logger = logger
        self.force_gap = None
        self.force_beam_position = None
        self._meta_data = None

        self.forward_options = forward_options
        self.backward_options = backward_options
        self.reconstruct_gauss_options = reconstruct_gauss_options
        self.beam_spec = beam_spec
        self.beam_optics = beam_optics
        self.find_beam_position_options = find_beam_position_options

        self.update_calib(calib)
        self.structure_name = structure_name
        self.beamline = beamline
        self.screen_name = screen_name
        self.n_particles = n_particles
        self.force_charge = force_charge

        self.structure = wf_model.get_structure(structure_name, self.logger)
        self.meta_data = meta_data
        self.logMsg('Tracker initialized')

    def update_calib(self, calib):
        self.calib = calib
        self.structure_position0 = calib.structure_position0
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
        self.lat = lattice.get_beamline_lattice(self.beamline, meta_data)
        self.matrix = self.lat.get_matrix(self.structure_name.replace('-', '.'), self.screen_name.replace('-', '.'))
        self.r12 = self.matrix[0,1]
        self.disp = self.matrix[2,5]
        self.energy_eV = meta_data[self.screen_name+':ENERGY-OP']*1e6
        calib_dict = self.calib.gap_and_beam_position_from_meta(meta_data)
        self.structure_position0 = calib_dict['structure_position0']
        self.structure_gap0 = calib_dict['gap0']

        if self.force_charge is None:
            self.total_charge = meta_data[config.beamline_chargepv[self.beamline]]*1e-12
        else:
            self.total_charge = self.force_charge

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

    def gen_beam(self, beamProfile):
        beam_options = self.beam_spec.copy()
        beam_options.update(self.beam_optics)
        beam = gen_beam.beam_from_spec(['x', 't'], beam_options, self.n_particles, beamProfile, self.total_charge, self.energy_eV)
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

    def forward_propagate(self, beam, plot_details=False, output_details=False):
        """
        beam: must be beam corresponding to beginning of self.lat
        """
        beam_init = beam
        mat0 = self.lat.get_matrix(self.lat.element_names[0].replace('-', '.'), self.structure_name.replace('-', '.'))
        beam_before_streaker = beam.linear_propagate(mat0)
        wake_time = beam_init.beamProfile.time
        energy_eV = beam_init.energy_eV
        wake_dict_dipole = self.calc_wake(beam_init.beamProfile, 'Dipole')
        delta_xp_dipole = wake_dict_dipole['wake_potential']/energy_eV
        delta_xp_coords_dip = np.interp(beam['t'], wake_time, delta_xp_dipole)
        quad_wake = self.forward_options['quad_wake']

        if quad_wake:
            wake_dict_quadrupole = self.calc_wake(beam_init.beamProfile, 'Quadrupole')
            delta_xp_quadrupole = wake_dict_quadrupole['wake_potential']/energy_eV
            delta_xp_coords_quad = np.interp(beam['t'], wake_time, delta_xp_quadrupole)*(beam['x']-beam['x'].mean())
        else:
            delta_xp_quadrupole = 0.
            delta_xp_coords_quad = 0.

        beam_after_streaker = beam_before_streaker.child()
        beam_after_streaker['xp'] += delta_xp_coords_dip
        beam_after_streaker['xp'] += delta_xp_coords_quad

        beam_at_screen = beam_after_streaker.linear_propagate(self.matrix)
        screen = beam_at_screen.to_screen_dist(self.forward_options['screen_bins'], 0)
        screen.smoothen(self.forward_options['screen_smoothen'])
        screen.aggressive_cutoff(self.forward_options['screen_cutoff'])
        screen.crop()
        screen.reshape(self.forward_options['len_screen'])
        outp_dict = {
                'screen': screen,
                }
        if output_details:
            outp_dict.update({
                'beam_init': beam_init,
                'beam_before_streaker': beam_before_streaker,
                'beam_after_streaker': beam_after_streaker,
                'beam_at_screen': beam_at_screen,
                'wake_dict_dipole': wake_dict_dipole,
                'transport_matrix0': mat0,
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

    def prepare_screen(self, screen):
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

        if compensate_negative_screen and np.any(mask_negative):
            x_positive = -screen.x[mask_negative][::-1]
            y_positive = screen.intensity[mask_negative][::-1]
            if np.all(np.diff(x_positive) < 0):
                x_positive = x_positive[::-1]
                y_positive = y_positive[::-1]
            positive_interp = np.interp(screen.x, x_positive, y_positive, left=0, right=0)
            screen_intensity = screen.intensity + positive_interp
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

    def backward_propagate(self, screen, beamProfile, plot_details=False):
        if self.total_charge != beamProfile.total_charge:
            raise ValueError('Charges are unequal (pC):', self.total_charge*1e12, beamProfile.total_charge*1e12)

        wf_dict = beamProfile.calc_wake(self.structure, self.structure_gap, self.beam_position, 'Dipole')
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
            bp.aggressive_cutoff(self.backward_options['profile_cutoff'])
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
                'wake_time': wake_time,
                'wake_x': wake_x,
                }
        #self.logMsg('Backward propagated screen with mean %i um and profile with rms %.1f fs to profile with rms %.1f fs' % (screen.mean()*1e6, beamProfile.rms()*1e15, bp.rms()*1e15))
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
        t0 = time.time()
        prec = self.reconstruct_gauss_options['precision']
        tt_range = self.reconstruct_gauss_options['gauss_profile_t_range']
        method = self.reconstruct_gauss_options['method']
        sig_t_range = self.reconstruct_gauss_options['sig_t_range']
        len_profile = self.backward_options['len_profile']

        prepare_dict = self.prepare_screen(meas_screen_raw)
        meas_screen = prepare_dict['screen']
        centroid_meas = prepare_dict['centroid']
        rms_meas = prepare_dict['rms']

        opt_func_screens = []
        opt_func_profiles = []
        opt_func_sigmas = []
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

        def gaussian_baf(sig_t0):
            sig_t = np.round(sig_t0/prec)*prec
            #print('sig_t %.2f, %.2f' % (sig_t*1e15, sig_t0*1e15))
            if sig_t in sig_t_list:
                return 0

            bp_gauss = beam_profile.get_gaussian_profile(sig_t, float(tt_range), len_profile, float(self.total_charge), float(self.energy_eV))


            bp_back0 = self.backward_propagate(meas_screen, bp_gauss)['profile']
            back_dict1 = self.backward_propagate(meas_screen, bp_back0)
            bp_back1 = back_dict1['profile']

            beam = self.gen_beam(bp_back1)
            screen = self.forward_propagate(beam, plot_details=False)['screen']

            index = bisect.bisect(sig_t_list, sig_t)
            sig_t_list.insert(index, sig_t)
            opt_func_screens.insert(index, screen)
            opt_func_profiles.insert(index, bp_back1)
            opt_func_sigmas.insert(index, sig_t)
            opt_func_wake_time.insert(index, back_dict1['wake_time'])
            opt_func_wake_x.insert(index, back_dict1['wake_x'])
            gauss_profiles.insert(index, bp_gauss)

            nonlocal n_iter
            if plot_details:
                bp_gauss.plot_standard(sp_profile, label='Gauss %i %i' % (n_iter, sig_t*1e15), center='Mean')
                bp_back0.plot_standard(sp_profile, label='Back0 %i' % n_iter, center='Mean')
                bp_back1.plot_standard(sp_profile, label='Back1 %i' % n_iter, center='Mean')
                if n_iter == 0:
                    color = meas_screen.plot_standard(sp_screen, label='Meas')[0].get_color()
                    sp_screen.axvline(meas_screen.mean()*1e3, color=color, ls='--')
                color = screen.plot_standard(sp_screen, label='Rec %i' % n_iter)[0].get_color()
                sp_screen.axvline(screen.mean()*1e3, color=color, ls='--')

            n_iter += 1
            return 1

        def get_index_min(output='index'):
            sig_t_arr = np.array(sig_t_list)
            if method == 'centroid':
                centroid_sim = np.array([x.mean() for x in opt_func_screens])
                index_min = np.argmin(np.abs(centroid_sim - centroid_meas))
                sort = np.argsort(centroid_sim)
                t_min = np.interp(centroid_meas, centroid_sim[sort], sig_t_arr[sort])
            elif method == 'rms' or method == 'beamsize':
                rms_sim = np.array([x.rms() for x in opt_func_screens])
                index_min = np.argmin(np.abs(rms_sim - rms_meas))
                sort = np.argsort(rms_sim)
                t_min = np.interp(rms_meas, rms_sim[sort], sig_t_arr[sort])
            else:
                raise ValueError('Method %s unknown' % method)

            if output == 'index':
                return index_min.squeeze()
            elif output == 't_sig':
                return t_min

        #sig_t_arr = np.exp(np.linspace(np.log(np.min(sig_t_range)), np.log(np.max(sig_t_range)), 2))
        for sig_t in sig_t_range:
            gaussian_baf(sig_t)

        for _ in range(self.reconstruct_gauss_options['max_iterations']):
            sig_t_min = get_index_min(output='t_sig')
            gaussian_baf(sig_t_min)
        index_min = get_index_min()

        if index_min == 0:
            print('Warning! index at left border!')
        if index_min == len(sig_t_list)-1:
            print('Warning! index at right border!')

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
               }

        if output_details:
            output.update({
                   'opt_func_screens': opt_func_screens,
                   'opt_func_profiles': opt_func_profiles,
                   'opt_func_sigmas': np.array(opt_func_sigmas),
                   'opt_func_wake_time': np.array(opt_func_wake_time),
                   'opt_func_wake_x': np.array(opt_func_wake_x),
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

        prepare_dict = self.prepare_screen(meas_screen_raw)
        meas_screen = prepare_dict['screen']
        centroid_meas = prepare_dict['centroid']
        rms_meas = prepare_dict['rms']

        beam = self.gen_beam(profile)

        def forward(beam_position):
            beam_position = np.round(beam_position/prec)*prec
            if beam_position in beam_position_list:
                return

            sim_screen = self.forward_propagate_forced(self.structure_gap, beam_position, beam)['screen']

            index = bisect.bisect(beam_position_list, beam_position)
            beam_position_list.insert(index, beam_position)
            sim_screens.insert(index, sim_screen)
            mean_list.insert(index, sim_screen.mean())
            rms_list.insert(index, sim_screen.rms())

        def get_index_min(output='index'):
            beam_offset_arr = np.array(beam_position_list)
            if method == 'centroid':
                centroid_sim = np.array(mean_list)
                index_min = np.argmin(np.abs(centroid_sim - centroid_meas))
                sort = np.argsort(centroid_sim)
                beam_position = np.interp(centroid_meas, centroid_sim[sort], beam_offset_arr[sort])
            elif method == 'rms' or method == 'beamsize':
                rms_sim = np.array(rms_list)
                index_min = np.argmin(np.abs(rms_sim - rms_meas))
                sort = np.argsort(rms_sim)
                beam_position = np.interp(rms_meas, rms_sim[sort], beam_offset_arr[sort])
            else:
                raise ValueError('Method %s unknown' % method)

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

        output = {
                'sim_screens': sim_screens,
                'meas_screen': meas_screen,
                'sim_screen': sim_screens[index],
                'beam_position': beam_position,
                'beam_positions': np.array(beam_position_list),
                'delta_position': delta_position,
                'gap': self.structure_gap,
                'structure_name': self.structure_name,
                'beam_offset0': position0,
                'rms_arr': np.array(rms_list),
                'mean_arr': np.array(mean_list),
                }
        return output


def get_default_tracker(beamline, structure_name, meta_data, calib, **kwargs):
    screen = config.beamline_screens[beamline]
    forward_options = config.get_default_forward_options()
    backward_options = config.get_default_backward_options()
    reconstruct_gauss_options = config.get_default_reconstruct_gauss_options()
    beam_spec = config.get_default_beam_spec()
    beam_optics = config.default_optics[beamline]
    find_beam_position_options = config.get_default_find_beam_position_options()
    return Tracker(beamline, screen, structure_name, meta_data, calib, forward_options, backward_options, reconstruct_gauss_options, beam_spec, beam_optics, find_beam_position_options, **kwargs)

