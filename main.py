import matplotlib.pyplot as plt; plt # Without this line, there is an error...
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib
matplotlib.use('Qt5Agg')

import sys
import os
import re
import socket
from datetime import datetime
import numpy as np
import PyQt5.Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtRemoveInputHook


path = os.path.join(os.path.dirname(__file__), '../')
if path not in sys.path:
    sys.path.append(path)

from PassiveWFMeasurement import config
from PassiveWFMeasurement import tracking
from PassiveWFMeasurement import lasing
from PassiveWFMeasurement import h5_storage
from PassiveWFMeasurement import calibration
from PassiveWFMeasurement import beam_profile
from PassiveWFMeasurement import plot_results
from PassiveWFMeasurement import resolution
from PassiveWFMeasurement import data_loader
from PassiveWFMeasurement import blmeas
from PassiveWFMeasurement import logMsg
from PassiveWFMeasurement import myplotstyle as ms

#TODO
#
# - add info of beamsize with / without assumed screen resolution
# - add tilt option
# - handle feedback in user interface
# - detune undulator button
# - streaker center calibration: repeat with one data point removed at one side
# - Offset based on centroid, offset based on sizes (?)
# - Dispersion (?)
# - Plot centroid of forward propagated
# - add blmeas option to lasing rec
# - Mean of square instead of square of mean of squareroot
# - Athos gas detector
# - Calibration based on TDC
# - Fix calibration - screen and streaker offset at same time

#Problematic / cannot be done easily:
# - save BPM data also
# - One-sided plate
# - Update load blmeas (need bugfixes by Thomas)

# Probably fixed:
# - sort out daq pyscan_result_to_dict
# - debug delay after using BsreadPositioner or any pyscan

# Not so important
# - noise reduction from the image
# - uJ instead of True, False
# - non blocking daq
# - One-sided plate

# Done
# - pulse energy from gas detector in pyscan
# - yum install libhdf5
# - streaker calibration fit guess improvements
# - meta data at begin and end of pyscan
# - lasing
# - y scale of optimization
# - elog
# - charge from pyscan
# - Forward propagation from TDC to screen inside tool
# - plot TDC blmeas next to current reconstruction (optional)
# - Show sizes
# - simplify lattice
# - restructure analysis
# - Rec plot legends
# - Comments to elog
# - optional provide the pulse energy calibration
# - png.png
# - R12 in Athos is wrong - does not change when quad is changed
# - Add resolution to tool
# - Fix non converging calibration
# - Fix erronous gap calibration
# - Fix systematic current profile reconstruction differences

try:
    from . import daq
    always_dryrun = False
except ImportError:
    print('Cannot import daq. Always dry_run True')
    always_dryrun = True
    daq = None

try:
    import elog
except ImportError:
    print('ELOG not available')
    elog = None

ms.set_fontsizes(config.fontsize)

pyqtRemoveInputHook() # for pdb to work
re_time = re.compile('(\\d{4})-(\\d{2})-(\\d{2}):(\\d{2})-(\\d{2})-(\\d{2})')

class StartMain(QtWidgets.QMainWindow, logMsg.LogMsgBase):

    charge_pv_text = 'Use PV'

    def __init__(self, logger=None):
        super(StartMain, self).__init__()
        uic.loadUi('gui.ui', self)
        self.logger = logger

        self.DoReconstruction.clicked.connect(self.reconstruct_current)
        self.SaveCurrentRecData.clicked.connect(self.save_current_rec_data)
        self.SaveLasingRecData.clicked.connect(self.save_lasing_rec_data)
        self.CloseAll.clicked.connect(self.clear_rec_plots)
        self.CalibrateStreaker.clicked.connect(self.calibrate_streaker)
        self.GapReconstruction.clicked.connect(self.gap_reconstruction)
        self.ClearCalibPlots.clicked.connect(self.clear_calib_plots)
        self.ClearGapRecPlots.clicked.connect(self.clear_gap_recon_plots)
        self.LoadCalibration.clicked.connect(self.load_calibration)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction)
        self.ObtainLasingOnData.clicked.connect(self.obtainLasingOn)
        self.ObtainLasingOffData.clicked.connect(self.obtainLasingOff)
        self.ReconstructLasing.clicked.connect(self.reconstruct_all_lasing)
        self.ObtainR12.clicked.connect(self.obtain_r12_0)
        self.PlotResolution.clicked.connect(self.plot_resolution)

        self.BeamlineSelect.addItems(sorted(config.structure_names.keys()))
        self.StructureSelect.addItems(sorted(config.structure_parameters.keys()))

        # Default strings in gui fields
        hostname = socket.gethostname()
        if hostname == 'desktop':
            default_dir = '/storage/data_2021-05-18/'
            save_dir = '/storage/tmp_reconstruction/'
        elif hostname == 'pubuntu':
            default_dir = '/home/work/data_2021-05-18/'
            save_dir = '/home/work/tmp_reconstruction/'
        elif 'psi' in hostname or 'lc6a' in hostname or 'lc7a' in hostname or True:
            default_dir = '/sf/data/measurements/2021/05/18/'
            date = datetime.now()
            save_dir = date.strftime('/sf/data/measurements/%Y/%m/%d/')

        bunch_length_meas_file = default_dir + '119325494_bunch_length_meas.h5'
        #recon_data_file = default_dir+'2021_05_18-17_41_02_PassiveReconstruction.h5'
        lasing_file_off = default_dir + '2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'
        lasing_file_on = default_dir + '2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
        streaker_calib_file = default_dir + '2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
        screen_center = 898.02e-6
        structure_center = 364e-6
        delta_gap = -62e-6
        pulse_energy = 180e-6

        self.ScreenCenterCalib.setText('%i' % round(screen_center*1e6))
        self.StructureCenter.setText('%i' % round(structure_center*1e6))
        self.StructureGapDelta.setText('%i' % round(delta_gap*1e6))
        self.LasingEnergyInput.setText('%i' % round(pulse_energy*1e6))

        self.ReconstructionDataLoad.setText(lasing_file_off)
        self.BunchLengthMeasFile.setText(bunch_length_meas_file)
        self.LasingOnDataLoad.setText(lasing_file_on)
        self.LasingOffDataLoad.setText(lasing_file_off)
        self.LoadCalibrationFilename.setText(streaker_calib_file)
        self.ForwardBlmeasFilename.setText(bunch_length_meas_file)

        bs = config.get_default_beam_spec()
        fs = config.get_default_forward_options()
        backs = config.get_default_backward_options()
        rs = config.get_default_reconstruct_gauss_options()
        optics = config.get_default_optics(self.beamline)
        gs = config.get_default_structure_calibrator_options()
        fbs = config.get_default_find_beam_position_options()
        ls = config.get_default_lasing_options()

        self.N_Particles.setText('%i' % config.default_n_particles)

        # Forward options
        self.ScreenBins.setText('%i' % fs['screen_bins'])
        self.ScreenCutoff.setText('%.4f' % fs['screen_cutoff'])
        self.ScreenSmoothen.setText('%i' % (fs['screen_smoothen']*1e6))
        self.ScreenSize.setText('%i' % fs['len_screen'])

        # Backward options
        self.ProfileCutoff.setText('%.4f' % backs['profile_cutoff'])
        self.ProfileSmoothen.setText('%.4f' % (backs['profile_smoothen']*1e15))
        self.ProfileSize.setText('%i' % backs['len_profile'])

        # Gauss reconstruction options
        self.SigTfsMin.setText('%.3f' % (rs['sig_t_range'][0]*1e15))
        self.SigTfsMax.setText('%.3f' % (rs['sig_t_range'][1]*1e15))
        self.GaussReconMaxIter.setText('%i' % rs['max_iterations'])
        self.ProfileExtent.setText('%.3f' % (rs['gauss_profile_t_range']*1e15))

        # Beam specifications
        self.TransEmittanceX.setText('%.4f' % (bs['nemitx']*1e9))
        self.TransEmittanceY.setText('%.4f' % (bs['nemity']*1e9))
        self.BetaX.setText('%.4f' % optics['betax'])
        self.AlphaX.setText('%.4f' % optics['alphax'])
        self.BetaY.setText('%.4f' % optics['betay'])
        self.AlphaY.setText('%.4f' % optics['alphay'])

        # Other
        self.Charge.setText(self.charge_pv_text)
        self.SaveDir.setText(save_dir)

        # Gap reconstruction options
        _lower = gs['delta_gap_scan_range'].min()
        _upper = gs['delta_gap_scan_range'].max()
        _step = gs['delta_gap_scan_range'][1] - gs['delta_gap_scan_range'][0]
        self.DeltaGapRangeLower.setText('%.3f' % (_lower*1e6))
        self.DeltaGapRangeUpper.setText('%.3f' % (_upper*1e6))
        self.DeltaGapRangeStep.setText('%.3f' % (_step*1e6))
        self.DeltaGapSearchLower.setText('%.3f' % (gs['delta_gap_range'].min()*1e6))
        self.DeltaGapSearchUpper.setText('%.3f' % (gs['delta_gap_range'].max()*1e6))
        self.DeltaGapSearchPoints.setText('%.3f' % len(gs['delta_gap_range']))
        self.StructCenterSearchLower.setText('%.3f' % (gs['delta_streaker0_range'].min()*1e6))
        self.StructCenterSearchUpper.setText('%.3f' % (gs['delta_streaker0_range'].max()*1e6))
        self.StructCenterSearchPoints.setText('%.3f' % len(gs['delta_streaker0_range']))

        # Find beam position options
        self.FindBeamMaxIter.setText('%i' % fbs['max_iterations'])
        self.FindBeamExplorationRange.setText('%.3f' % (fbs['position_explore']*1e6))

        # Lasing options
        self.LasingReconstructionSliceFactor.setText('%i' % ls['slice_factor'])
        self.LasingNoiseCut.setText('%.4f' % ls['noise_cut'])
        self.LasingIntensityCut.setText('%.4f' % ls['subtract_quantile'])

        if elog is not None:
            self.logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning+data/', user='robot', password='robot')

        self.current_rec_dict = None
        self.lasing_rec_dict = None

        ## Init plots
        def get_new_tab(fig, title):
            new_tab = QtWidgets.QWidget()
            layout = PyQt5.Qt.QVBoxLayout()
            new_tab.setLayout(layout)
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, self)
            layout.addWidget(canvas)
            layout.addWidget(toolbar)
            tab_index = self.tabWidget.addTab(new_tab, title)
            return tab_index, canvas

        self.reconstruction_fig, self.reconstruction_plot_handles = plot_results.reconstruction_figure()
        self.rec_plot_tab_index, self.rec_canvas = get_new_tab(self.reconstruction_fig, 'I Rec.')

        self.streaker_calib_fig, self.structure_calib_plot_handles = plot_results.streaker_calibration_figure()
        self.streaker_calib_plot_tab_index, self.streaker_calib_canvas = get_new_tab(self.streaker_calib_fig, 'Calib.')

        self.gap_recon_fig, self.gap_recon_plot_handles = plot_results.calib_figure()
        self.gap_recon_tab_index, self.gap_recon_canvas = get_new_tab(self.gap_recon_fig, 'Gap rec.')

        self.all_lasing_fig, self.all_lasing_plot_handles = plot_results.lasing_figure()
        self.all_lasing_tab_index, self.all_lasing_canvas = get_new_tab(self.all_lasing_fig, 'All lasing')

        self.resolution_fig, self.resolution_plot_handles = plot_results.resolution_figure()
        self.resolution_tab_index, self.resolution_canvas = get_new_tab(self.resolution_fig, 'Resolution')

    def clear_rec_plots(self):
        plot_results.clear_reconstruction(*self.reconstruction_plot_handles)
        self.rec_canvas.draw()

    def clear_gap_recon_plots(self):
        plot_results.clear_gap_recon(*self.gap_recon_plot_handles)
        self.gap_recon_canvas.draw()

    def clear_calib_plots(self):
        plot_results.clear_streaker_calibration(*self.structure_calib_plot_handles)
        self.streaker_calib_canvas.draw()

    def clear_screen_plots(self):
        plot_results.clear_screen_calibration(*self.screen_calib_plot_handles)
        self.screen_calib_canvas.draw()

    def clear_all_lasing_plots(self):
        plot_results.clear_lasing_figure(*self.all_lasing_plot_handles)

    def gui_to_calib(self):
        return calibration.StructureCalibration(
                self.structure_name,
                float(self.ScreenCenterCalib.text())/1e6,
                float(self.StructureGapDelta.text())/1e6,
                float(self.StructureCenter.text())/1e6)

    def calib_to_gui(self, calib):
        self.ScreenCenterCalib.setText('%.3f' % (calib.screen_center*1e6))
        self.StructureGapDelta.setText('%.3f' % (calib.delta_gap*1e6))
        self.StructureCenter.setText('%.3f' % (calib.structure_position0*1e6))

    def get_tracker(self, meta_data):
        if self.Charge.text() == self.charge_pv_text:
            force_charge = None
        else:
            force_charge = float(self.Charge.text())*1e-12
        n_particles = int(self.N_Particles.text())

        tracker = tracking.Tracker(
                self.beamline,
                self.screen,
                self.structure_name,
                meta_data,
                self.gui_to_calib(),
                self.get_forward_options(),
                self.get_backward_options(),
                self.get_reconstruct_gauss_options(),
                self.get_beam_spec(), self.get_beam_optics(),
                self.get_find_beam_position_options(),
                force_charge,
                n_particles,
                self.logger)
        return tracker

    def plot_resolution(self):
        # Not ideal but ok for now
        bp_dict = h5_storage.loadH5Recursive(os.path.join(os.path.dirname(__file__), './example_current_profile.h5'))
        gap = 10e-3
        beam_offset = gap/2 - (float(self.PlotResolutionDistance.text())*1e-6)
        meta_data = daq.get_meta_data(self.screen, False, self.beamline)
        tracker = self.get_tracker(meta_data)
        beamprofile = beam_profile.BeamProfile(bp_dict['time_profile'], bp_dict['current'], tracker.energy_eV, tracker.total_charge, self.logger)
        res_dict = resolution.calc_resolution(beamprofile, gap, beam_offset, tracker)

        plot_results.clear_resolution_figure(*self.resolution_plot_handles)
        resolution.plot_resolution(res_dict, *self.resolution_plot_handles)
        self.resolution_canvas.draw()

    def obtain_r12_0(self):
        return self.obtain_r12()

    def obtain_r12(self, meta_data=None):
        if meta_data is None:
            meta_data = daq.get_meta_data(self.screen, self.dry_run, self.beamline)
            print(meta_data)
        tracker = self.get_tracker(meta_data)
        r12 = tracker.r12
        disp = tracker.disp
        print('R12:', r12)
        print('Dispersion:', disp)
        return r12, disp

    @property
    def delta_gap(self):
        return float(self.StructureGapDelta.text())*1e-6

    def get_forward_options(self):
        outp = config.get_default_forward_options()
        outp['screen_bins'] = int(self.ScreenBins.text())
        outp['screen_smoothen'] = float(self.ScreenSmoothen.text())*1e-6
        outp['screen_cutoff'] = float(self.ScreenCutoff.text())
        outp['len_screen'] = int(self.ScreenSize.text())
        return outp

    def get_backward_options(self):
        outp = config.get_default_backward_options()
        outp['profile_cutoff'] = float(self.ProfileCutoff.text())
        outp['len_profile'] = int(self.ProfileSize.text())
        outp['profile_smoothen'] = float(self.ProfileSmoothen.text())*1e-15
        return outp

    def get_reconstruct_gauss_options(self):
        outp = config.get_default_reconstruct_gauss_options
        outp['sig_t_range'] = np.array([float(self.SigTfsMin.text())*1e-15, float(self.SigTfsMax.text())*1e-15])
        outp['max_iterations'] = int(self.GaussReconMaxIter.text())
        outp['gauss_profile_t_range'] = float(self.ProfileExtent.text())*1e-15
        return outp

    def get_beam_spec(self):
        outp = config.get_default_beam_spec()
        outp['nemitx'] = float(self.TransEmittanceX.text())*1e-9
        outp['nemity'] = float(self.TransEmittanceY.text())*1e-9
        return outp

    def get_beam_optics(self):
        outp = config.get_default_optics(self.beamline)
        outp['betax'] = float(self.BetaX.text())
        outp['alphax'] = float(self.AlphaX.text())
        outp['betay'] = float(self.BetaY.text())
        outp['alphay'] = float(self.AlphaY.text())
        return outp

    def get_find_beam_position_options(self):
        outp = config.get_default_find_beam_position_options()
        outp['position_explore'] = float(self.FindBeamExplorationRange.text())*1e-6
        outp['max_iterations'] = int(self.FindBeamMaxIter.text())

    def get_structure_calib_options(self):
        outp = config.get_default_structure_calibrator_options()
        _lower = float(self.DeltaGapRangeLower.text())*1e-6
        _upper = float(self.DeltaGapRangeUpper.text())*1e-6
        _step = float(self.DeltaGapRangeStep.text())*1e-6
        outp['delta_gap_scan_range'] = np.arange(_lower, _upper, _step)
        _lower = float(self.DeltaGapSearchLower.text())*1e-6
        _upper = float(self.DeltaGapSearchUpper.text())*1e-6
        _points = int(self.DeltaGapSearchPoints.text())
        outp['delta_gap_range'] = np.linspace(_lower, _upper, _points)
        _lower = float(self.StructCenterSearchLower.text())*1e-6
        _upper = float(self.StructCenterSearchUpper.text())*1e-6
        _points = int(self.StructCenterSearchPoints.text())
        outp['delta_streaker0_range'] = np.linspace(_lower, _upper, _points)
        return outp

    def get_lasing_options(self):
        outp = config.get_default_lasing_options()
        outp['slice_factor'] = int(self.LasingReconstructionSliceFactor.text())
        outp['noise_cut'] = float(self.LasingNoiseCut.text())
        outp['subtract_quantile'] = float(self.LasingIntensityCut.text())
        return outp

    def reconstruct_current(self):
        self.clear_rec_plots()
        filename = self.ReconstructionDataLoad.text().strip()
        if self.ShowBlmeasCheck.isChecked():
            blmeas_file = self.BunchLengthMeasFile.text()
        else:
            blmeas_file = None

        screen_data = h5_storage.loadH5Recursive(filename)
        if 'meta_data' in screen_data:
            meta_data = screen_data['meta_data']
        elif 'meta_data_begin' in screen_data:
            meta_data = screen_data['meta_data_begin']
        else:
            self.logMsg('Problems with screen data meta data. Available keys: %s' % screen_data.keys(), 'E')
            raise ValueError

        tracker = self.get_tracker(meta_data)
        x_axis, proj = data_loader.screen_data_to_median(screen_data['pyscan_result'])
        meas_screen = beam_profile.ScreenDistribution(x_axis, proj, total_charge=tracker.total_charge)
        self.current_rec_dict = tracker.reconstruct_profile_Gauss(meas_screen, output_details=True)
        plot_results.plot_rec_gauss(self.current_rec_dict, plot_handles=self.reconstruction_plot_handles, blmeas_profiles=blmeas_file)

        self.logMsg('Current profile reconstructed.')

        self.rec_canvas.draw()
        self.tabWidget.setCurrentIndex(self.rec_plot_tab_index)

    def save_current_rec_data(self):
        if self.current_rec_dict is None:
            raise ValueError('No current reconstruction to save!')

        save_path = self.save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_PassiveReconstruction.h5')
        elog_text = 'Passive current reconstruction'
        elog_text +='\nComment: %s' % self.CurrentElogComment.text()
        self.elog_and_H5(elog_text, [self.reconstruction_fig], 'Passive current reconstruction', basename, self.current_rec_dict)

    @property
    def save_dir(self):
        return os.path.expanduser(self.SaveDir.text())

    def calibrate_streaker(self):
        self.clear_calib_plots()
        start, stop, step= float(self.Range1Begin.text()), float(self.Range1Stop.text()), int(float(self.Range1Step.text()))
        range1 = np.linspace(start, stop, step)
        start, stop, step= float(self.Range2Begin.text()), float(self.Range2Stop.text()), int(float(self.Range2Step.text()))
        range2 = np.linspace(start, stop, step)
        range_ = np.concatenate([range1, [0], range2])*1e-3 # Convert mm to m
        range_.sort()
        range_ = np.unique(range_)

        n_images = int(self.CalibrateStreakerImages.text())

        if daq is None:
            raise ImportError('Daq not available')

        result_dict = daq.data_streaker_offset(self.structure_name, range_, self.screen, n_images, self.dry_run, self.beamline)

        try:
            fit_dicts = self._analyze_streaker_calib(result_dict)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_') +'Calibration_data_%s.h5' % self.structure_name.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, result_dict)
            self.logMsg('Saved streaker calibration data %s' % filename)
            raise

        full_dict = {
                'fit_results': fit_dicts,
                'raw_data': result_dict,
                }

        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Calibration_%s.h5' % self.structure_name.replace('.','_')

        calib = fit_dicts['centroid']['calibration']
        elog_text = 'Streaker calibration streaker %s\nCenter: %i um' % (self.structure_name, calib.structure_position0*1e6)
        self.elog_and_H5(elog_text, [self.streaker_calib_fig], 'Streaker center calibration', basename, full_dict)
        self.tabWidget.setCurrentIndex(self.streaker_calib_plot_tab_index)

    def _analyze_streaker_calib(self, result_dict):
        forward_blmeas = self.ForwardBlmeasCheck.isChecked()
        tracker = self.get_tracker(result_dict['meta_data_begin'])
        structure_calib_options = self.get_structure_calib_options()
        if forward_blmeas:
            blmeasfile = self.ForwardBlmeasFilename.text()
        else:
            blmeasfile = None

        sc = calibration.StructureCalibrator(tracker, structure_calib_options, result_dict, self.logger)

        sc.fit()
        if self.ForwardBlmeasCheck.isChecked():
            bp_dict = blmeas.load_avg_blmeas(blmeasfile)
            bp = beam_profile.BeamProfile(bp_dict['time'], bp_dict['current_reduced'], tracker.energy_eV, tracker.total_charge, self.logger)
            forward_options = self.get_forward_options()
            bp.reshape(forward_options['len_screen'])
            bp.aggressive_cutoff(forward_options['screen_cutoff'])
            bp.crop()
            bp.reshape(forward_options['len_screen'])

            sc.forward_propagate(bp)

        calib = sc.fit_dicts['centroid']['calibration']
        self.calib_to_gui(calib)
        plot_results.plot_structure_calib(self.structure_calib_plot_handles)
        self.streaker_calib_canvas.draw()
        return sc.fit_dicts

    def gap_reconstruction(self):
        self.clear_gap_recon_plots()

        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']

        tracker = self.get_tracker(saved_dict['meta_data_begin'])
        structure_calib_options = self.get_structure_calib_options()
        calibrator = calibration.StructureCalibrator(tracker, structure_calib_options, saved_dict)
        calib_dict = calibrator.calibrate_gap_and_struct_position()
        new_calib = calib_dict['calibration']
        self.calib_to_gui(new_calib)
        plot_results.plot_calib(calib_dict, self.gap_recon_fig, self.gap_recon_plot_handles)
        self.gap_recon_canvas.draw()

    def load_calibration(self):
        self.clear_calib_plots()
        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']
        self._analyze_streaker_calib(saved_dict)

    def obtain_reconstruction(self):
        n_images = int(self.ReconNumberImages.text())
        screen_dict = daq.get_images(self.screen, n_images, self.beamline, self.dry_run)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_data_%s.h5' % self.screen.replace('.','_')
        elog_text = 'Screen %s data taken' % self.screen
        self.elog_and_H5(elog_text, [], 'Screen data', basename, screen_dict)

    @property
    def structure_name(self):
        return self.StructureSelect.currentText()

    @property
    def beamline(self):
        return self.BeamlineSelect.currentText()

    @property
    def dry_run(self):
        return (self.DryRun.isChecked() or always_dryrun)

    @property
    def screen_center(self):
        return float(self.ScreenCenterCalib.text())*1e-6

    @property
    def screen(self):
        if self.dry_run:
            return 'simulation'
        else:
            return self.ScreenSelect.currentText()

    def obtainLasing(self, lasing_on_off):
        if lasing_on_off:
            n_images = int(self.LasingOnNumberImages.text())
        else:
            n_images = int(self.LasingOffNumberImages.text())

        image_dict = daq.get_images(self.screen, n_images, self.beamline, self.dry_run)
        date = datetime.now()
        screen_str = self.screen.replace('.','_')
        lasing_str = str(lasing_on_off)
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_%s_%s.h5' % (lasing_str, screen_str)
        if lasing_on_off:
            elog_text = 'Saved lasing ON'
        else:
            elog_text = 'Saved lasing OFF'
        filename = self.elog_and_H5(elog_text, [], 'Saved lasing images', basename, image_dict)
        if lasing_on_off:
            self.LasingOnDataLoad.setText(filename)
        else:
            self.LasingOffDataLoad.setText(filename)

    def obtainLasingOn(self):
        return self.obtainLasing(True)

    def obtainLasingOff(self):
        return self.obtainLasing(False)

    def reconstruct_all_lasing(self):
        self.clear_all_lasing_plots()

        pulse_energy = float(self.LasingEnergyInput.text())*1e-6

        file_on = self.LasingOnDataLoad.text()
        file_off = self.LasingOffDataLoad.text()
        lasing_off_dict = h5_storage.loadH5Recursive(file_off)
        lasing_on_dict = h5_storage.loadH5Recursive(file_on)

        las_rec_images = {}

        tracker = self.get_tracker(lasing_off_dict['meta_data_begin'])
        lasing_options = self.get_lasing_options()

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
        las_rec.plot(plot_handles=self.all_lasing_plot_handles)
        self.all_lasing_canvas.draw()

    def save_lasing_rec_data(self):
        if self.lasing_rec_dict is None:
            raise ValueError('No lasing reconstruction data to save')
        elog_text = 'Lasing reconstruction'
        elog_text +='\nComment: %s' % self.LasingElogComment.text()
        date = datetime.now()
        screen_str = self.screen.replace('.','_')
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_reconstruction_%s.h5' % screen_str
        filename = self.elog_and_H5(elog_text, self.lasing_figs, 'Lasing reconstruction', basename, self.lasing_rec_dict)
        self.ReconstructionDataLoad.setText(filename)

    def elog_and_H5(self, text, figs, title, basename, data_dict):

        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, data_dict)
        print('Saved %s' % filename)

        attachments = []
        for num, fig in enumerate(figs):
            fig_title = filename.replace('.h5', '_%i.png' % num)
            fig_filename = os.path.join(self.save_dir, fig_title)
            fig.savefig(fig_filename, bbox_inches='tight', pad_inches=0)
            print('Saved %s' % fig_filename)
            attachments.append(fig_filename)

        text += '\nData saved in %s' % filename
        text += '\nBeamline: %s' % self.beamline
        text += '\nStreaker: %s' % self.structure_name
        text += '\nScreen: %s' % self.screen

        if elog is None:
            print('Cannot save to ELOG')
            print('I would post:')
            print(text)
        elif self.ElogSaveCheck.isChecked():
            dict_att = {'Author': 'Application: PostUndulatorStreakerAnalysis', 'Application': 'PostUndulatorStreakerAnalysis', 'Category': 'Measurement', 'Title': title}
            self.logbook.post(text, attributes=dict_att, attachments=attachments)

            print('ELOG entry saved.')
        else:
            print('Save to ELOG is not checked in GUI')
        return filename

if __name__ == '__main__':
    def my_excepthook(type, value, tback):
        # log the exception here
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        print(type, value, tback)
    sys.excepthook = my_excepthook
    logger = logMsg.get_logger(config.logfile, 'PassiveWFMeasurement')

    app = QtWidgets.QApplication(sys.argv)
    window = StartMain(logger)
    window.show()
    app.exec_()
