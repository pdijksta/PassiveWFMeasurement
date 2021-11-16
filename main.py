import matplotlib.pyplot as plt; plt # Without this line, there is an error...
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib
matplotlib.use('Qt5Agg')

import sys
import os
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
from PassiveWFMeasurement import logMsg
from PassiveWFMeasurement import myplotstyle as ms

#TODO
#
# - add tilt option
# - handle feedback in user interface
# - detune undulator button
# - Plot centroid of forward propagated
# - Calibration based on TDC

#Problematic / cannot be done easily:
# - save BPM data also
# - One-sided plate
# - Update load blmeas (need bugfixes by Thomas)

# Probably fixed:
# - sort out daq pyscan_result_to_dict
# - debug delay after using BsreadPositioner or any pyscan
# - Dispersion (?)

# Not so important
# - add blmeas option to lasing rec
# - noise reduction from the image
# - uJ instead of True, False
# - non blocking daq
# - One-sided plate
# - streaker center calibration: repeat with one data point removed at one side
# - add info of beamsize with / without assumed screen resolution

# Done
# - Offset based on centroid, offset based on sizes (?)
# - Mean of square instead of square of mean of squareroot
# - Athos gas detector
# - Fix calibration - screen and streaker offset at same time
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
    from PassiveWFMeasurement import daq
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

def w2f(w):
    return float(w.text())

def w2i(w):
    return int(round(w2f(w)))

class StartMain(QtWidgets.QMainWindow, logMsg.LogMsgBase):

    charge_pv_text = 'Use PV'

    def __init__(self, logger=None):
        super(StartMain, self).__init__()
        uic.loadUi('gui.ui', self)
        self.logger = logger

        self.DoReconstruction.clicked.connect(self.reconstruct_current)
        self.SaveElogGUI.clicked.connect(self.elog_and_H5_button)
        self.CloseAll.clicked.connect(self.clear_rec_plots)
        self.FitStreaker.clicked.connect(self.daq_calibration)
        self.ClearStructureFitPlots.clicked.connect(self.clear_structure_fit_plots)
        self.GapCalibration.clicked.connect(self.calibrate_gap)
        self.ClearCalibPlots.clicked.connect(self.clear_structure_calib_plots)
        self.LoadFit.clicked.connect(self.load_structure_fit)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction)
        self.ObtainLasingOnData.clicked.connect(self.obtainLasingOn)
        self.ObtainLasingOffData.clicked.connect(self.obtainLasingOff)
        self.ReconstructLasing.clicked.connect(self.reconstruct_all_lasing)
        self.DestroyLasing.clicked.connect(self.destroy_lasing)
        self.RestoreLasing.clicked.connect(self.restore_lasing)
        self.TdcCalibration.clicked.connect(self.tdc_calibration)

        self.PlotResolution.clicked.connect(self.plot_resolution)

        self.BeamlineSelect.addItems(sorted(config.structure_names.keys()))
        self.StructureSelect.addItems(sorted(config.structure_parameters.keys(), reverse=True))

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
        structure_calib_file = default_dir + '2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
        screen_center = 898.02e-6
        structure_position0 = 364e-6
        delta_gap = -62e-6
        pulse_energy = 180e-6

        self.ScreenCenterCalib.setText('%i' % round(screen_center*1e6))
        self.StructureCenter.setText('%i' % round(structure_position0*1e6))
        self.StructureGapDelta.setText('%i' % round(delta_gap*1e6))
        self.LasingEnergyInput.setText('%i' % round(pulse_energy*1e6))

        self.ReconstructionDataLoad.setText(lasing_file_off)
        self.BunchLengthMeasFile.setText(bunch_length_meas_file)
        self.LasingOnDataLoad.setText(lasing_file_on)
        self.LasingOffDataLoad.setText(lasing_file_off)
        self.LoadCalibrationFilename.setText(structure_calib_file)
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
        self.StructCenterSearchLower.setText('%.3f' % (gs['delta_structure0_range'].min()*1e6))
        self.StructCenterSearchUpper.setText('%.3f' % (gs['delta_structure0_range'].max()*1e6))
        self.StructCenterSearchPoints.setText('%.3f' % len(gs['delta_structure0_range']))

        # Find beam position options
        self.FindBeamMaxIter.setText('%i' % fbs['max_iterations'])
        self.FindBeamExplorationRange.setText('%.3f' % (fbs['position_explore']*1e6))

        # Lasing options
        self.LasingReconstructionSliceFactor.setText('%i' % ls['slice_factor'])
        self.LasingNoiseCut.setText('%.4f' % ls['noise_cut'])
        self.LasingIntensityCut.setText('%.4f' % ls['subtract_quantile'])

        if elog is not None:
            self.logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning+data/', user='robot', password='robot')

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
        self.rec_plot_tab_index, self.rec_canvas = get_new_tab(self.reconstruction_fig, 'I rec.')

        self.structure_fit_fig, self.structure_fit_plot_handles = plot_results.structure_fit_figure()
        self.structure_fit_plot_tab_index, self.structure_fit_canvas = get_new_tab(self.structure_fit_fig, 'Gap fit')

        self.structure_calib_fig, self.structure_calib_plot_handles = plot_results.calib_figure()
        self.structure_calib_tab_index, self.structure_calib_canvas = get_new_tab(self.structure_calib_fig, 'Gap rec.')

        self.all_lasing_fig, self.all_lasing_plot_handles = plot_results.lasing_figure()
        self.all_lasing_tab_index, self.all_lasing_canvas = get_new_tab(self.all_lasing_fig, 'Lasing rec.')

        self.resolution_fig, self.resolution_plot_handles = plot_results.resolution_figure()
        self.resolution_tab_index, self.resolution_canvas = get_new_tab(self.resolution_fig, 'Res.')

        # Init ELOG
        self.elog_button_title = 'Empty'
        self.elog_button_figures = []
        self.elog_button_save_dict = None
        self.elog_button_save_name = 'Empty'

        self.undulator_pvs = {}
        self.lasing_undulator_vals = {}
        self.LasingStatus.setText('Undulator K values not stored')

        self.logMsg('Main window initialized')

    def clear_rec_plots(self):
        plot_results.clear_reconstruction(*self.reconstruction_plot_handles)
        self.rec_canvas.draw()

    def clear_structure_calib_plots(self):
        plot_results.clear_calib(*self.structure_calib_plot_handles)
        self.structure_calib_canvas.draw()

    def clear_structure_fit_plots(self):
        plot_results.clear_structure_fit(*self.structure_fit_plot_handles)
        self.structure_fit_canvas.draw()

    def clear_screen_plots(self):
        plot_results.clear_screen_calibration(*self.screen_calib_plot_handles)
        self.screen_calib_canvas.draw()

    def clear_all_lasing_plots(self):
        plot_results.clear_lasing_figure(*self.all_lasing_plot_handles)

    def gui_to_calib(self):
        return calibration.StructureCalibration(
                self.structure_name,
                w2f(self.ScreenCenterCalib)/1e6,
                w2f(self.StructureGapDelta)/1e6,
                w2f(self.StructureCenter)/1e6)

    def calib_to_gui(self, calib):
        self.ScreenCenterCalib.setText('%.3f' % (calib.screen_center*1e6))
        self.StructureGapDelta.setText('%.3f' % (calib.delta_gap*1e6))
        self.StructureCenter.setText('%.3f' % (calib.structure_position0*1e6))

    def get_tracker(self, meta_data, structure_name=None):
        if self.Charge.text() == self.charge_pv_text:
            force_charge = None
        else:
            force_charge = w2f(self.Charge)*1e-12
        n_particles = w2i(self.N_Particles)

        if structure_name is None:
            structure_name = self.structure_name

        tracker = tracking.Tracker(
                self.beamline,
                self.screen,
                structure_name,
                meta_data,
                self.gui_to_calib(),
                self.get_forward_options(),
                self.get_backward_options(),
                self.get_reconstruct_gauss_options(),
                self.get_beam_spec(),
                self.get_beam_optics(),
                self.get_find_beam_position_options(),
                force_charge,
                n_particles,
                self.logger)
        return tracker

    def plot_resolution(self):
        # Not ideal but ok for now
        bp_file = os.path.join(os.path.dirname(__file__), './example_current_profile.h5')
        bp_dict = h5_storage.loadH5Recursive(bp_file)
        gap = 10e-3
        beam_position = gap/2 - w2f(self.PlotResolutionDistance)*1e-6
        meta_data = daq.get_meta_data(self.screen, False, self.beamline)
        tracker = self.get_tracker(meta_data)
        beamprofile = beam_profile.BeamProfile(bp_dict['time_profile'], bp_dict['current'], tracker.energy_eV, tracker.total_charge, self.logger)
        res_dict = resolution.calc_resolution(beamprofile, gap, beam_position, tracker)

        plot_results.clear_resolution_figure(*self.resolution_plot_handles)
        resolution.plot_resolution(res_dict, *self.resolution_plot_handles)
        self.resolution_canvas.draw()
        self.tabWidget.setCurrentIndex(self.resolution_tab_index)

        self.elog_button_title = 'Resolution calculated'
        self.elog_button_figures = [self.resolution_fig]
        self.elog_button_save_dict = res_dict
        self.elog_button_save_name = '%s_resolution' % self.structure_name

        elog_text = 'Resolution calculated for'
        elog_text += '\ngap: %.3f mm' % (gap*1e3)
        elog_text += '\nbeam position: %.3f mm' % (beam_position*1e3)
        elog_text += '\nCurrent profile from: %s' % bp_file
        self.ELOGtextAuto.insertPlainText(elog_text)

    def get_forward_options(self):
        outp = config.get_default_forward_options()
        outp['screen_bins'] = w2i(self.ScreenBins)
        outp['screen_smoothen'] = w2f(self.ScreenSmoothen)*1e-6
        outp['screen_cutoff'] = w2f(self.ScreenCutoff)
        outp['len_screen'] = w2i(self.ScreenSize)
        return outp

    def get_backward_options(self):
        outp = config.get_default_backward_options()
        outp['profile_cutoff'] = w2f(self.ProfileCutoff)
        outp['len_profile'] = w2i(self.ProfileSize)
        outp['profile_smoothen'] = w2f(self.ProfileSmoothen)*1e-15
        return outp

    def get_reconstruct_gauss_options(self):
        outp = config.get_default_reconstruct_gauss_options()
        outp['sig_t_range'] = np.array([w2f(self.SigTfsMin)*1e-15, w2f(self.SigTfsMax)*1e-15])
        outp['max_iterations'] = w2i(self.GaussReconMaxIter)
        outp['gauss_profile_t_range'] = w2f(self.ProfileExtent)*1e-15
        return outp

    def get_beam_spec(self):
        outp = config.get_default_beam_spec()
        outp['nemitx'] = w2f(self.TransEmittanceX)*1e-9
        outp['nemity'] = w2f(self.TransEmittanceY)*1e-9
        return outp

    def get_beam_optics(self):
        outp = config.get_default_optics(self.beamline)
        outp['betax'] = w2f(self.BetaX)
        outp['alphax'] = w2f(self.AlphaX)
        outp['betay'] = w2f(self.BetaY)
        outp['alphay'] = w2f(self.AlphaY)
        return outp

    def get_find_beam_position_options(self):
        outp = config.get_default_find_beam_position_options()
        outp['position_explore'] = w2f(self.FindBeamExplorationRange)*1e-6
        outp['max_iterations'] = w2i(self.FindBeamMaxIter)
        return outp

    def get_structure_calib_options(self):
        outp = config.get_default_structure_calibrator_options()
        _lower = w2f(self.DeltaGapRangeLower)*1e-6
        _upper = w2f(self.DeltaGapRangeUpper)*1e-6
        _step = w2f(self.DeltaGapRangeStep)*1e-6
        outp['delta_gap_scan_range'] = np.arange(_lower, _upper, _step)
        _lower = w2f(self.DeltaGapSearchLower)*1e-6
        _upper = w2f(self.DeltaGapSearchUpper)*1e-6
        _points = w2i(self.DeltaGapSearchPoints)
        outp['delta_gap_range'] = np.linspace(_lower, _upper, _points)
        _lower = w2f(self.StructCenterSearchLower)*1e-6
        _upper = w2f(self.StructCenterSearchUpper)*1e-6
        _points = w2i(self.StructCenterSearchPoints)
        outp['delta_structure0_range'] = np.linspace(_lower, _upper, _points)
        return outp

    def get_lasing_options(self):
        outp = config.get_default_lasing_options()
        outp['slice_factor'] = w2i(self.LasingReconstructionSliceFactor)
        outp['noise_cut'] = w2f(self.LasingNoiseCut)
        outp['subtract_quantile'] = w2f(self.LasingIntensityCut)
        return outp

    def reconstruct_current(self):
        self.clear_rec_plots()
        filename = self.ReconstructionDataLoad.text().strip()

        screen_data = h5_storage.loadH5Recursive(filename)
        if 'meta_data' in screen_data:
            meta_data = screen_data['meta_data']
        elif 'meta_data_begin' in screen_data:
            meta_data = screen_data['meta_data_begin']
        else:
            self.logMsg('Problems with screen data meta data. Available keys: %s' % screen_data.keys(), 'E')
            raise ValueError

        tracker = self.get_tracker(meta_data)

        if self.ShowBlmeasCheck.isChecked():
            blmeas_file = self.BunchLengthMeasFile.text()
            time_range = tracker.reconstruct_gauss_options['gauss_profile_t_range']
            blmeas_profile = beam_profile.profile_from_blmeas(blmeas_file, time_range, tracker.total_charge, tracker.energy_eV, 2e-2)
        else:
            blmeas_profile = None

        x_axis, proj = data_loader.screen_data_to_median(screen_data['pyscan_result'])
        x_axis = x_axis - tracker.calib.screen_center
        meas_screen = beam_profile.ScreenDistribution(x_axis, proj, total_charge=tracker.total_charge)
        current_rec_dict = tracker.reconstruct_profile_Gauss(meas_screen, output_details=True)

        plot_results.plot_rec_gauss(current_rec_dict, plot_handles=self.reconstruction_plot_handles, blmeas_profiles=[blmeas_profile])

        self.logMsg('Current profile reconstructed.')

        self.rec_canvas.draw()
        self.tabWidget.setCurrentIndex(self.rec_plot_tab_index)

        self.elog_button_title = 'Current profile reconstructed'
        self.elog_button_figures = [self.reconstruction_fig]
        self.elog_button_save_dict = current_rec_dict
        self.elog_button_save_name = '%s_current_profile_reconstruction' % self.structure_name

        elog_text = 'Current profile reconstructed'
        elog_text += '\nstructure:  %s' % tracker.structure_name
        self.ELOGtextAuto.insertPlainText(elog_text)

    @property
    def save_dir(self):
        return os.path.expanduser(self.SaveDir.text())

    def daq_calibration(self):
        self.logMsg('DAQ for calibration started.')
        self.clear_structure_fit_plots()
        start, stop, step = w2f(self.Range1Begin), w2f(self.Range1Stop), w2i(self.Range1Step)
        range1 = np.linspace(start, stop, step)
        start, stop, step= w2f(self.Range2Begin), w2f(self.Range2Stop), w2i(self.Range2Step)
        range2 = np.linspace(start, stop, step)
        range_ = np.concatenate([range1, [0], range2])*1e-3 # Convert mm to m
        range_.sort()
        range_ = np.unique(range_)

        n_images = w2i(self.CalibrateStreakerImages)

        if daq is None:
            raise ImportError('Daq not available')

        result_dict = daq.data_structure_offset(self.structure_name, range_, self.screen, n_images, self.dry_run, self.beamline)

        try:
            fit_dicts = self._analyze_structure_fit(result_dict)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_') +'Calibration_data_%s.h5' % self.structure_name.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, result_dict)
            self.logMsg('Saved structure calibration data %s' % filename)
            raise

        full_dict = {
                'fit_results': fit_dicts,
                'raw_data': result_dict,
                }

        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Calibration_%s.h5' % self.structure_name.replace('.','_')

        calib = fit_dicts['centroid']['calibration']
        elog_text = 'Streaker calibration structure %s\nCenter: %i um' % (self.structure_name, calib.structure_position0*1e6)
        filename = self.elog_and_H5_auto(elog_text, [self.structure_fit_fig], 'Streaker center calibration', basename, full_dict)
        self.LoadCalibrationFilename.setText(filename)
        self.tabWidget.setCurrentIndex(self.structure_fit_plot_tab_index)
        self.logMsg('DAQ for calibration ended.')

    def _analyze_structure_fit(self, result_dict):
        tracker = self.get_tracker(result_dict['meta_data_begin'])
        structure_calib_options = self.get_structure_calib_options()

        sc = calibration.StructureCalibrator(tracker, structure_calib_options, result_dict, self.logger)

        sc.fit()
        if self.ForwardBlmeasCheck.isChecked():
            blmeasfile = self.ForwardBlmeasFilename.text()
            tt_range = tracker.reconstruct_gauss_options['gauss_profile_t_range']
            bp = beam_profile.profile_from_blmeas(blmeasfile, tt_range, tracker.total_charge, tracker.energy_eV, 5e-2, len_profile=tracker.backward_options['len_profile'])

            sc.forward_propagate(bp)

        calib = sc.fit_dicts['centroid']['calibration']
        self.calib_to_gui(calib)
        plot_results.plot_structure_position0_fit(sc.fit_dicts, self.structure_fit_plot_handles)
        self.structure_fit_canvas.draw()
        self.tabWidget.setCurrentIndex(self.structure_fit_plot_tab_index)
        return sc.fit_dicts

    def calibrate_gap(self):
        self.logMsg('Start gap calibration')
        self.clear_structure_calib_plots()

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
        plot_results.plot_calib(calib_dict, self.structure_calib_fig, self.structure_calib_plot_handles)
        self.structure_calib_canvas.draw()
        self.tabWidget.setCurrentIndex(self.structure_calib_tab_index)

        self.elog_button_title = 'Structure gap and center calibrated'
        self.elog_button_figures = [self.structure_calib_fig]
        self.elog_button_save_dict = calib_dict
        self.elog_button_save_name = '%s_calibration' % self.structure_name

        elog_text = 'Structure gap and center calibrated'
        elog_text += '\nstructure: %s' % tracker.structure_name
        elog_text += '\ndelta gap: %.3f um' % (new_calib.delta_gap*1e6)
        elog_text += '\nstructure center: %.3f um' % (new_calib.structure_position0*1e6)
        elog_text += '\nscreen center: %.3f um' % (new_calib.screen_center*1e6)
        self.ELOGtextAuto.insertPlainText(elog_text)

        self.logMsg('End gap calibration')

    def tdc_calibration(self):
        meta_data = daq.get_meta_data(self.screen, self.dry_run, self.beamline)
        tracker = self.get_tracker(meta_data)
        tt_range = tracker.reconstruct_gauss_options['gauss_profile_t_range']
        blmeasfile = self.ForwardBlmeasFilename.text()
        bp = beam_profile.profile_from_blmeas(blmeasfile, tt_range, tracker.total_charge, tracker.energy_eV, 5e-2, len_profile=tracker.backward_options['len_profile'])

        # TODO
        calibration.tdc_calibration(tracker, bp, )

    def load_structure_fit(self):
        self.clear_structure_fit_plots()
        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']
        self._analyze_structure_fit(saved_dict)

    def obtain_reconstruction(self):
        n_images = w2i(self.ReconNumberImages)
        screen_dict = daq.get_images(self.screen, n_images, self.beamline, self.dry_run)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_data_%s.h5' % self.screen.replace('.','_')
        elog_text = 'Screen %s data taken' % self.screen
        fig, _ = plot_results.plot_simple_daq(screen_dict)
        self.elog_and_H5_auto(elog_text, [fig], 'Screen data', basename, screen_dict)

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
    def screen(self):
        return self.ScreenSelect.currentText()

    def obtainLasing(self, lasing_on_off):
        if lasing_on_off:
            n_images = w2i(self.LasingOnNumberImages)
        else:
            n_images = w2i(self.LasingOffNumberImages)

        image_dict = daq.get_images(self.screen, n_images, self.beamline, self.dry_run)
        date = datetime.now()
        screen_str = self.screen.replace('.','_')
        lasing_str = str(lasing_on_off)
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_%s_%s.h5' % (lasing_str, screen_str)
        if lasing_on_off:
            elog_text = 'Saved lasing ON'
        else:
            elog_text = 'Saved lasing OFF'
        fig, _ = plot_results.plot_simple_daq(image_dict)
        filename = self.elog_and_H5_auto(elog_text, [fig], 'Saved lasing images', basename, image_dict)
        if lasing_on_off:
            self.LasingOnDataLoad.setText(filename)
        else:
            self.LasingOffDataLoad.setText(filename)

    def obtainLasingOn(self):
        return self.obtainLasing(True)

    def obtainLasingOff(self):
        return self.obtainLasing(False)

    def reconstruct_all_lasing(self):
        self.logMsg('Lasing reconstruction start')
        self.clear_all_lasing_plots()

        pulse_energy = w2f(self.LasingEnergyInput)*1e-6

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
        result_dict = las_rec.get_result_dict()
        plot_results.plot_lasing(result_dict, plot_handles=self.all_lasing_plot_handles)
        self.all_lasing_canvas.draw()
        self.tabWidget.setCurrentIndex(self.all_lasing_tab_index)

        self.elog_button_title = 'FEL power profile reconstructed'
        self.elog_button_figures = [self.all_lasing_fig]
        self.elog_button_save_dict = result_dict
        self.elog_button_save_name = '%s_FEL_power_profile_reconstruction' % self.structure_name

        elog_text = 'FEL power profile reconstructed'
        elog_text += '\nstructure:  %s' % tracker.structure_name
        self.ELOGtextAuto.insertPlainText(elog_text)

        self.logMsg('Lasing reconstruction end')

    def elog_and_H5_auto(self, text, figs, title, basename, data_dict):

        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, data_dict)
        self.logMsg('Saved %s' % filename)

        attachments = []
        for num, fig in enumerate(figs):
            fig_title = filename.replace('.h5', '_%i.png' % num)
            fig_filename = os.path.join(self.save_dir, fig_title)
            fig.savefig(fig_filename, bbox_inches='tight', pad_inches=0)
            self.logMsg('Saved %s' % fig_filename)
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
            self.logMsg('Save to ELOG is not checked in GUI. Not saving.')
        return filename

    def elog_and_H5_button(self):
        auto_text = self.ELOGtextAuto.toPlainText()
        manual_text = self.ELOGtextInput.toPlainText()
        title = self.elog_button_title
        figures = self.elog_button_figures
        save_dict = self.elog_button_save_dict
        save_name = self.elog_button_save_name
        if not save_name.endswith('.h5'):
            save_name += '.h5'
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_PassiveReconstruction_')
        save_name = basename + save_name
        h5_filename = os.path.join(self.SaveDir.text(), save_name)
        h5_storage.saveH5Recursive(h5_filename, save_dict)
        auto_text += '\nData saved in %s' % h5_filename
        attachments = []

        for n_fig, fig in enumerate(figures, 1):
            fig_filename = h5_filename.replace('.h5', '_%i.png' % n_fig)
            fig.savefig(fig_filename, bbox_inches='tight', pad_inches=0)
            attachments.append(fig_filename)
            auto_text += '\nFigure %i saved in %s' % (n_fig, fig_filename)

        dict_att = {'Author': 'Application: PostUndulatorStreakerAnalysis', 'Application': 'PostUndulatorStreakerAnalysis', 'Category': 'Measurement', 'Title': title}
        if elog is None:
            self.logMsg('I would post to ELOG:')
            self.logMsg(auto_text)
            self.logMsg(manual_text)
        else:
            self.logbook.post(auto_text+'\n'+manual_text, attributes=dict_att, attachments=attachments)
            self.logMsg('Elog entry generated')

    def destroy_lasing(self):
        beamline = self.beamline
        self.undulator_pvs[beamline], self.lasing_undulator_vals[beamline], _ = daq.destroy_lasing(self.beamline, self.dry_run)
        self.LasingStatus.setText('Lasing in %s destroyed' % beamline)

    def restore_lasing(self):
        beamline = self.beamline
        pvs = self.undulator_pvs[beamline]
        vals = self.lasing_undulator_vals[beamline]
        daq.restore_lasing(pvs, vals, self.dry_run)
        self.LasingStatus.setText('Lasing in %s restored' % beamline)


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

