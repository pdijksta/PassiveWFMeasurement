import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt; plt # Without this line, there is an error...
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib
matplotlib.use('Qt5Agg')

import PyQt5
import PyQt5.uic
import PyQt5.QtWidgets

from PassiveWFMeasurement import logMsg
from PassiveWFMeasurement import lasing
from PassiveWFMeasurement import h5_storage
from PassiveWFMeasurement import plot_results
from PassiveWFMeasurement import config
from PassiveWFMeasurement import config_tds
from PassiveWFMeasurement import workers

if __name__ == '__main__':
    logger = logMsg.get_logger(config.logfile, 'TDSmeasurement')
else:
    logger = None

try:
    from PassiveWFMeasurement import daq
    always_dryrun = False
    logMsg.logMsg('Daq available', logger=logger)
except ImportError:
    always_dryrun = True
    daq = None
    logMsg.logMsg('Daq unavailable. Always dry_run True', logger=logger)

try:
    import elog
    logMsg.logMsg('Elog available', logger=logger)
except ImportError:
    elog = None
    logMsg.logMsg('ELOG unavailable', logger=logger)

class StartMain(PyQt5.QtWidgets.QMainWindow, logMsg.LogMsgBase):
    def __init__(self, logger=None):
        PyQt5.QtWidgets.QMainWindow.__init__(self)
        PyQt5.uic.loadUi('gui_tds.ui', self)
        self.logger = logger
        self.beamline = 'Athos post-undulator'

        # Fill GUI widgets with default / init values
        if always_dryrun:
            self.config_dry_run.setChecked(True)
            self.config_dry_run.setEnabled(False)
        if elog:
            self.logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning+data/', user='robot', password='robot')
        else:
            self.config_elog.setChecked(False)
            self.config_elog.setEnabled(False)
        self.main_deltaK.setValue(config_tds.default_deltaK)
        self.tabWidget.setCurrentIndex(0)
        self.activate_tds_select()
        self.activate_pulse_energy()
        self.default_savedir_text = 'Folder of the day on /sf/data/measurements'
        if os.path.isdir('/sf/data/measurements/'):
            self.config_savedir.setText(self.default_savedir_text)
        else:
            self.config_savedir.setText('./')

        # Connect buttons
        self.perform_calibration.clicked.connect(self.calibrate)
        self.config_type.currentTextChanged.connect(self.activate_tds_select)
        self.settings_pulse_energy_input.currentTextChanged.connect(self.activate_pulse_energy)
        self.ObtainLasingOnData.clicked.connect(self.obtainLasingOn)
        self.ObtainLasingOffData.clicked.connect(self.obtainLasingOff)
        self.SelectLasingOff.clicked.connect(self.select_file(self.main_filename_off))
        self.SelectLasingOn.clicked.connect(self.select_file(self.main_filename_on))
        self.DoAnalysis.clicked.connect(self.do_analysis)

    def select_file(self, widget):
        def f():
            QFileDialog = PyQt5.QtWidgets.QFileDialog
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", self.save_dir, "Hdf5 files (*.h5);;All Files (*)", options=options)
            if filename:
                widget.setText(filename)
        return f

    @property
    def beam_monitor(self):
        return self.config_beam_monitor.currentText()

    @property
    def save_dir(self):
        return self.config_savedir.text()

    @property
    def dry_run(self):
        return self.config_dry_run.isChecked()

    def calibrate(self):
        raise NotImplementedError

    def activate_tds_select(self):
        self.config_tds_device.setEnabled(self.config_type.currentText() == 'TDS')

    def activate_pulse_energy(self):
        self.pulse_energy.setEnabled(self.settings_pulse_energy_input.currentText() == 'Manual input')

    def obtainImages(self, lasing_on_off):
        n_images = self.config_number_images.value()

        dry_run = self.dry_run
        image_dict = daq.get_images(self.beam_monitor, n_images, self.beamline, dry_run)
        date = datetime.now()
        screen_str = self.beam_monitor.replace('.','_')
        lasing_str = {'True': 'enabled', 'False': 'disabled'}[lasing_on_off]
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_%s_%s.h5' % (lasing_str, screen_str)
        if lasing_on_off:
            elog_text = 'Saved lasing ON'
        else:
            elog_text = 'Saved lasing OFF'
        dim = config.structure_dimensions[self.structure_name]
        fig, _ = plot_results.plot_simple_daq(image_dict, dim)
        filename = self.elog_and_H5_auto(elog_text, [fig], 'Saved lasing images', basename, image_dict, dry_run)
        if lasing_on_off:
            self.LasingOnDataLoad.setText(filename)
        else:
            self.LasingOffDataLoad.setText(filename)

    def obtainLasingOn(self):
        return self.obtainImages(True)

    def obtainLasingOff(self):
        return self.obtainImages(False)

    def get_gui_state(self):
        outp = {
                'config': {},
                'calibration': {},
                'main': {},
                'settings': {},
                }

        outp['config']['type'] = self.config_type.currentText()
        outp['config']['tds_device'] = self.config_tds_device.currentText()
        outp['config']['n_images'] = self.config_number_images.value()
        outp['config']['beam_monitor'] = self.beam_monitor
        outp['config']['elog'] = self.config_beam_monitor.currentText()
        outp['config']['save_dir'] = self.save_dir

        outp['calibration']['value'] = self.calibration_value.value() * (1e-6/1e-15)
        outp['calibration']['info'] = self.calibration_info.text()

        outp['main']['filename_off'] = self.main_filename_off.text()
        outp['main']['filename_on'] = self.main_filename_on.text()
        outp['main']['deltaK'] = self.main_deltaK.value()
        outp['main']['immediate_analysis'] = self.main_immediate_analysis.isChecked()

        outp['settings']['pulse_energy_input'] = self.settings_pulse_energy_input.currentText()
        outp['settings']['pulse_energy'] = self.pulse_energy.value() * 1e-6
        outp['settings']['time_limit_check'] = self.TimeLimitCheck.isChecked()
        outp['settings']['time_limit1'] = self.TimeLimit1.value()*1e-15
        outp['settings']['time_limit2'] = self.TimeLimit2.value()*1e-15
        outp['settings']['energy_limit_check'] = self.EnergyLimitCheck.isChecked()
        outp['settings']['energy_limit1'] = self.EnergyLimit1.value()*1e6
        outp['settings']['energy_limit2'] = self.EnergyLimit2.value()*1e6
        outp['settings']['current_cutoff'] = self.CurrentCutoff.value()*1e3
        outp['settings']['pixel_per_slice'] = self.pixel_per_slice.value()

        return outp

    def elog_and_H5_auto(self, text, figs, title, basename, data_dict, dry_run):
        if dry_run:
            basename.replace('.h5', '_dry_run.h5')
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
        text += '\nStructure: %s' % self.structure_name
        text += '\nScreen: %s' % self.beam_monitor

        if elog is None:
            self.logMsg('Cannot save to ELOG', 'W')
            self.logMsg('I would post:')
            self.logMsg(text)
        elif self.config_elog.isChecked():
            dict_att = {'Author': 'Application LinearLasingAnalysis', 'Application': 'LinearLasingAnalysis', 'Category': 'Measurement', 'Title': title}
            self.logbook.post(text, attributes=dict_att, attachments=attachments)

            self.logMsg('ELOG entry saved.')
        else:
            self.logMsg('Save to ELOG is not checked in GUI. Not saving.')
        return filename

    def destroy_lasing(self):
        beamline = self.beamline
        delta_k = self.main_deltaK.value()
        self.undulator_pvs[beamline], self.lasing_undulator_vals[beamline], new_vals = daq.destroy_lasing(self.beamline, self.dry_run, delta_k)
        delta_k = new_vals - self.lasing_undulator_vals[beamline]
        self.logMsg('Lasing destroyed. Delta K values: %s' % delta_k)

    def restore_lasing(self):
        beamline = self.beamline
        pvs = self.undulator_pvs[beamline]
        vals = self.lasing_undulator_vals[beamline]
        daq.restore_lasing(pvs, vals, self.dry_run)
        self.logMsg('Lasing restored: Undulator K values: %s' % vals)

    def do_analysis(self):
        self.logMsg('Lasing reconstruction start')
        #print(time.time())
        self.all_lasing_fig, self.all_lasing_subplots = plot_results.lasing_figure()
        canvas = FigureCanvasQTAgg(self.all_lasing_fig)
        toolbar = NavigationToolbar2QT(canvas, self)

        # Delete everything in layout
        for i in reversed(range(self.plotspace.count())):
            widget = self.plotspace.itemAt(i).widget()
            self.plotspace.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        self.plotspace.addWidget(canvas)
        self.plotspace.addWidget(toolbar)

        if self.settings_pulse_energy_input.currentIndex == 1:
            pulse_energy = self.pulse_energy.value()*1e-6
        else:
            pulse_energy = None

        file_on = self.main_filename_on.text().strip()
        file_off = self.main_filename_off.text().strip()
        lasing_off_dict = h5_storage.loadH5Recursive(file_off)
        lasing_on_dict = h5_storage.loadH5Recursive(file_on)

        lasing_options = self.get_lasing_options()
        result_dict = lasing.obtain_lasing(tracker, lasing_off_dict, lasing_on_dict, lasing_options, pulse_energy)['result_dict']
        plot_results.plot_lasing(result_dict, plot_handles=(self.all_lasing_fig, self.all_lasing_subplots))
        self.all_lasing_canvas.draw()
        self.tabWidget.setCurrentIndex(self.all_lasing_tab_index)

        self.elog_button_title = 'FEL power profile reconstructed'
        self.elog_button_figures = [self.all_lasing_fig]
        self.elog_button_save_dict = {
                'main_result': result_dict['lasing_dict'],
                'phase_spaces': {
                    'lasing_on': result_dict['images_on']['tE_images'],
                    'lasing_off': result_dict['images_off']['tE_images'],
                    },
                'other_results': {'all_slice_dict': result_dict['all_slice_dict'], 'mean_slice_dict': result_dict['mean_slice_dict']},
                'Input': {'pulse_energy': pulse_energy, 'file_on': file_on, 'file_off': file_off, 'lasing_options': lasing_options},
                'Calibration': self.gui_to_calib().to_dict_custom(),
                }

        self.elog_button_save_name = '%s_FEL_power_profile_reconstruction.h5' % self.structure_name

        elog_text = 'FEL power profile reconstructed'
        elog_text += '\nstructure:  %s' % tracker.structure_name
        elog_text += '\nraw data lasing ON:  %s' % file_on
        elog_text += '\nraw data lasing OFF:  %s' % file_off
        self.setElogAutoText(elog_text)

        self.logMsg('Lasing reconstruction end')
        #print(time.time())
        #print(tracking.forward_ctr, tracking.backward_ctr, tracking.rec_ctr)



if __name__ == '__main__':
    def my_excepthook(type, value, tback):
        # log the exception here
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        logMsg.logMsg(str((type, value, tback)), logger, 'E')
    sys.excepthook = my_excepthook

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    window = StartMain(logger)
    window.show()
    app.exec_()

