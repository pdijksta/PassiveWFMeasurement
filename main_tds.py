import sys

import PyQt5
import PyQt5.uic
import PyQt5.QtWidgets

from PassiveWFMeasurement import logMsg
from PassiveWFMeasurement import config
from PassiveWFMeasurement import config_tds

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
    haselog = True
    logMsg.logMsg('Elog available', logger=logger)
except ImportError:
    elog = None
    haselog = False
    logMsg.logMsg('ELOG unavailable', logger=logger)

class StartMain(PyQt5.QtWidgets.QMainWindow, logMsg.LogMsgBase):
    def __init__(self, logger=None):
        PyQt5.QtWidgets.QMainWindow.__init__(self)
        PyQt5.uic.loadUi('gui_tds.ui', self)
        self.logger = logger

        # Fill GUI widgets with default / init values
        if always_dryrun:
            self.config_dry_run.setChecked(True)
            self.config_dry_run.setEnabled(False)
        if not haselog:
            self.config_elog.setChecked(False)
            self.config_elog.setEnabled(False)
        self.main_deltaK.setValue(config_tds.default_deltaK)



        # Connect buttons
        self.perform_calibration.clicked.connect(self.calibrate)
        self.config_type.currentTextChanged.connect(self.activate_tds_select)
        self.settings_pulse_energy_input.currentTextChanged.connect(self.activate_pulse_energy)

        # Run default functions
        self.tabWidget.setCurrentIndex(0)
        self.activate_tds_select()
        self.activate_pulse_energy()

        print(self.get_gui_state())

    def calibrate(self):
        raise NotImplementedError

    def activate_tds_select(self):
        self.config_tds_device.setEnabled(self.config_type.currentText() == 'TDS')

    def activate_pulse_energy(self):
        self.pulse_energy.setEnabled(self.settings_pulse_energy_input.currentText() == 'Manual input')

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
        outp['config']['beam_monitor'] = self.config_beam_monitor.currentText()
        outp['config']['elog'] = self.config_beam_monitor.currentText()
        outp['config']['save_dir'] = self.config_savedir.text()

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

