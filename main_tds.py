import sys

import PyQt5
import PyQt5.uic
import PyQt5.QtWidgets

from PassiveWFMeasurement import logMsg
from PassiveWFMeasurement import config

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

