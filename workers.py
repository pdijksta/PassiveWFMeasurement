import traceback
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class WorkerBase(QObject):
    """
    Wraps a function call such that it can be used by QThread.
    Inherit this class and implement the "func" method.
    """
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    new_result = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        QObject.__init__(self)
        self.args = args
        self.kwargs = kwargs
        self.outp = None
        self.error = False
        self.abort = False
        self.pause = False
        self.result_dict = {}

    def run(self):
        try:
            self.outp = self.func(*self.args, **self.kwargs)
        except Exception:
            self.error = True
            print('Worker failed with following error message:')
            print(traceback.format_exc())
        finally:
            self.finished.emit()

def threaded_func(parent, func_worker, start_funcs, finish_funcs, progress_funcs, result_funcs):

    func_thread = QThread(parent=parent)
    func_worker = func_worker
    func_worker.moveToThread(func_thread)

    for func in start_funcs:
        func_thread.started.connect(func)
    func_thread.started.connect(func_worker.run)

    for func in finish_funcs:
        func_worker.finished.connect(func)

    for func in progress_funcs:
        func_worker.progress.connect(func)

    for func in result_funcs:
        func_worker.new_result.connect(func)

    func_worker.finished.connect(func_thread.quit)
    func_worker.finished.connect(func_thread.wait)
    func_worker.finished.connect(func_thread.deleteLater)

    func_thread.start()
    return func_thread, func_worker

