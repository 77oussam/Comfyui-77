from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from processing import resize_image

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error: tuple (exctype, value, traceback.format_exc())
    - result: object data returned from processing, anything
    - progress: int indicating % progress
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str)

class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    """
    def __init__(self, source_path, output_path, target_size, unit, dpi, export_format, quality):
        super().__init__()
        self.signals = WorkerSignals()
        
        # Store constructor arguments
        self.source_path = source_path
        self.output_path = output_path
        self.target_size = target_size
        self.unit = unit
        self.dpi = dpi
        self.export_format = export_format
        self.quality = quality

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        try:
            success, message = resize_image(
                self.source_path,
                self.output_path,
                self.target_size,
                self.unit,
                self.dpi,
                self.export_format,
                self.quality
            )
            if success:
                self.signals.result.emit(message)
            else:
                self.signals.error.emit(message)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
