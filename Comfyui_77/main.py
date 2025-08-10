import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
    QListWidget, QHBoxLayout, QLineEdit, QComboBox, QFileDialog, QGroupBox,
    QSpinBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import QThreadPool, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from worker import Worker

class DraggableListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_window = parent # Store reference to MainWindow

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data and mime_data.hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        mime_data = event.mimeData()
        if not mime_data:
            return
        files = [u.toLocalFile() for u in mime_data.urls()]
        valid_files = []
        for f in files:
            if os.path.isdir(f):
                for root, _, filenames in os.walk(f):
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            valid_files.append(os.path.join(root, filename))
            elif f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                valid_files.append(f)
        self.addItems(valid_files)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High-Performance Image Resizer")
        self.setGeometry(100, 100, 1024, 768)
        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for image list and preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 2)

        # Right panel for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 1)

        # --- Left Panel: Image List & Status Log ---
        image_group = QGroupBox("Image Queue")
        image_layout = QVBoxLayout()
        
        self.image_list_widget = DraggableListWidget(self) # Use custom widget
        image_layout.addWidget(self.image_list_widget)

        add_images_button = QPushButton("Add Images...")
        add_images_button.clicked.connect(self.add_images)
        image_layout.addWidget(add_images_button)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)

        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        log_layout.addWidget(self.status_log)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)

        # --- Right Panel: Controls ---
        sizes_group = QGroupBox("1. Output Sizes")
        sizes_layout = QVBoxLayout()
        self.output_sizes_list = QListWidget()
        sizes_layout.addWidget(self.output_sizes_list)
        add_size_layout = QHBoxLayout()
        self.size_input: QLineEdit = QLineEdit()
        self.size_input.setPlaceholderText("e.g., 1024")
        self.unit_combo: QComboBox = QComboBox()
        self.unit_combo.addItems(["px", "cm", "in"])
        add_size_button = QPushButton("Add")
        add_size_button.clicked.connect(self.add_output_size)
        add_size_layout.addWidget(self.size_input)
        add_size_layout.addWidget(self.unit_combo)
        add_size_layout.addWidget(add_size_button)
        sizes_layout.addLayout(add_size_layout)
        sizes_group.setLayout(sizes_layout)
        right_layout.addWidget(sizes_group)

        export_group = QGroupBox("2. Export Options")
        export_layout = QVBoxLayout()
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG", "WEBP"])
        format_layout.addWidget(self.format_combo)
        export_layout.addLayout(format_layout)
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_spinbox = QSpinBox()
        self.quality_spinbox.setRange(1, 100)
        self.quality_spinbox.setValue(90)
        quality_layout.addWidget(self.quality_spinbox)
        export_layout.addLayout(quality_layout)
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)

        export_action_group = QGroupBox("3. Start Export")
        export_action_layout = QVBoxLayout()
        export_button = QPushButton("Start Batch Export")
        export_button.clicked.connect(self.start_export)
        export_action_layout.addWidget(export_button)
        self.progress_bar = QProgressBar()
        export_action_layout.addWidget(self.progress_bar)
        export_action_group.setLayout(export_action_layout)
        right_layout.addWidget(export_action_group)
        right_layout.addStretch()

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if files:
            self.image_list_widget.addItems(files)

    def add_output_size(self):
        size_text = ""
        unit = ""
        if self.size_input:
            size_text = self.size_input.text()
        if self.unit_combo:
            unit = self.unit_combo.currentText()

        if size_text.isdigit() and int(size_text) > 0:
            self.output_sizes_list.addItem(f"{size_text} {unit}")
            if self.size_input: # Clear only if it exists
                self.size_input.clear()
        else:
            self.log_status("Invalid size. Please enter a positive number.", "error")

    def start_export(self):
        image_paths = []
        for i in range(self.image_list_widget.count()):
            item = self.image_list_widget.item(i)
            if item is not None:
                image_paths.append(item.text())

        output_sizes = []
        for i in range(self.output_sizes_list.count()):
            item = self.output_sizes_list.item(i)
            if item is not None:
                output_sizes.append(item.text())
        
        if not image_paths or not output_sizes:
            self.log_status("Please add images and at least one output size.", "error")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        self.jobs_to_run = len(image_paths) * len(output_sizes)
        self.jobs_completed = 0
        self.progress_bar.setValue(0)

        for img_path in image_paths:
            for size_str in output_sizes:
                self.create_worker(img_path, size_str, output_dir)

    def create_worker(self, img_path, size_str, output_dir):
        size, unit = size_str.split()
        export_format = self.format_combo.currentText()
        quality = self.quality_spinbox.value()
        
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        output_filename = f"{base_name}_{size}{unit}.{export_format.lower()}"
        output_path = os.path.join(output_dir, output_filename)

        worker = Worker(img_path, output_path, int(size), unit, 72, export_format, quality)
        worker.signals.result.connect(self.log_status)
        worker.signals.error.connect(self.log_status)
        worker.signals.finished.connect(self.update_progress)
        
        self.threadpool.start(worker)

    def log_status(self, message, level="info"):
        if level == "error":
            self.status_log.append(f"<font color='red'>ERROR: {message}</font>")
        else:
            self.status_log.append(message)

    def update_progress(self):
        self.jobs_completed += 1
        progress = int((self.jobs_completed / self.jobs_to_run) * 100)
        self.progress_bar.setValue(progress)
        if self.jobs_completed == self.jobs_to_run:
            self.log_status("Batch export finished!", "info")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
