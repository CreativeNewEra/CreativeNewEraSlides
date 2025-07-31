from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QTextEdit, QLabel, QSpinBox,
    QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QPushButton, QProgressBar
)
from PyQt5 import QtCore


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 700)
        # Central widget
        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")
        MainWindow.setCentralWidget(self.central_widget)

        # Tab widget
        self.tabs = QTabWidget(self.central_widget)
        self.tabs.setObjectName("tabs")
        # Image tab
        self.image_tab = QWidget()
        self.image_tab.setObjectName("image_tab")
        # Video tab
        self.video_tab = QWidget()
        self.video_tab.setObjectName("video_tab")
        self.tabs.addTab(self.image_tab, "Image (Flux)")
        self.tabs.addTab(self.video_tab, "Video (Wan2.2)")

        # Build Image tab UI
        image_layout = QVBoxLayout(self.image_tab)
        # Prompt
        self.prompt_label = QLabel("Prompt:")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter image prompt here...")
        # Negative prompt
        self.neg_label = QLabel("Negative prompt (optional):")
        self.neg_prompt_edit = QTextEdit()
        self.neg_prompt_edit.setPlaceholderText("Enter negative prompt...")
        # Parameters layout
        params_layout = QHBoxLayout()
        self.width_label = QLabel("Width:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 1024)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(512)
        self.height_label = QLabel("Height:")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 1024)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(512)
        self.steps_label = QLabel("Steps:")
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(50)
        self.guidance_label = QLabel("Guidance:")
        self.guidance_spin = QSpinBox()
        self.guidance_spin.setRange(1, 30)
        self.guidance_spin.setValue(7)
        params_layout.addWidget(self.width_label)
        params_layout.addWidget(self.width_spin)
        params_layout.addWidget(self.height_label)
        params_layout.addWidget(self.height_spin)
        params_layout.addWidget(self.steps_label)
        params_layout.addWidget(self.steps_spin)
        params_layout.addWidget(self.guidance_label)
        params_layout.addWidget(self.guidance_spin)
        # Options layout
        options_layout = QHBoxLayout()
        self.quant_checkbox = QCheckBox("Use quantized weights (nf4)")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu"])
        options_layout.addWidget(self.quant_checkbox)
        options_layout.addWidget(QLabel("Device:"))
        options_layout.addWidget(self.device_combo)
        # Generate controls
        self.gen_button = QPushButton("Generate Image")
        self.image_progress = QProgressBar()
        self.image_display = QLabel()
        self.image_display.setAlignment(QtCore.Qt.AlignCenter)
        self.image_display.setMinimumHeight(300)
        # Assemble image tab
        image_layout.addWidget(self.prompt_label)
        image_layout.addWidget(self.prompt_edit)
        image_layout.addWidget(self.neg_label)
        image_layout.addWidget(self.neg_prompt_edit)
        image_layout.addLayout(params_layout)
        image_layout.addLayout(options_layout)
        image_layout.addWidget(self.gen_button)
        image_layout.addWidget(self.image_progress)
        image_layout.addWidget(self.image_display)

        # Build Video tab UI
        video_layout = QVBoxLayout(self.video_tab)
        # Prompt
        self.video_prompt_label = QLabel("Prompt:")
        self.video_prompt_edit = QTextEdit()
        self.video_prompt_edit.setPlaceholderText("Enter video prompt here...")
        # Negative prompt
        self.video_neg_label = QLabel("Negative prompt (optional):")
        self.video_neg_prompt_edit = QTextEdit()
        self.video_neg_prompt_edit.setPlaceholderText("Enter negative prompt...")
        # Video parameters
        video_params_layout = QHBoxLayout()
        self.video_width_label = QLabel("Width:")
        self.video_width_spin = QSpinBox()
        self.video_width_spin.setRange(256, 1024)
        self.video_width_spin.setSingleStep(64)
        self.video_width_spin.setValue(480)
        self.video_height_label = QLabel("Height:")
        self.video_height_spin = QSpinBox()
        self.video_height_spin.setRange(256, 1024)
        self.video_height_spin.setSingleStep(64)
        self.video_height_spin.setValue(480)
        self.frames_label = QLabel("Frames:")
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 300)
        self.frames_spin.setValue(16)
        self.video_steps_label = QLabel("Steps:")
        self.video_steps_spin = QSpinBox()
        self.video_steps_spin.setRange(1, 150)
        self.video_steps_spin.setValue(50)
        video_params_layout.addWidget(self.video_width_label)
        video_params_layout.addWidget(self.video_width_spin)
        video_params_layout.addWidget(self.video_height_label)
        video_params_layout.addWidget(self.video_height_spin)
        video_params_layout.addWidget(self.frames_label)
        video_params_layout.addWidget(self.frames_spin)
        video_params_layout.addWidget(self.video_steps_label)
        video_params_layout.addWidget(self.video_steps_spin)
        # Options layout
        video_options_layout = QHBoxLayout()
        self.offload_checkbox = QCheckBox("Enable offloading")
        self.t5_cpu_checkbox = QCheckBox("T5 on CPU")
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["bfloat16", "fp16"])
        video_options_layout.addWidget(self.offload_checkbox)
        video_options_layout.addWidget(self.t5_cpu_checkbox)
        video_options_layout.addWidget(QLabel("Precision:"))
        video_options_layout.addWidget(self.precision_combo)
        # Generate controls
        self.video_button = QPushButton("Generate Video")
        self.video_progress = QProgressBar()
        # Video display placeholder
        self.video_display = QLabel()
        self.video_display.setAlignment(QtCore.Qt.AlignCenter)
        self.video_display.setMinimumHeight(300)
        # Assemble video tab
        video_layout.addWidget(self.video_prompt_label)
        video_layout.addWidget(self.video_prompt_edit)
        video_layout.addWidget(self.video_neg_label)
        video_layout.addWidget(self.video_neg_prompt_edit)
        video_layout.addLayout(video_params_layout)
        video_layout.addLayout(video_options_layout)
        video_layout.addWidget(self.video_button)
        video_layout.addWidget(self.video_progress)
        video_layout.addWidget(self.video_display)

        # Status bar
        self.status_bar = MainWindow.statusBar()
        self.status_bar.showMessage("Ready")

        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.tabs)
