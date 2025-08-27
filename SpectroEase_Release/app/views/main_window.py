import sys  # Import system module for accessing system features
import os  # Import operating system module for handling files and directories

  

import pandas as pd  # Import Pandas library for data processing
import numpy as np  # Import NumPy library for numerical calculations
import matplotlib.pyplot as plt  # Import Matplotlib library for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Import Qt5 plotting backend
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar  # Import Qt5 navigation toolbar
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QFileDialog, QMessageBox, QVBoxLayout,
    QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget, QAbstractItemView,
    QGroupBox, QDialog, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QProgressBar, QSplitter, QToolBar, QShortcut, QSpinBox, QCheckBox, QDoubleSpinBox,
    QScrollArea, QSizePolicy, QFrame, QTextEdit, QSlider, QDockWidget, QMenu,
    QStatusBar, QLineEdit, QHeaderView, QFormLayout, QDialogButtonBox
)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPalette  # Import icon class, font, color, and palette
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal  # Import Qt core module, size, thread, and signal
import matplotlib
matplotlib.use('Qt5Agg')

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of current file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Get project root directory
if project_root not in sys.path:  # If project root directory is not in Python path
    sys.path.insert(0, project_root)  # Add project root directory to Python path

# Ensure app directory is in Python path
app_dir = os.path.dirname(os.path.dirname(current_dir))  # Get app directory
if app_dir not in sys.path:  # If app directory is not in Python path
    sys.path.insert(0, app_dir)  # Add app directory to Python path

# Import required widgets from PyQt5 library
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QFileDialog, QMessageBox, QVBoxLayout,
    QHBoxLayout, QWidget, QPushButton, QLabel, QListWidget, QAbstractItemView,
    QGroupBox, QDialog, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QProgressBar, QSplitter, QToolBar, QShortcut, QSpinBox, QCheckBox, QDoubleSpinBox,
    QScrollArea, QSizePolicy, QFrame, QTextEdit, QSlider, QDockWidget, QMenu,
    QStatusBar, QLineEdit, QHeaderView, QFormLayout, QDialogButtonBox
)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPalette  # Import icon class, font, color, and palette
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal  # Import Qt core module, size, thread, and signal

from app.controllers.main_controller import MainController  # Import main controller
from app.views.data_view import DataView, DataPartitioningView  # Import data view and data partitioning view
from app.views.preprocessing_view import PreprocessingView  # Import preprocessing view
from app.views.feature_selection_view import FeatureSelectionView  # Import feature selection view
from app.views.modeling_view import ModelingView  # Import modeling view
from app.views.evaluation_view import EvaluationView  # Import evaluation view
from app.views.hyperparameter_optimization_view import HyperparameterOptimizationView  # Import hyperparameter optimization view
from app.views.llm_dialog import LLMDialog  # Import LLM dialog
from config.llm_config import LLMConfig  # Import LLM configuration
from app.services.llm_service import LLMService  # Import LLM service
from app.services.onnx_service import ONNXService  # Import ONNX service
from PyQt5.QtCore import Qt, QSize  # Import Qt core module
from PyQt5.QtGui import QFont, QKeySequence, QPalette, QColor  # Import font, keyboard shortcut, palette, and color
from PyQt5.QtGui import QPixmap  # Import Pixmap class
from PyQt5.QtCore import QThread, pyqtSignal  # Import thread and signal

# Import various plugins
from plugins.preprocessing.spectrum_converter import SpectrumConverter  # Import spectrum converter
from plugins.preprocessing.spectrum_visualizer import SpectrumVisualizer  # Import spectrum visualizer
from plugins.preprocessing.spectrum_preprocessor import SpectrumPreprocessor  # Import spectrum preprocessor
from plugins.feature_selection.spectrum_feature_extractor import SpectrumFeatureExtractor  # Import spectrum feature extractor
from plugins.modeling.quantitative_analyzer import QuantitativeAnalyzer  # Import quantitative analyzer
from plugins.modeling.qualitative_analyzer import QualitativeAnalyzer  # Import qualitative analyzer
from plugins.reporting.spectrum_report_generator import SpectrumReportGenerator  # Import spectrum report generator
from app.views.visualization_widget import VisualizationWidget  # Import visualization widget
from app.views.progress_dialog import ProgressDialog  # Import progress dialog
from app.models.preprocessing_model import PreprocessingModel  # Import preprocessing model

class MainWindow(QMainWindow):  # Define main window class, inheriting from QMainWindow
    def __init__(self, translator, preprocessing_plugins, feature_selection_plugins, 
                 modeling_plugins, data_partitioning_plugins, llm_service):
        super().__init__()  # Call parent class constructor
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logo.png')  # Get path to icon
        self.setWindowIcon(QIcon(icon_path))  # Set window icon
        # Initialize various plugins and services
        self.translator = translator  # Translator
        self.preprocessing_plugins = preprocessing_plugins  # Preprocessing plugins
        self.feature_selection_plugins = feature_selection_plugins  # Feature selection plugins
        self.modeling_plugins = modeling_plugins  # Modeling plugins
        self.data_partitioning_plugins = data_partitioning_plugins  # Data partitioning plugins
        self.llm_service = llm_service  # LLM service
        self.onnx_service = ONNXService(save_directory="models")  # Initialize ONNX service
        self.spectra_data = None  # Initialize spectral data
        self.wavelengths = None  # Initialize wavelengths
        self.current_spectra = None  # Initialize current spectral data
        # Initialize views
        self.preprocessing_view = PreprocessingView(self.preprocessing_plugins)  # Create preprocessing view
        self.feature_selection_view = FeatureSelectionView(self.feature_selection_plugins)  # Create feature selection view
        self.modeling_view = ModelingView(self.modeling_plugins)  # Create modeling view
        self.evaluation_view = EvaluationView()  # Create evaluation view
        self.hyperopt_view = HyperparameterOptimizationView()  # Create hyperparameter optimization view
        self.data_partitioning_view = DataPartitioningView(self.data_partitioning_plugins)  # Create data partitioning view
        # Set application style
        self.set_application_style()  # Call method to set style
        self.init_ui()  # Initialize user interface
        self.controller = MainController(self, self.translator)  # Create main controller
        self.connect_signals()  # Connect signals and slots
          
        self.scatter_method.currentTextChanged.connect(self.on_scatter_method_changed)
          
        self.snv_parameters = {
            'center': True,
            'scale': True,
            'min_std': 1e-6,
            'axis': 1,
            'copy': True
        }

    def set_application_style(self):
        """Set the application style to a modern dark theme"""
          
        modern_stylesheet = """English text"form-label"English text"secondary"] {
            background-color: #718096;
        }
        QPushButton[role="secondary"]:hover {
            background-color: #4a5568;
        }
        QPushButton[role="success"] {
            background-color: #38a169;
        }
        QPushButton[role="success"English text"""
        self.setStyleSheet(modern_stylesheet)
        # Set color scheme for visualization window
        plt.style.use('default')  # Use default plotting style
        # Custom light gray theme
        plt.rcParams['axes.facecolor'] = '#f8f8f8'  # Set background color of axes
        plt.rcParams['figure.facecolor'] = '#ffffff'  # Set background color of figures
        plt.rcParams['grid.color'] = '#e0e0e0'  # Set color of grid lines
        plt.rcParams['text.color'] = '#333333'  # Set text color
        plt.rcParams['axes.labelcolor'] = '#333333'  # Set color of axes labels
        plt.rcParams['xtick.color'] = '#666666'  # Set color of x-axis ticks
        plt.rcParams['ytick.color'] = '#666666'  # Set color of y-axis ticks

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("SpectroEase")  # Set window title to "SpectroEase"
        # Set window size, top-left corner at (50, 50), width 2000, height 1200
        self.setGeometry(50, 50, 2000, 1200)  
        # Set font for application, Arial, size 9
        app_font = QFont("Arial", 9)  # Change to Arial to support English and set appropriate font size
        self.setFont(app_font)  # Apply this font to entire window
        # Create a horizontal splitter to separate left and right content
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)  # Set splitter as central widget of main window
        # Create container for left panel with improved layout
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(6)    
        left_layout.setContentsMargins(10, 10, 10, 10)    
        # Create a scroll area to wrap left content, ensuring all content is visible
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to adjust size based on content
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar based on need
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show vertical scroll bar based on need
        scroll_area.setWidget(left_container)  # Set left container as content of scroll area
        scroll_area.setMinimumWidth(550)    
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # Remove border of scroll area
        # Adjust size of left label and control
        min_label_width = 140    
        min_combobox_width = 200    
        # Data management group - Create a group box to manage data-related widgets
        self.data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()  # Create a vertical layout manager
        data_layout.setSpacing(0)  # Reduce spacing between widgets to 0 pixels
        data_layout.setContentsMargins(2, 2, 2, 2)  # Reduce margins
        # Add import data button
        self.import_btn = QPushButton("Import Data")  # Create import data button
        self.import_btn.setFixedHeight(25)  # Set button height to 25px
        data_layout.addWidget(self.import_btn)  # Add import button to data layout
        self.data_group.setFixedHeight(55)  # Increase height to accommodate added button
        data_layout.addWidget(self.data_group)  # Add data management group to left layout
        # Data partitioning group - Create a group box to manage data partitioning-related widgets
        self.data_partitioning_group = QGroupBox("Data Partitioning")
        self.data_partitioning_group.setFixedHeight(150)  # Reduce height to 130px (was 160px)
        data_partitioning_layout = QVBoxLayout()  # Create a vertical layout manager
        data_partitioning_layout.setSpacing(3)  # Reduce spacing between components to 3px (was 5px)
        data_partitioning_layout.setContentsMargins(5, 5, 5, 5)  # Leave 5px space on all sides
        # Create a container to wrap data partitioning view, limiting its height
        data_partition_container = QWidget()
        data_partition_container_layout = QVBoxLayout(data_partition_container)
        data_partition_container_layout.setSpacing(2)  # Reduce internal spacing
        data_partition_container_layout.setContentsMargins(2, 2, 2, 2)  # Reduce internal margins
        data_partition_container_layout.addWidget(self.data_partitioning_view)
        data_partition_container.setFixedHeight(80)  # Reduce height of view container (was 120px)
        # Add data partitioning view container
        data_partitioning_layout.addWidget(data_partition_container)
        # Apply button
        self.data_partition_btn = QPushButton("Partition Data")
        self.data_partition_btn.setFixedHeight(22)  # Set button height to 22px
        self.data_partition_btn.clicked.connect(self.partition_data_with_params)
        data_partitioning_layout.addWidget(self.data_partition_btn)
        self.data_partitioning_group.setLayout(data_partitioning_layout)  # Set data partitioning layout to data partitioning group box
        left_layout.addWidget(self.data_partitioning_group)  # Add data partitioning group box to left layout
        # Preprocessing group - Create a group box to manage preprocessing-related widgets
        self.preprocessing_group = QGroupBox("Preprocessing")
        self.preprocessing_layout = QVBoxLayout()  # Create a vertical layout manager and save as instance variable
        self.preprocessing_layout.setSpacing(0)  # Completely remove spacing
        self.preprocessing_layout.setContentsMargins(1, 1, 1, 1)  # Try to reduce margins
        # Create preprocessing tab
        self.preprocessing_tabs = QTabWidget()
        self.preprocessing_tabs.setMinimumHeight(160)  # Increase minimum height to ensure full content is displayed
        self.preprocessing_tabs.setContentsMargins(2, 2, 2, 2)  # Leave 2px space on all sides
        # Compress spacing between Preprocessing and tabs
        self.preprocessing_layout.setSpacing(0)  # Completely remove spacing
        self.preprocessing_tabs.setTabText(0, "Basic")  # Set text of first tab to "Basic"
        self.preprocessing_tabs.setTabText(1, "Advanced")  # Set text of second tab to "Advanced"
        self.preprocessing_tabs.setTabText(2, "Special")  # Set text of third tab to "Special"
        # Basic preprocessing tab
        basic_tab = QWidget()  # Create a new tab
        basic_layout = QHBoxLayout()  # Create horizontal layout for two columns
        basic_layout.setSpacing(10)  # Add spacing between columns
        basic_layout.setContentsMargins(5, 5, 5, 5)  # Add margins
        
        # Left column
        left_column = QVBoxLayout()
        left_column.setSpacing(8)
        
        # Right column  
        right_column = QVBoxLayout()
        right_column.setSpacing(8)
        
        # All components use uniform height setting
        row_height = 25 # Set row height to 20 pixels
        
        # --- Spike Removal Group (Left Column) ---
        despiking_group = QGroupBox("Spike Removal")
        despiking_outer_layout = QVBoxLayout()
        despiking_outer_layout.setSpacing(3)
        despiking_outer_layout.setContentsMargins(5, 5, 5, 5)

        self.despiking_check = QCheckBox("Enable Spike Removal")
        self.despiking_check.setFont(QFont("Arial", 9))
        self.despiking_check.setChecked(False)  # Default off
        despiking_outer_layout.addWidget(self.despiking_check)

        self.despiking_params_container = QWidget()
        despiking_form_layout = QFormLayout(self.despiking_params_container)
        despiking_form_layout.setSpacing(3)
        despiking_form_layout.setContentsMargins(0, 3, 0, 0)
        despiking_form_layout.setLabelAlignment(Qt.AlignLeft)

        # Method selection
        self.despiking_method = QComboBox()
        self.despiking_method.addItems(["MAD", "Local Z-score"])
        self.despiking_method.setFixedHeight(22)
        self.despiking_method.setFixedWidth(120)
        despiking_form_layout.addRow("Method:", self.despiking_method)

        # Window size
        self.despiking_window = QSpinBox()
        self.despiking_window.setRange(7, 15)
        self.despiking_window.setValue(7)
        self.despiking_window.setSingleStep(2)
        self.despiking_window.setFixedHeight(22)
        self.despiking_window.setFixedWidth(120)
        despiking_form_layout.addRow("Window:", self.despiking_window)

        # Threshold
        self.despiking_threshold = QDoubleSpinBox()
        self.despiking_threshold.setRange(3.0, 6.0)
        self.despiking_threshold.setValue(5.0)
        self.despiking_threshold.setSingleStep(0.5)
        self.despiking_threshold.setFixedHeight(22)
        self.despiking_threshold.setFixedWidth(120)
        despiking_form_layout.addRow("Threshold:", self.despiking_threshold)

        despiking_outer_layout.addWidget(self.despiking_params_container)
        despiking_group.setLayout(despiking_outer_layout)
        
        # Enable/disable parameters based on checkbox
        self.despiking_params_container.setEnabled(False)
        self.despiking_check.toggled.connect(self.despiking_params_container.setEnabled)
        
        left_column.addWidget(despiking_group)
        
        # --- Baseline Correction Group (Rebuilt with QFormLayout) ---
        baseline_group = QGroupBox("Baseline Correction")
        baseline_outer_layout = QVBoxLayout()
        baseline_outer_layout.setSpacing(3)
        baseline_outer_layout.setContentsMargins(5, 5, 5, 5)

        self.baseline_check = QCheckBox("Enable Baseline Correction")
        self.baseline_check.setFont(QFont("Arial", 9))
        baseline_outer_layout.addWidget(self.baseline_check)

        self.baseline_params_container = QWidget()
        baseline_form_layout = QFormLayout(self.baseline_params_container)
        baseline_form_layout.setSpacing(3)
        baseline_form_layout.setContentsMargins(0, 3, 0, 0)
        baseline_form_layout.setLabelAlignment(Qt.AlignLeft)

        # Method selection for baseline correction
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["Polynomial", "ALS", "airPLS"])
        self.baseline_method.setFixedHeight(22)
        self.baseline_method.setFixedWidth(120)
        baseline_form_layout.addRow("Method:", self.baseline_method)

        self.poly_order = QSpinBox()
        self.poly_order.setRange(1, 10)
        self.poly_order.setFixedHeight(22)
        self.poly_order.setFixedWidth(120)
        baseline_form_layout.addRow("Polynomial Order:", self.poly_order)
        baseline_outer_layout.addWidget(self.baseline_params_container)
        baseline_group.setLayout(baseline_outer_layout)

        self.baseline_check.toggled.connect(self.baseline_params_container.setEnabled)
        self.baseline_params_container.setEnabled(False)

        left_column.addWidget(baseline_group)
        # --- Scatter Correction Group (Rebuilt with QFormLayout) ---
        scatter_group = QGroupBox("Scatter Correction")
        scatter_outer_layout = QVBoxLayout()
        scatter_outer_layout.setSpacing(3)
        scatter_outer_layout.setContentsMargins(5, 5, 5, 5)

        self.scatter_check = QCheckBox("Enable Scatter Correction")
        self.scatter_check.setFont(QFont("Arial", 9))
        scatter_outer_layout.addWidget(self.scatter_check)

        self.scatter_params_container = QWidget()
        scatter_form_layout = QFormLayout(self.scatter_params_container)
        scatter_form_layout.setSpacing(8)
        scatter_form_layout.setContentsMargins(0, 5, 0, 0)
        scatter_form_layout.setLabelAlignment(Qt.AlignLeft)

        self.scatter_method = QComboBox()
        self.scatter_method.addItems(["MSC", "SNV", "EMSC", "RNV", "OSC"])
        self.scatter_method.setFixedHeight(22)
        self.scatter_method.setFixedWidth(120)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFixedHeight(22)
        self.settings_btn.setFixedWidth(70)
        self.settings_btn.clicked.connect(self.open_scatter_parameters)   
        scatter_method_layout = QHBoxLayout()
        scatter_method_layout.setContentsMargins(0,0,0,0)
        scatter_method_layout.addWidget(self.scatter_method)
        scatter_method_layout.addWidget(self.settings_btn)
        scatter_method_layout.addStretch()
        scatter_form_layout.addRow("Method:", scatter_method_layout)
        scatter_outer_layout.addWidget(self.scatter_params_container)
        scatter_group.setLayout(scatter_outer_layout)
        self.scatter_check.toggled.connect(self.scatter_params_container.setEnabled)
        self.scatter_params_container.setEnabled(False)
        right_column.addWidget(scatter_group)
          
        smooth_group = QGroupBox("Smoothing")
        smooth_outer_layout = QVBoxLayout()   
        smooth_outer_layout.setSpacing(3)
        smooth_outer_layout.setContentsMargins(5, 5, 5, 5)

          
        self.smooth_check = QCheckBox("Enable Smoothing")
        self.smooth_check.setFont(QFont("Arial", 9))
        self.smooth_check.setFixedHeight(22)
        smooth_outer_layout.addWidget(self.smooth_check)

          
        self.smoothing_params_container = QWidget()   
        smooth_form_layout = QFormLayout(self.smoothing_params_container)
        smooth_form_layout.setSpacing(8)
        smooth_form_layout.setContentsMargins(0, 5, 0, 0)   
        smooth_form_layout.setLabelAlignment(Qt.AlignLeft)   

          
        self.smooth_method = QComboBox()
        self.smooth_method.addItems(["S-Golay", "Moving Avg", "Median", "Wavelet"])
        self.smooth_method.setFixedHeight(22)
        self.smooth_method.setFixedWidth(120)
        smooth_form_layout.addRow("Method:", self.smooth_method)
          
        self.window_size = QSpinBox()
        self.window_size.setRange(3, 51)
        self.window_size.setValue(11)
        self.window_size.setSingleStep(2)
        self.window_size.setFixedHeight(22)
        self.window_size.setFixedWidth(120)
        smooth_form_layout.addRow("Window:", self.window_size)
          
        self.poly_order_label = QLabel("Poly:")
        self.smooth_poly_order = QSpinBox()
        self.smooth_poly_order.setRange(2, 5)
        self.smooth_poly_order.setValue(2)
        self.smooth_poly_order.setFixedHeight(22)
        self.smooth_poly_order.setFixedWidth(120)
        self.poly_order_row = smooth_form_layout.addRow(self.poly_order_label, self.smooth_poly_order)

        smooth_outer_layout.addWidget(self.smoothing_params_container)
        smooth_group.setLayout(smooth_outer_layout)
          
        self.smooth_check.toggled.connect(self.smoothing_params_container.setEnabled)
        self.smooth_method.currentTextChanged.connect(self._toggle_poly_order_visibility)

          
        self.smoothing_params_container.setEnabled(False)
        self._toggle_poly_order_visibility(self.smooth_method.currentText())

        right_column.addWidget(smooth_group)
        
        # Add stretch to balance columns
        left_column.addStretch()
        right_column.addStretch()
        
        # Add columns to basic layout
        basic_layout.addLayout(left_column)
        basic_layout.addLayout(right_column)
        basic_tab.setLayout(basic_layout)    
        self.preprocessing_tabs.addTab(basic_tab, "Basic")    
          
        advanced_tab = QWidget()    
        advanced_layout = QVBoxLayout()    
        advanced_layout.setSpacing(1)    
        advanced_layout.setContentsMargins(3, 3, 3, 3)    
          
        standardize_group = QGroupBox("Standardization")
        standardize_layout = QVBoxLayout()    
        standardize_layout.setSpacing(1)    
        standardize_layout.setContentsMargins(2, 2, 2, 2)    
        self.standardize_check = QCheckBox("Enable Standardization")    
        self.standardize_check.setFont(QFont("Arial", 9))    
        standardize_params = QHBoxLayout()    
        standardize_params.setSpacing(1)    
        std_method_label = QLabel("Method:")    
        std_method_label.setMinimumWidth(min_label_width)    
        std_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        standardize_params.addWidget(std_method_label)    
        self.standardize_method = QComboBox()    
        self.standardize_method.addItems(["Z-Score", "Robust", "Min-Max"])    
        self.standardize_method.setMinimumWidth(min_combobox_width)    
        standardize_params.addWidget(self.standardize_method)    
        standardize_layout.addWidget(self.standardize_check)    
        standardize_layout.addLayout(standardize_params)    
        standardize_group.setLayout(standardize_layout)    
        standardize_group.setMaximumHeight(70)    
        advanced_layout.addWidget(standardize_group)    
          
        norm_group = QGroupBox("Normalization")
        norm_layout = QVBoxLayout()    
        norm_layout.setSpacing(1)    
        norm_layout.setContentsMargins(2, 2, 2, 2)    
        self.norm_check = QCheckBox("Enable Normalization")    
        self.norm_check.setFont(QFont("Arial", 9))    
        norm_params = QHBoxLayout()    
        norm_params.setSpacing(2)    
        norm_method_label = QLabel("Method:")    
        norm_method_label.setMinimumWidth(min_label_width)    
        norm_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        norm_params.addWidget(norm_method_label)    
        self.norm_method = QComboBox()    
        self.norm_method.addItems(["Min-Max", "Vector", "Area", "Maximum"])    
        self.norm_method.setMinimumWidth(min_combobox_width)    
        norm_params.addWidget(self.norm_method)    
        norm_layout.addWidget(self.norm_check)    
        norm_layout.addLayout(norm_params)    
        norm_group.setLayout(norm_layout)    
        norm_group.setMaximumHeight(70)    
        advanced_layout.addWidget(norm_group)    
          
        derivative_group = QGroupBox("Derivative")
        derivative_layout = QVBoxLayout()    
        derivative_layout.setSpacing(1)    
        derivative_layout.setContentsMargins(2, 2, 2, 2)    
        self.derivative_check = QCheckBox("Enable Derivative")    
        self.derivative_check.setFont(QFont("Arial", 9))    
        derivative_params = QHBoxLayout()    
        derivative_params.setSpacing(2)    
        derivative_order_label = QLabel("Order:")    
        derivative_order_label.setMinimumWidth(min_label_width)    
        derivative_order_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        derivative_params.addWidget(derivative_order_label)    
        self.derivative_order = QSpinBox()    
        self.derivative_order.setRange(1, 3)    
        self.derivative_order.setMinimumWidth(60)    
        derivative_params.addWidget(self.derivative_order)    
        derivative_layout.addWidget(self.derivative_check)    
        derivative_layout.addLayout(derivative_params)    
        derivative_group.setLayout(derivative_layout)    
        derivative_group.setMaximumHeight(70)    
        advanced_layout.addWidget(derivative_group)    
        advanced_tab.setLayout(advanced_layout)    
        self.preprocessing_tabs.addTab(advanced_tab, "Advanced")    
          
        special_tab = QWidget()    
        special_layout = QVBoxLayout()    
        special_layout.setSpacing(4)    
        special_layout.setContentsMargins(3, 5, 3, 3)    
          
        special_form_layout = QHBoxLayout()    
        special_form_layout.setSpacing(8)    
          
        left_column = QVBoxLayout()    
        left_column.setSpacing(6)    
          
        outlier_group = QGroupBox("Outlier Detection")
        outlier_layout = QVBoxLayout()    
        outlier_layout.setSpacing(2)    
        outlier_layout.setContentsMargins(4, 4, 4, 4)    
        self.outlier_check = QCheckBox("Enable Outlier Detection")    
        self.outlier_check.setFont(QFont("Arial", 9))    
        outlier_params = QHBoxLayout()    
        outlier_params.setSpacing(4)    
        outlier_threshold_label = QLabel("Threshold:")    
        outlier_threshold_label.setMinimumWidth(min_label_width)    
        outlier_threshold_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        outlier_threshold_label.setFont(QFont("Arial", 9))    
        outlier_params.addWidget(outlier_threshold_label)    
        self.outlier_threshold = QDoubleSpinBox()    
        self.outlier_threshold.setRange(1.0, 5.0)    
        self.outlier_threshold.setValue(3.0)    
        self.outlier_threshold.setSingleStep(0.1)    
        self.outlier_threshold.setMinimumWidth(60)    
        outlier_params.addWidget(self.outlier_threshold)    
        outlier_layout.addWidget(self.outlier_check)    
        outlier_layout.addLayout(outlier_params)    
        outlier_group.setLayout(outlier_layout)    
        outlier_group.setMaximumHeight(80)    
        left_column.addWidget(outlier_group)    
          
        denoise_group = QGroupBox("Denoising")
        denoise_layout = QVBoxLayout()    
        denoise_layout.setSpacing(2)    
        denoise_layout.setContentsMargins(4, 4, 4, 4)    
        self.denoise_check = QCheckBox("Enable Denoising")    
        self.denoise_check.setFont(QFont("Arial", 9))    
        denoise_params = QHBoxLayout()    
        denoise_params.setSpacing(4)    
        denoise_strength_label = QLabel("Strength:")    
        denoise_strength_label.setMinimumWidth(min_label_width)    
        denoise_strength_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        denoise_strength_label.setFont(QFont("Arial", 9))    
        denoise_params.addWidget(denoise_strength_label)    
        self.denoise_strength = QDoubleSpinBox()    
        self.denoise_strength.setRange(0.1, 2.0)    
        self.denoise_strength.setValue(0.5)    
        self.denoise_strength.setSingleStep(0.1)    
        self.denoise_strength.setMinimumWidth(60)    
        denoise_params.addWidget(self.denoise_strength)    
        denoise_layout.addWidget(self.denoise_check)    
        denoise_layout.addLayout(denoise_params)    
        denoise_group.setLayout(denoise_layout)    
        denoise_group.setMaximumHeight(80)    
        left_column.addWidget(denoise_group)    
        special_form_layout.addLayout(left_column)    
          
        right_column = QVBoxLayout()    
        right_column.setSpacing(6)    
          
        alignment_group = QGroupBox("Peak Alignment")
        alignment_layout = QVBoxLayout()    
        alignment_layout.setSpacing(2)    
        alignment_layout.setContentsMargins(4, 4, 4, 4)    
        self.alignment_check = QCheckBox("Enable Peak Alignment")    
        self.alignment_check.setFont(QFont("Arial", 9))    
        alignment_params = QHBoxLayout()    
        alignment_params.setSpacing(4)    
        alignment_method_label = QLabel("Method:")    
        alignment_method_label.setMinimumWidth(min_label_width)    
        alignment_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        alignment_method_label.setFont(QFont("Arial", 9))    
        alignment_params.addWidget(alignment_method_label)    
        self.alignment_method = QComboBox()    
        self.alignment_method.addItems(["DTW", "COW", "ICS", "PAFFT"])    
        self.alignment_method.setMinimumWidth(min_combobox_width)    
        alignment_params.addWidget(self.alignment_method)    
        alignment_layout.addWidget(self.alignment_check)    
        alignment_layout.addLayout(alignment_params)    
        alignment_group.setLayout(alignment_layout)    
        alignment_group.setMaximumHeight(80)    
        right_column.addWidget(alignment_group)    
          
        reference_group = QGroupBox("Reference Selection")
        reference_layout = QVBoxLayout()    
        reference_layout.setSpacing(2)    
        reference_layout.setContentsMargins(4, 4, 4, 4)    
        reference_method_layout = QHBoxLayout()    
        reference_method_layout.setSpacing(4)    
        reference_method_label = QLabel("Method:")    
        reference_method_label.setMinimumWidth(min_label_width)    
        reference_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        reference_method_label.setFont(QFont("Arial", 9))    
        reference_method_layout.addWidget(reference_method_label)    
        self.reference_method = QComboBox()    
        self.reference_method.addItems(["Mean", "Med", "Max", "First"])    
        self.reference_method.setMinimumWidth(min_combobox_width)    
        reference_method_layout.addWidget(self.reference_method)    
        reference_layout.addLayout(reference_method_layout)    
        reference_group.setLayout(reference_layout)    
        reference_group.setMaximumHeight(70)    
        right_column.addWidget(reference_group)    
        special_form_layout.addLayout(right_column)    
        special_layout.addLayout(special_form_layout)
        
        # Raman Specific Preprocessing Placeholders
        raman_group = QGroupBox("Raman Specific Preprocessing")
        raman_layout = QVBoxLayout()
        raman_layout.setSpacing(3)
        raman_layout.setContentsMargins(5, 5, 5, 5)
        
        # Wavelength calibration placeholder
        wavelength_layout = QHBoxLayout()
        self.wavelength_calib_check = QCheckBox("Wavelength Calibration")
        self.wavelength_calib_check.setChecked(False)
        wavelength_layout.addWidget(self.wavelength_calib_check)
        
        self.wavelength_calib_file_btn = QPushButton("Select CSV File")
        self.wavelength_calib_file_btn.setEnabled(False)
        self.wavelength_calib_file_btn.setMaximumWidth(120)
        wavelength_layout.addWidget(self.wavelength_calib_file_btn)
        wavelength_layout.addStretch()
        raman_layout.addLayout(wavelength_layout)
        
        # Intensity calibration placeholder  
        intensity_layout = QHBoxLayout()
        self.intensity_calib_check = QCheckBox("Intensity Calibration")
        self.intensity_calib_check.setChecked(False)
        intensity_layout.addWidget(self.intensity_calib_check)
        
        self.intensity_calib_file_btn = QPushButton("Select CSV File")
        self.intensity_calib_file_btn.setEnabled(False)
        self.intensity_calib_file_btn.setMaximumWidth(120)
        intensity_layout.addWidget(self.intensity_calib_file_btn)
        intensity_layout.addStretch()
        raman_layout.addLayout(intensity_layout)
        
        # Info label
        raman_info_label = QLabel("Note: External preprocessing can also meet these requirements")
        raman_info_label.setStyleSheet("color: #666; font-style: italic;")
        raman_layout.addWidget(raman_info_label)
        
        # Connect checkboxes to enable/disable buttons
        self.wavelength_calib_check.toggled.connect(self.wavelength_calib_file_btn.setEnabled)
        self.intensity_calib_check.toggled.connect(self.intensity_calib_file_btn.setEnabled)
        
        raman_group.setLayout(raman_layout)
        special_layout.addWidget(raman_group)
        
        special_tab.setLayout(special_layout)    
        self.preprocessing_tabs.addTab(special_tab, "Special")    
        self.preprocessing_layout.addWidget(self.preprocessing_tabs)    
          
        checkboxes = [
            self.despiking_check, self.baseline_check, self.scatter_check, self.smooth_check, 
            self.standardize_check, self.norm_check, self.derivative_check,
            self.outlier_check, self.denoise_check, self.alignment_check,
            self.wavelength_calib_check, self.intensity_calib_check
        ]
        for checkbox in checkboxes:    
            checkbox.setFont(QFont("Arial", 9))    
              
            checkbox.setStyleSheet("QCheckBox { padding: 2px; margin: 1px; }")    
          
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout()    
        preview_layout.setSpacing(5)    
        preview_layout.setContentsMargins(5, 5, 5, 5)    
          
        display_label = QLabel("Display:")    
        display_label.setFixedHeight(20)    
        preview_layout.addWidget(display_label)    
        self.preview_mode = QComboBox()    
        self.preview_mode.addItems(["Original", "Processed", "Comparison"])    
        self.preview_mode.setFixedHeight(20)    
        self.preview_mode.setFixedWidth(120)    
        preview_layout.addWidget(self.preview_mode)    
          
        preview_btn = QPushButton("Preview")    
        preview_btn.setFixedHeight(20)    
        preview_btn.clicked.connect(self.preview_preprocessing)    
        preview_layout.addWidget(preview_btn)    
        preview_group.setLayout(preview_layout)    
        preview_group.setFixedHeight(80)    
        self.preprocessing_layout.addWidget(preview_group)    
          
        button_group = QHBoxLayout()    
        button_group.setSpacing(5)    
        button_group.setContentsMargins(5, 5, 5, 5)    
        self.preprocess_btn = QPushButton("Apply")    
        self.preprocess_btn.setMinimumHeight(22)    
        self.preprocess_btn.setMaximumHeight(22)    
        self.preprocess_btn.clicked.connect(self.apply_preprocessing)    
        button_group.addWidget(self.preprocess_btn)    
        reset_btn = QPushButton("Reset")    
        reset_btn.setMinimumHeight(22)    
        reset_btn.setMaximumHeight(22)    
        reset_btn.clicked.connect(self.reset_preprocessing_params)    
        button_group.addWidget(reset_btn)
        help_btn = QPushButton("Help")    
        help_btn.setMinimumHeight(22)    
        help_btn.setMaximumHeight(22)    
        help_btn.clicked.connect(self.show_preprocessing_help)    
        button_group.addWidget(help_btn)
        self.preprocessing_layout.addLayout(button_group)    
        self.preprocessing_group.setLayout(self.preprocessing_layout)    
        left_layout.addWidget(self.preprocessing_group)    
        left_layout.setSpacing(0)    
          
        button_group.setContentsMargins(5, 5, 5, 5)    
        self.preprocessing_layout.setSpacing(0)    
          
        self.preprocess_btn.setMinimumHeight(25)    
        self.preprocess_btn.setMaximumHeight(25)    
        reset_btn.setMinimumHeight(25)    
        reset_btn.setMaximumHeight(25)    
          
        self.feature_group = QGroupBox("Feature")
        feature_layout = QVBoxLayout()    
        feature_layout.setSpacing(3)    
        feature_layout.setContentsMargins(5, 5, 5, 5)    
          
        feature_controls = QHBoxLayout()    
        feature_controls.setSpacing(5)    
        method_label = QLabel("Method:")    
        method_label.setFixedHeight(20)    
        feature_controls.addWidget(method_label)    
        self.feature_method = QComboBox()    
        self.feature_method.addItems(["PCA", "PLSR", "Peak Detection", "Wavelet"])    
        self.feature_method.setFixedHeight(20)    
        feature_controls.addWidget(self.feature_method)    
        param_label = QLabel("Param:")    
        param_label.setFixedHeight(20)    
        feature_controls.addWidget(param_label)    
        self.feature_param_spin = QSpinBox()    
        self.feature_param_spin.setRange(1, 100)    
        self.feature_param_spin.setValue(10)    
        self.feature_param_spin.setFixedHeight(20)    
        feature_controls.addWidget(self.feature_param_spin)    
        feature_layout.addLayout(feature_controls)    
          
        self.feature_btn = QPushButton("Extract")    
        self.feature_btn.setFixedHeight(22)    
        # self.feature_btn.clicked.connect(self.extract_features)    
        feature_layout.addWidget(self.feature_btn)    
        self.feature_group.setLayout(feature_layout)    
        self.feature_group.setFixedHeight(120)    
        left_layout.addWidget(self.feature_group)    
          
        self.analysis_group = QGroupBox("Analysis")
        self.analysis_group.setMinimumHeight(280)    
        self.analysis_group.setMaximumHeight(320)    
        analysis_layout = QVBoxLayout()    
        analysis_layout.setSpacing(8)    
        analysis_layout.setContentsMargins(5, 8, 5, 8)    
          
        analysis_controls = QHBoxLayout()    
        analysis_controls.setSpacing(2)    
        type_label = QLabel("Type:")    
        type_label.setMinimumWidth(min_label_width)    
        type_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        type_label.setFixedHeight(row_height)    
        analysis_controls.addWidget(type_label)    
        self.analysis_type = QComboBox()    
        self.analysis_type.addItems(["Quantitative", "Qualitative"])    
        self.analysis_type.setMinimumWidth(min_combobox_width)    
        self.analysis_type.setFixedHeight(row_height)    
        analysis_controls.addWidget(self.analysis_type)    
        method_label = QLabel("Method:")    
        method_label.setMinimumWidth(min_label_width)    
        method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        method_label.setFixedHeight(row_height)    
        analysis_controls.addWidget(method_label)    
        self.analysis_method = QComboBox()    
        self.analysis_method.setMinimumWidth(min_combobox_width)    
        self.analysis_method.setFixedHeight(row_height)    
        analysis_controls.addWidget(self.analysis_method)    
        analysis_layout.addLayout(analysis_controls)    
          
        hyperparameter_group = QGroupBox("Hyperparameter Optimization")
        hyperparameter_group.setFixedHeight(90)    
        hyperparameter_layout = QVBoxLayout()    
        hyperparameter_layout.setContentsMargins(5, 5, 5, 5)    
        hyperparameter_layout.setSpacing(5)    
          
        opt_method_row = QHBoxLayout()    
        opt_method_label = QLabel("Method:")    
        opt_method_label.setMinimumWidth(min_label_width)    
        opt_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        opt_method_label.setFixedHeight(row_height)    
        opt_method_row.addWidget(opt_method_label)    
        self.optimization_method = QComboBox()    
        self.optimization_method.addItems(["Grid Search", "Random Search", "Bayesian"])    
        self.optimization_method.setMinimumWidth(min_combobox_width)    
        self.optimization_method.setFixedHeight(row_height)    
        opt_method_row.addWidget(self.optimization_method)    
        hyperparameter_layout.addLayout(opt_method_row)    
          
        metric_cv_row = QHBoxLayout()    
        metric_label = QLabel("Metric:")    
        metric_label.setMinimumWidth(min_label_width)    
        metric_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        metric_label.setFixedHeight(row_height)    
        metric_cv_row.addWidget(metric_label)    
        self.evaluation_metric = QComboBox()    
        self.evaluation_metric.addItems(["RMSE", "R²", "Accuracy", "F1"])    
        self.evaluation_metric.setMinimumWidth(min_combobox_width)    
        self.evaluation_metric.setFixedHeight(row_height)    
        metric_cv_row.addWidget(self.evaluation_metric)    
        hyperparameter_layout.addLayout(metric_cv_row)    
          
        cv_iter_row = QHBoxLayout()    
        cv_label = QLabel("CV Folds:")    
        cv_label.setMinimumWidth(min_label_width)    
        cv_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        cv_label.setFixedHeight(row_height)    
        cv_iter_row.addWidget(cv_label)    
        self.cv_folds = QSpinBox()    
        self.cv_folds.setRange(2, 10)    
        self.cv_folds.setValue(5)    
        self.cv_folds.setMinimumWidth(50)    
        self.cv_folds.setFixedHeight(row_height)    
        cv_iter_row.addWidget(self.cv_folds)    
        iter_label = QLabel("Max Iter:")    
        iter_label.setMinimumWidth(40)    
        iter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        iter_label.setFixedHeight(row_height)    
        cv_iter_row.addWidget(iter_label)    
        self.max_iterations = QSpinBox()    
        self.max_iterations.setRange(10, 1000)    
        self.max_iterations.setValue(100)    
        self.max_iterations.setSingleStep(10)    
        self.max_iterations.setMinimumWidth(50)    
        self.max_iterations.setFixedHeight(row_height)    
        cv_iter_row.addWidget(self.max_iterations)    
        hyperparameter_layout.addLayout(cv_iter_row)    
        hyperparameter_group.setLayout(hyperparameter_layout)    
        hyperparameter_group.setFixedHeight(95)    
        analysis_layout.addWidget(hyperparameter_group)    
          
        self.analyze_btn = QPushButton("Start Analysis")    
        self.analyze_btn.setFixedHeight(22)    
        self.analyze_btn.clicked.connect(self.start_analysis)    
        analysis_layout.addWidget(self.analyze_btn)    
        self.analysis_group.setLayout(analysis_layout)    
        left_layout.addWidget(self.analysis_group)    
          
        self.data_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.preprocessing_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)    
        self.data_partitioning_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.feature_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.analysis_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)    
          
        report_btn = QPushButton("Generate Report")    
        report_btn.setMinimumHeight(22)    
        report_btn.setMaximumHeight(22)    
        report_btn.clicked.connect(self.generate_report)    
        report_btn.setContentsMargins(3, 3, 3, 3)    
        left_layout.addWidget(report_btn)    
          
        right_panel = QWidget()    
        right_layout = QVBoxLayout(right_panel)    
        right_layout.setSpacing(2)    
        right_layout.setContentsMargins(8, 8, 8, 8)    
          
        self.visualization_widget = VisualizationWidget()    
        right_layout.addWidget(self.visualization_widget)    
          
        self.result_table = QTableWidget()
        self.result_table.setMinimumHeight(200)
        self.result_table.setMaximumHeight(200)
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Algorithm", "R² Score", "RMSE"])
          
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)    
        header.setSectionResizeMode(1, QHeaderView.Stretch)             
        header.setSectionResizeMode(2, QHeaderView.Stretch)             
          
        self.result_table.setAlternatingRowColors(True)    
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)    
        self.result_table.verticalHeader().setVisible(False)    
        right_layout.addWidget(self.result_table)
          
        splitter.addWidget(scroll_area)
        splitter.addWidget(right_panel)
          
        splitter.setSizes([600, 1400])    
        splitter.setChildrenCollapsible(False)    
        splitter.setHandleWidth(4)    
          
        splitter.setStretchFactor(0, 1)    
        splitter.setStretchFactor(1, 3)    
          
        self.setup_toolbar()    
          
        self.setup_status_bar()    
          
        self.setup_shortcuts()    
          
        self.setup_operation_history()    
          
        self.update_analysis_methods(self.analysis_type.currentText())    

    def setup_toolbar(self):
        """Setup the main toolbar with action buttons"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))    
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)    
        self.addToolBar(toolbar)
          
        llm_action = QAction("AI Assistant", self)
        llm_action.setStatusTip("English text")
        llm_action.setToolTip("English text")
        llm_action.triggered.connect(self.show_llm_dialog)
        toolbar.addAction(llm_action)
          
        llm_config_action = QAction("AI Config", self)
        llm_config_action.setStatusTip("English text")
        llm_config_action.setToolTip("English text")
        llm_config_action.triggered.connect(self.show_llm_config)
        toolbar.addAction(llm_config_action)
        toolbar.addSeparator()
          
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setStatusTip("English text")
        zoom_in_action.setToolTip("English text")
        zoom_in_action.setShortcut(QKeySequence("Ctrl++"))
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setStatusTip("English text")
        zoom_out_action.setToolTip("English text")
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        toolbar.addSeparator()
          
        save_image_action = QAction("Save Plot", self)
        save_image_action.setStatusTip("English text")
        save_image_action.setToolTip("English text")
        save_image_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_image_action.triggered.connect(self.save_plot)
        toolbar.addAction(save_image_action)
        save_data_action = QAction("Save Data", self)
        save_data_action.setStatusTip("English text")
        save_data_action.setToolTip("English text")
        save_data_action.setShortcut(QKeySequence("Ctrl+S"))
        save_data_action.triggered.connect(self.save_data)
        toolbar.addAction(save_data_action)
        toolbar.addSeparator()
          
        history_action = QAction("History", self)
        history_action.setStatusTip("English text")
        history_action.setToolTip("English text")
        history_action.setShortcut(QKeySequence("Ctrl+H"))
        history_action.triggered.connect(self.show_history)
        toolbar.addAction(history_action)
          
        back_action = QAction("Back", self)
        back_action.setStatusTip("Go back to previous data state")
        back_action.setToolTip("Go back to previous data state")
        back_action.setShortcut(QKeySequence("Ctrl+Z"))
        back_action.triggered.connect(self.undo_last_operation)
        toolbar.addAction(back_action)
    def setup_status_bar(self):
        """Setup the status bar to show application status"""
        status_bar = self.statusBar()    
          
        self.data_status = QLabel("No data loaded")    
        status_bar.addPermanentWidget(self.data_status)    
          
        self.progress_bar = QProgressBar()    
        self.progress_bar.setMaximumWidth(200)    
        status_bar.addPermanentWidget(self.progress_bar)    
          
        self.tip_label = QLabel("")    
        status_bar.addWidget(self.tip_label)    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
          
        import_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)    
        import_shortcut.activated.connect(self.import_data)    
          
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)    
        save_shortcut.activated.connect(self.save_data)    
          
        back_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)    
        back_shortcut.activated.connect(self.undo_last_operation)    
    def setup_operation_history(self):
        """Setup operation history tracking"""
        self.operation_history = []  # Initialize operation history list
        history_btn = QPushButton("Operation History")    
        history_btn.clicked.connect(self.show_history)    
        self.statusBar().addPermanentWidget(history_btn)    
    def zoom_in(self):
        """Zoom in on the current visualization"""
        self.visualization_widget.zoom_in()    
    def zoom_out(self):
        """Zoom out on the current visualization"""
        self.visualization_widget.zoom_out()    
    def save_plot(self):
        """Save the current plot to file"""
        file_path, _ = QFileDialog.getSaveFileName(    
            self, "Save Image", "", "PNG Files (*.png);;PDF Files (*.pdf)"    
        )
        if file_path:    
            self.visualization_widget.save_plot(file_path)    
    def save_data(self):
        """Save the current data to file"""
        file_path, _ = QFileDialog.getSaveFileName(    
            self, "Save Data", "", "CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx)"    
        )
        if file_path:    
            try:
                SpectrumConverter.save_spectrum(    
                    self.wavelengths, self.current_spectra,
                    file_path, format_type=file_path.split('.')[-1].lower()    
                )
                self.statusBar().showMessage(f"Data saved: {file_path}")    
            except Exception as e:    
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")    
    def undo_last_operation(self):
        """Undo the last operation"""
        if self.operation_history:    
            last_operation = self.operation_history.pop()    
            self.current_spectra = last_operation['data']    
            self.update_data_info()    
            self.visualization_widget.plot_spectra(self.wavelengths, self.current_spectra)    
    def show_history(self):
        """Show operation history dialog"""
        if not self.operation_history:    
            QMessageBox.information(self, "Operation History", "No operation history")    
            return
        history_text = "Operation History:\n\n"    
        for operation in self.operation_history:    
            history_text += f"Time: {operation['time']}\n"    
            history_text += f"Operation: {operation['operation']}\n\n"    
        QMessageBox.information(self, "Operation History", history_text)    

    def connect_signals(self):
          
        self.import_btn.clicked.connect(self.import_data)    
          
        self.data_partitioning_view.load_data_btn.clicked.connect(self.import_data)
          
        self.preprocess_btn.clicked.connect(self.apply_preprocessing)
          
        self.feature_btn.clicked.connect(self.apply_feature_selection)
          
          
        # self.analyze_btn.clicked.connect(    
        #     lambda: self.controller.check_data_split() and 
        #     self.controller.modeling_controller.train_model()
        # )
          
        self.analysis_type.currentTextChanged.connect(self.update_analysis_methods)    
          
        self.hyperopt_view.apply_button.clicked.connect(    
            lambda: self.controller.check_model_trained() and 
            self.controller.hyperopt_controller.apply_optimization()
        )
          
        self.evaluation_view.evaluate_button.clicked.connect(    
            lambda: self.controller.check_model_trained() and 
            self.controller.evaluation_controller.evaluate_model()
        )

    def display_message(self, message, title="Information"):
        """Display an information message to the user"""
        if title == "Error":
            QMessageBox.critical(self, title, message)
        elif title == "Warning":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def update_data_view(self, data):
        self.data_group.layout().addWidget(DataView().update_data(data))    

    def update_split_view(self, train_data, test_data):
        """Update the view with training and testing data splits"""
          
        pass

    def show_llm_dialog(self):
        """Show the LLM (Large Language Model) dialog"""
        dialog = LLMDialog(self, self.llm_service)    
        dialog.exec_()    

    def add_converted_algorithm(self, algorithm_type: str, code: str):
        """Add a converted algorithm to the appropriate plugin list"""
        try:
              
            if algorithm_type == "Preprocessing":
                self.preprocessing_view.add_custom_algorithm(code)    
            elif algorithm_type == "Feature Selection":
                self.feature_selection_view.add_custom_algorithm(code)    
            elif algorithm_type == "Modeling":
                self.modeling_view.add_custom_algorithm(code)    
            elif algorithm_type == "Data Splitting":
                self.data_partitioning_plugins.add_custom_algorithm(code)    
            QMessageBox.information(self, 'Success', 'Algorithm successfully added to the system')    
        except Exception as e:    
            QMessageBox.critical(self, 'Error', f'Error adding algorithm: {str(e)}')    

    def show_llm_config(self):
        """Show the LLM configuration dialog"""
        from app.views.llm_config_dialog import LLMConfigDialog    
        dialog = LLMConfigDialog(self)    
        if dialog.exec_() == QDialog.Accepted:    
              
            self.llm_service = LLMService(LLMConfig.API_KEY)    

    def import_data(self):
        """Load spectral data"""
        file_path, _ = QFileDialog.getOpenFileName(  # Open file dialog to get file path
            self, "Select Spectrum Data File", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx)"  # Set file types
        )
        if file_path:  # If a file path is selected
            try:
                  
                print("INFO: Updating")
                  
                self.current_spectra = None
                self.original_spectra = None
                self.spectra_data = None
                self.wavelengths = None
                self.sample_labels = None

                  
                if hasattr(self, 'controller'):
                      
                    self.controller.current_data = None
                    self.controller.train_data = None
                    self.controller.test_data = None
                    self.controller.preprocessed_train = None
                    self.controller.preprocessed_test = None
                    self.controller.feature_selected_train = None
                    self.controller.feature_selected_test = None
                    self.controller.trained_model = None
                      
                    if hasattr(self.controller, 'data_controller'):
                        if hasattr(self.controller.data_controller, 'data_model'):
                            self.controller.data_controller.data_model.data = None
                            self.controller.data_controller.data_model.X_train = None
                            self.controller.data_controller.data_model.X_test = None
                            self.controller.data_controller.data_model.y_train = None
                            self.controller.data_controller.data_model.y_test = None
                            self.controller.data_controller.data_model.file_path = None
                      
                    if hasattr(self.controller, 'preprocessing_controller'):
                        if hasattr(self.controller.preprocessing_controller, 'preprocessing_model'):
                            from app.models.preprocessing_model import PreprocessingModel
                            self.controller.preprocessing_controller.preprocessing_model = PreprocessingModel()
                      
                    if hasattr(self.controller, 'feature_selection_controller'):
                        if hasattr(self.controller.feature_selection_controller, 'feature_selection_model'):
                            from app.models.feature_selection_model import FeatureSelectionModel
                            self.controller.feature_selection_controller.feature_selection_model = FeatureSelectionModel()
                      
                    if hasattr(self.controller, 'modeling_controller'):
                        if hasattr(self.controller.modeling_controller, 'modeling_model'):
                              
                            from app.models.modeling_model import ModelingModel
                            self.controller.modeling_controller.modeling_model = ModelingModel()
                  
                print("INFO: Updating")
                  
                if hasattr(self, 'visualization_widget'):
                    self.visualization_widget.clear()
                  
                self.reset_preprocessing_params()
                  
                if hasattr(self, 'feature_method'):
                    self.feature_method.setCurrentIndex(0)
                if hasattr(self, 'feature_num'):
                    self.feature_num.setValue(10)
                  
                if hasattr(self, 'analysis_type'):
                    self.analysis_type.setCurrentIndex(0)
                if hasattr(self, 'analysis_method'):
                    self.analysis_method.setCurrentIndex(0)
                  
                if hasattr(self, 'data_partitioning_view'):
                      
                    if hasattr(self.data_partitioning_view, 'method_combo'):
                        self.data_partitioning_view.method_combo.setCurrentIndex(0)
                      
                    if hasattr(self.data_partitioning_view, 'result_text'):
                        self.data_partitioning_view.result_text.clear()
                  
                if hasattr(self, 'evaluation_view'):
                    if hasattr(self.evaluation_view, 'results_text'):
                        self.evaluation_view.results_text.clear()
                    if hasattr(self.evaluation_view, 'plot_widget'):
                        self.evaluation_view.plot_widget.clear()
                  
                if hasattr(self, 'operation_history'):
                    self.operation_history.clear()
                print("SUCCESS: Operation completed")
                  
                # Show progress bar
                self.progress_dialog = ProgressDialog("Loading Data", self)  # Create progress dialog
                self.progress_dialog.show()  # Show progress dialog
                # Load data
                self.wavelengths, self.spectra_data = SpectrumConverter.read_spectrum_file(  # Call spectrum converter to read spectral data
                    file_path, format_type=file_path.split('.')[-1].lower()  # Get file format
                )
                
                # **CRITICAL DEBUG: Check wavelength data**

                
                self.current_spectra = self.spectra_data.copy()  # Copy current spectral data
                # Initialize/reset original data reference
                self.original_spectra = self.spectra_data.copy()  # Save copy of original data
                # Read sample labels - Read first column from original file as labels
                try:
                    # Read data based on file format
                    if file_path.endswith('.csv'):
                        data = pd.read_csv(file_path)
                    elif file_path.endswith(('.xls', '.xlsx')):
                        data = pd.read_excel(file_path)
                    elif file_path.endswith('.txt'):
                        data = pd.read_csv(file_path, sep='\t')
                    # Get first column as sample labels
                    self.sample_labels = data.iloc[:, 0].values
                    print(f"Sample labels loaded successfully, total {len(self.sample_labels)} samples")
                except Exception as e:
                    print(f"Failed to load sample labels: {str(e)}")
                    self.sample_labels = None
                # **CRITICAL FIX: Update controller's data model with COMPLETE original data including labels**
                if hasattr(self, 'controller') and hasattr(self.controller, 'data_controller'):
                    if hasattr(self.controller.data_controller, 'data_model'):
                        # Use the complete original data including wine variety labels, not just spectra
                        if file_path.endswith('.csv'):
                            complete_data = pd.read_csv(file_path)
                        elif file_path.endswith(('.xls', '.xlsx')):
                            complete_data = pd.read_excel(file_path)
                        elif file_path.endswith('.txt'):
                            complete_data = pd.read_csv(file_path, sep='\t')

                        
                        # **CRITICAL FIX: Initialize labels using unified data processor**
                        from app.utils.unified_data_processor import unified_processor
                        
                        # Get first column as labels
                        first_column = complete_data.iloc[:, 0]
                        processor_info = unified_processor.initialize_from_data(first_column)
                        
                        # Set complete data (including labels) to data controller
                        self.controller.data_controller.data_model.data = complete_data
                        self.controller.current_data = complete_data
                        # Also save the file path for reference
                        self.controller.data_controller.data_model.file_path = file_path
                  
                # **CRITICAL FIX: Use unified data processor safe labels for visualization**
                from app.utils.unified_data_processor import unified_processor
                safe_labels = None
                if unified_processor.is_initialized:
                    safe_labels = unified_processor.get_safe_labels_for_visualization()
                else:
                    safe_labels = self.sample_labels if hasattr(self, 'sample_labels') else None
                
                # Update visualization - Pass safe labels
                self.visualization_widget.plot_spectra(self.wavelengths, self.current_spectra, labels=safe_labels)  # Plot spectra
                  
                print("English text")
                # Update progress
                self.progress_dialog.update_progress(100, "Data loading complete")  # Update progress to 100%
                self.progress_dialog.close()  # Close progress dialog
                  
                self.statusBar().showMessage(f"Data loaded: {file_path}")  # Show loading success message in status bar
                  
                if hasattr(self, 'data_status'):
                    self.data_status.setText(f"Data loaded: {self.spectra_data.shape[0]} samples, {self.spectra_data.shape[1]} features")
                print(f"English text")
                print("INFO: Updating")
            except Exception as e:  # If an exception occurs
                if hasattr(self, 'progress_dialog') and self.progress_dialog:  # If progress dialog exists
                    self.progress_dialog.close()  # Close progress dialog
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")  # Show error message


    def update_preview(self):
        """Update the data preview"""
        if self.current_spectra is not None:    
              
            self.ax.clear()    
              
            if hasattr(self, 'sample_labels') and self.sample_labels is not None:
                  
                for i in range(min(5, self.current_spectra.shape[0])):    
                    self.ax.plot(self.wavelengths, self.current_spectra[i], 
                               label=f"{self.sample_labels[i]}")    
            else:
                  
                for i in range(min(5, self.current_spectra.shape[0])):    
                    self.ax.plot(self.wavelengths, self.current_spectra[i], 
                               label=f"Spectrum {i+1}")    
            self.ax.set_xlabel("Wavelength (nm)")    
            self.ax.set_ylabel("Absorbance")    
            self.ax.grid(True)    
            self.ax.legend()    
            self.canvas.draw()    
              
            self.data_table.setRowCount(self.current_spectra.shape[0])    
            self.data_table.setColumnCount(self.current_spectra.shape[1])    
            for i in range(self.current_spectra.shape[0]):    
                for j in range(self.current_spectra.shape[1]):    
                    self.data_table.setItem(i, j, 
                        QTableWidgetItem(f"{self.current_spectra[i,j]:.4f}"))    
    def apply_preprocessing(self):
        """Apply preprocessing using the fixed controller"""
        if not self.controller.check_data_ready():
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        try:
            # Show progress dialog
            self.progress_dialog = ProgressDialog("Preprocessing", self)
            self.progress_dialog.show()
            # Get preprocessing parameters
            params_dict = {
                'despiking': {
                    'enabled': self.despiking_check.isChecked(),
                    'method': self.despiking_method.currentText(),
                    'window': self.despiking_window.value(),
                    'threshold': self.despiking_threshold.value()
                },
                'baseline_correction': {
                    'enabled': self.baseline_check.isChecked(),
                    'method': self.baseline_method.currentText(),
                    'polynomial_order': self.poly_order.value()
                },
                'smoothing': {
                    'enabled': self.smooth_check.isChecked(),
                    'window_length': self.window_size.value(),
                    'polyorder': self.smooth_poly_order.value()
                },
                'scatter_correction': {
                    'enabled': self.scatter_check.isChecked(),
                    'method': self.scatter_method.currentText()
                },
                'standardization': {
                    'enabled': self.standardize_check.isChecked(),
                    'method': self.standardize_method.currentText()
                },
                'normalization': {
                    'enabled': self.norm_check.isChecked(),
                    'method': self.norm_method.currentText()
                },
                'derivative': {
                    'enabled': self.derivative_check.isChecked(),
                    'order': self.derivative_order.value()
                }
            }
            # Extract enabled methods
            methods = []
            for method_name, params in params_dict.items():
                if params.get('enabled', False):
                    methods.append(method_name)
            print(f"Applying preprocessing methods: {methods}")
            print(f"Recorded {len(methods)} preprocessing methods to applied_methods")
            # Update progress
            self.progress_dialog.update_progress(25, "Preprocessing in progress...")
            # Call the fixed preprocessing controller
            success = self.controller.preprocessing_controller.apply_preprocessing(methods, params_dict)
            if success:
                # Record operation in history before updating data
                if hasattr(self, 'current_spectra') and self.current_spectra is not None:
                    import datetime
                    operation_record = {
                        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'operation': f"Preprocessing: {', '.join(methods)}",
                        'data': self.current_spectra.copy()  # Save current state before change
                    }
                    self.operation_history.append(operation_record)
                    # Keep only last 10 operations to avoid memory issues
                    if len(self.operation_history) > 10:
                        self.operation_history.pop(0)
                
                # Update progress
                self.progress_dialog.update_progress(75, "Updating visualization...")
                # Update current data reference
                if hasattr(self.controller.data_controller.data_model, 'data') and self.controller.data_controller.data_model.data is not None:
                    self.current_spectra = self.controller.data_controller.data_model.data.values
                    # **CRITICAL FIX: Extract only spectral data, excluding label columns**
                    if hasattr(self.controller.data_controller.data_model, 'features') and self.controller.data_controller.data_model.features is not None:
                        # Use already separated feature data (excluding labels)
                        self.processed_spectra = self.controller.data_controller.data_model.features.values
                        print(f"🔧 Use separated feature data as processed_spectra: {self.processed_spectra.shape}")
                    else:
                        # If no separated feature data, exclude first column (label column) from complete data
                        if self.current_spectra.shape[1] > 1:
                            self.processed_spectra = self.current_spectra[:, 1:]  # exclude first column labels
                            print(f"🔧 Processed spectra after excluding label column: {self.processed_spectra.shape}")
                        else:
                            self.processed_spectra = self.current_spectra.copy()
                # Update visualization if available
                if hasattr(self, 'visualization_widget') and hasattr(self, 'original_spectra'):
                    try:
                        # **CRITICAL FIX: Use unified data processor to get safe labels**
                        from app.utils.unified_data_processor import unified_processor
                        
                        safe_labels = None
                        if unified_processor.is_initialized:
                            safe_labels = unified_processor.get_safe_labels_for_visualization()
                            print(f"🔧 Using unified processor safe labels: {safe_labels[:3] if len(safe_labels) > 0 else []}")
                        
                        if hasattr(self, 'processed_spectra') and self.processed_spectra is not None:
                            # **CRITICAL FIX: Ensure original_spectra also excludes labels**
                            original_spectra_clean = self.original_spectra
                            if hasattr(self, 'original_spectra') and self.original_spectra.shape[1] > 100:  # If too many columns, may contain labels
                                if self.original_spectra.shape[1] == self.processed_spectra.shape[1] + 1:
                                    original_spectra_clean = self.original_spectra[:, 1:]  # exclude first column labels
                                    print(f"🔧 Original spectra after excluding label column: {original_spectra_clean.shape}")
                            
                            self.visualization_widget.plot_spectra_comparison(
                                self.wavelengths, 
                                original_spectra_clean,
                                self.processed_spectra,
                                title="Preprocessing Result"
                            )
                        else:
                            # If no processed data, only display original data
                            self.visualization_widget.plot_spectra(
                                self.wavelengths, 
                                self.original_spectra,
                                title="Original Spectra",
                                labels=safe_labels  # use safe labels
                            )
                    except Exception as viz_error:
                        print(f"Visualization update failed: {viz_error}")
                        import traceback
                        traceback.print_exc()
                # Complete
                self.progress_dialog.update_progress(100, "Preprocessing complete")
                self.progress_dialog.close()
                self.statusBar().showMessage("Preprocessing complete")
                QMessageBox.information(self, "Success", f"Successfully applied {len(methods)} preprocessing methods")
            else:
                self.progress_dialog.close()
                QMessageBox.critical(self, "Error", "Preprocessing failed. Check console for details.")
        except Exception as e:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Preprocessing failed: {str(e)}")
            print(f"Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
    def extract_features(self):
        """Extract features from the spectral data"""
        QMessageBox.information(self, "Info", "This feature is now handled by the new 'apply_feature_selection' method.")
        return

    def update_analysis_methods(self, analysis_type):
        """Update available analysis methods based on analysis type"""
        self.analysis_method.clear()    
        if analysis_type == "Quantitative":    
            self.analysis_method.addItems([    
                "PLSR",
                "SVR",
                "RF",
                "NN",
                "GPR",
                "XGBoost",
                "LightGBM"
            ])
            self.evaluation_metric.clear()    
            self.evaluation_metric.addItems([    
                "RMSE",
                "MAE",
                "R²",
                "R",
                "RE"
            ])
        else:    
            self.analysis_method.addItems([    
                "SVM",
                "RF",
                "KNN",
                "DT",
                "NN",
                "XGBoost",
                "LightGBM"
            ])
            self.evaluation_metric.clear()    
            self.evaluation_metric.addItems([    
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "AUC"
            ])
    def start_analysis(self):
        """Start the spectral analysis process"""
        global pd
        if self.current_spectra is None:    
            QMessageBox.warning(self, "Warning", "Please load spectral data first")    
            return
          
        if self.reference_method.currentText() == "":
            QMessageBox.warning(self, "Warning", "Please select a reference method")
            return
        analysis_type = self.analysis_type.currentText()    
        method = self.analysis_method.currentText()    
          
        method_mapping = {
            "Neural Network": "nn", 
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Random Forest": "rf"
        }
          
        if method in method_mapping:
            method = method_mapping[method]
        else:
              
            method = method.lower()
        try:
              
            self.progress_dialog = ProgressDialog("Analysis", self)    
            self.progress_dialog.show()    
              
            if not hasattr(self, 'controller') or not hasattr(self.controller, 'data_controller'):
                raise ValueError("Data controller not initialized")
            data_model = self.controller.data_controller.data_model
              
            if hasattr(self.controller, 'feature_selection_controller') and hasattr(self.controller.feature_selection_controller, 'feature_selection_model'):
                if hasattr(self.controller.feature_selection_controller.feature_selection_model, 'selected_features'):
                    selected_features = self.controller.feature_selection_controller.feature_selection_model.selected_features
                    self.progress_dialog.update_progress(10, "Using selected features...")
                      
                    if hasattr(data_model, 'X_train_selected') and data_model.X_train_selected is not None:
                        X_train = data_model.X_train_selected
                        X_test = data_model.X_test_selected
                    else:
                          
                        X_train = data_model.X_train
                        X_test = data_model.X_test
                else:
                      
                    X_train = data_model.X_train
                    X_test = data_model.X_test
            else:
                  
                X_train = data_model.X_train
                X_test = data_model.X_test
            y_train = data_model.y_train
            y_test = data_model.y_test
            # **ULTIMATE FIX: Force all labels to be consistent string type for qualitative analysis**
            if analysis_type.startswith("Qualitative") and hasattr(data_model, 'label_mapping') and data_model.label_mapping:
                label_mapping = data_model.label_mapping
                index_to_label = label_mapping.get('index_to_label', {})
                if index_to_label:
                    print(f"Converting encoded labels back to original wine varieties ({len(index_to_label)} labels)")
                    # **ULTIMATE FIX: Triple-ensure all labels are strings**
                    # Step 1: Force all values in mapping to be strings
                    index_to_label_safe = {k: str(v) for k, v in index_to_label.items()}
                    # Step 2: Convert training labels with forced string conversion
                    if y_train is not None:
                        # **CRITICAL FIX: Use enhanced label processor instead of forced numeric conversion**
                        from app.utils.label_processor import EnhancedLabelProcessor
                        label_processor = EnhancedLabelProcessor()
                        
                        detected_task_type = label_processor.detect_task_type(y_train)
                        print(f"🤖 Label processing task type: {detected_task_type}")
                        
                        if detected_task_type == 'classification':
                            print("🔧 Classification labels detected - preserving string format")
                            y_train_safe = y_train  # Keep original string labels
                        else:
                            print("🔧 Regression labels detected - attempting numeric conversion")
                            try:
                                # **CRITICAL FIX: Safe numeric conversion with string label detection**
                                # Check for string labels before conversion
                                sample_labels = [str(label) for label in y_train[:5]]
                                has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                       for label in sample_labels if label.strip())
                                
                                if has_string_labels:
                                    raise ValueError(f"String labels detected: {sample_labels[:3]}. Cannot convert to numeric for quantitative analysis.")
                                
                                y_train_safe = pd.to_numeric(y_train, errors='coerce')
                                # Check if conversion failed (NaN values indicate string labels)
                                if pd.isna(y_train_safe).any():
                                    failed_labels = y_train[pd.isna(y_train_safe)][:3].tolist()
                                    raise ValueError(f"Cannot convert string labels to numeric: {failed_labels}. Use Qualitative Analysis for classification tasks.")
                            except (ValueError, TypeError):
                                print("❌ Numeric conversion failed - treating as classification")
                                y_train_safe = y_train
                        print(f"Training labels converted: {list(y_train_safe[:3])}")
                        print(f"Training labels dtype: {y_train_safe.dtype}")
                        y_train = y_train_safe
                    # Step 3: Convert test labels with forced string conversion
                    if y_test is not None:
                        if isinstance(y_test, pd.Series):
                            # Use safe mapping and force string conversion
                            y_test_original = y_test.map(lambda x: str(index_to_label_safe.get(int(x), str(x))))
                        else:
                            # Force all elements to be strings
                            y_test_original = np.array([str(index_to_label_safe.get(int(label), str(label))) for label in y_test], dtype='<U50')
                        print(f"Test labels converted: {list(y_test_original[:3])}")
                        print(f"Test labels dtype: {y_test_original.dtype}")
                        y_test = y_test_original
                    print("✅Label conversion completed - ALL labels are now guaranteed strings")
                else:
                    print("Warning: No index_to_label mapping found")
            else:
                if analysis_type.startswith("Qualitative"):
                    print("Warning: No label mapping available for qualitative analysis")
            # **CRITICAL FIX: Ensure consistent label types for qualitative analysis**
            if analysis_type.startswith("Qualitative"):
                # Final safety check - ensure labels are string arrays with consistent dtype
                print("Final label type verification for qualitative analysis...")
                if y_train is not None:
                    if not isinstance(y_train, np.ndarray) or y_train.dtype.kind not in 'SU':
                        y_train = np.array([str(label) for label in y_train], dtype=str)
                        print(f"y_train converted to string array: {y_train.dtype}")
                if y_test is not None:
                    if not isinstance(y_test, np.ndarray) or y_test.dtype.kind not in 'SU':
                        y_test = np.array([str(label) for label in y_test], dtype=str)
                        print(f"y_test converted to string array: {y_test.dtype}")
                print(f"Label verification completed - Training: {type(y_train)}, Test: {type(y_test)}")
            # **CRITICAL FIX: Modeling must use feature-selected data！**
            print("🎯 Checking modeling data source...")
            
            # Check if feature selection has been performed
            if (hasattr(data_model, 'X_train_selected') and data_model.X_train_selected is not None and
                X_train is data_model.X_train_selected):
                print("✅ Correctly using feature-selected data for modeling")
                print(f"   Data shape after feature selection: {X_train.shape}")
                print(f"   Number of selected features: {X_train.shape[1]} features")
                # Feature-selected data already includes preprocessing, use directly
            else:
                # **Modified to optional suggestion rather than mandatory requirement**
                print("ℹ️  Suggestion: For better modeling results, it is recommended to perform feature selection first")
                print(f"   Current data shape: {X_train.shape}")
                print(f"   Current number of features: {X_train.shape[1]} features")
                print("   You can continue modeling, or perform feature selection first for better results")
                # Continue modeling, feature selection not mandatory
            # Build feature arrays (avoid reshaping labels to prevent dtype changes)
            X_train = np.array(X_train)
            X_test = np.array(X_test) 
            # DON'T convert y_train/y_test again if they're already properly formatted
            if not isinstance(y_train, np.ndarray):
                y_train = np.array(y_train)
            if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
              
            if X_train is None or y_train is None:
                x_status = "English text" if X_train is None else "English text"
                y_status = "English text" if y_train is None else "English text"
                error_msg = f"""
Cannot perform analysis: Missing required data

Detected issues:
✅Training data (X_train): {x_status}
✅Label data (y_train): {y_status}

This is the root cause of low model accuracy!

Required steps to complete:
1. Load spectral data file (with labels in first column)
2. Perform data partitioning (train/test)
3. Complete data preprocessing
4. Perform feature selection

Warning: Randomly generated data cannot train meaningful models!
Please use real spectral data with corresponding labels for analysis.
                """
                self.display_error(error_msg.strip(), "Data Missing Error")
                self.progress_dialog.hide()
                return
              
            print("=== Data Quality Check ===")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Unique labels: {len(np.unique(y_train))}")
            # Check training sample count
            if X_train.shape[0] < 20:
                error_msg = f"Insufficient training data: Only {X_train.shape[0]} samples available. At least 20 samples are recommended for reliable modeling."
                self.display_error(error_msg.strip(), "Insufficient Training Data")
                self.progress_dialog.hide()
                return
            # Check feature count
            if X_train.shape[1] < 10:
                error_msg = f"Insufficient features: Only {X_train.shape[1]} features available. At least 10 features are recommended for reliable modeling."
                self.display_error(error_msg.strip(), "Insufficient Features")
                self.progress_dialog.hide()
                return
            elif X_train.shape[1] < 50:
                print(f"Warning: Only {X_train.shape[1]} features available. Consider feature extraction or dimensionality expansion for better results.")
            # Check label quality for qualitative analysis
            if analysis_type.startswith("Qualitative"):
                unique_labels = len(np.unique(y_train))
                if unique_labels < 2:
                    self.display_error("Classification requires at least 2 different classes. Current data has only 1 class.", "Insufficient Label Classes")
                    self.progress_dialog.hide()
                    return
                # Remove the overly strict check for too many classes
                # Users should be allowed to have many classes if their data requires it
                elif unique_labels > X_train.shape[0] * 0.95:  # Only warn if classes exceed 95% of samples (extremely unrealistic)
                    print(f"Warning: High number of classes ({unique_labels} classes for {X_train.shape[0]} samples). This might indicate a regression problem disguised as classification.")
                    # Don't return - allow the analysis to continue
              
            hyperopt_params = {
                'method': self.optimization_method.currentText(),    
                'metric': self.evaluation_metric.currentText(),    
                'cv_folds': self.cv_folds.value(),    
                'max_iterations': self.max_iterations.value()    
            }
              
              
            # **SMART ANALYSIS TYPE DETECTION AND AUTO-CORRECTION**
            # Check if the selected analysis type matches the data type
            # **CRITICAL FIX: Use enhanced label processor for consistent task type detection**
            from app.utils.label_processor import EnhancedLabelProcessor
            label_processor = EnhancedLabelProcessor()
            if y_train is not None:
                detected_task_type = label_processor.detect_task_type(y_train)
                detected_is_classification = (detected_task_type == 'classification')
                
                # Only auto-switch when user selects quantitative analysis but data is classification
                # If user selects qualitative analysis, respect user choice without auto-switching
                if analysis_type.startswith("Quantitative") and detected_is_classification:
                    print("🔄 SMART DETECTION: Data contains categorical labels but Quantitative analysis was selected")
                    print(f"   Sample labels: {list(set(str(label) for label in y_train[:10]))}")
                    print("   Auto-switching to Qualitative (Classification) analysis...")
                    analysis_type = "Qualitative"  # Auto-correct the analysis type
                    self.display_message("Data type auto-detection: Detected classification labels, auto-switched to qualitative analysis mode", "Smart Analysis Type Switch")
                elif analysis_type.startswith("Qualitative"):
                    # User selected qualitative analysis, respect user choice regardless of data type
                    print(f"✅USER CHOICE: User selected Qualitative analysis, respecting user choice")
                    print(f"   Data type detected as: {'Classification' if detected_is_classification else 'Regression'}")
                    print("   Will proceed with qualitative analysis as requested")
                    # No auto-switching
                
            if analysis_type.startswith("Quantitative"):  # If quantitative analysis
                self.progress_dialog.update_progress(25, "Performing quantitative analysis...")  # Update progress to 25%
                # Set analysis type to quantitative
                if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                    self.controller.modeling_controller.modeling_model.set_analysis_type('quantitative')
                    print("Explicitly setting analysis type to quantitative")
                analyzer = QuantitativeAnalyzer(  # Create quantitative analyzer
                    method=method.lower()  # Set analysis method
                )
                # Train model
                self.progress_dialog.update_progress(40, "Training quantitative model...")
                # Prepare parameters, ensure task type is correct
                model_params = {'task_type': 'regression'}
                try:
                    # **ENHANCED SAFE CONVERSION with better error messages**
                    # For quantitative analysis, ensure y_train is numeric
                    # Note: This branch only executes when analysis_type.startswith("Quantitative")
                    if isinstance(y_train, pd.Series):
                        # Safety check: If contains obvious string labels, provide better error messages
                        sample_labels = [str(label) for label in y_train.head(5)]
                        if any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                               for label in sample_labels if label.strip()):
                            raise ValueError(f"Quantitative analysis requires numeric labels, but detected string labels: {sample_labels[:3]}\n"
                                           f"Solution: Please select 'Qualitative Analysis' for classification tasks")
                        y_train_safe = pd.to_numeric(y_train, errors='coerce')
                    else:
                        # Safe conversion for mixed types
                        def safe_float_convert(val):
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                # If conversion fails, this might be a classification task
                                # with string labels, return NaN to indicate conversion failure
                                return np.nan
                        
                        y_train_safe = np.array([safe_float_convert(val) for val in y_train])
                        
                        # Check if any conversions failed (NaN values)
                        if np.isnan(y_train_safe).any():
                            failed_labels = [str(label) for label in y_train[:5] if safe_float_convert(label) != safe_float_convert(label)]  # NaN check
                            print("✅CONVERSION ERROR: Some labels could not be converted to float.")
                            print(f"   Failed labels: {list(set(failed_labels[:5]))}")
                            print("   This suggests the data is better suited for classification (qualitative) analysis.")
                            # Provide a helpful error message with solution
                            raise ValueError(f"Quantitative analysis requires numeric labels, but detected string labels: {list(set(failed_labels[:3]))}\n"
                                           f"Solution: Please select 'Qualitative Analysis' for classification tasks")
                    
                    analyzer.fit(X_train, y_train_safe, **model_params)  # Fit model with explicit regression task type
                except ImportError as e:
                    # Handle import errors - if required libraries are not installed
                    error_msg = f"Required library not installed: {str(e)}. Please install it and try again."
                    self.display_error(error_msg)
                    self.progress_dialog.hide()
                    return
                except Exception as e:
                    # Handle other training errors
                    error_msg = f"Error training model: {str(e)}"
                    self.display_error(error_msg)
                    self.progress_dialog.hide()
                    return
                # Update progress bar to avoid getting stuck at 40%
                self.progress_dialog.update_progress(50, "Model training completed...")
                # Cross-validation
                self.progress_dialog.update_progress(60, "Performing cross-validation...")
                # Use the converted y_train_safe for cross-validation to maintain consistency
                results = analyzer.cross_validate(X_train, y_train_safe)
                # Make predictions
                if X_test is not None and y_test is not None:
                    self.progress_dialog.update_progress(70, "Making predictions...")
                    y_pred = analyzer.predict(X_test)
                      
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    
                    # Safe evaluation for regression - ensure both y_test and y_pred are numeric
                    try:
                        # Define safe conversion function
                        def safe_float_convert(val):
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                return np.nan
                        
                        # **CRITICAL FIX: Use enhanced label processor for safe y_test conversion**
                        from app.utils.label_processor import EnhancedLabelProcessor
                        label_processor = EnhancedLabelProcessor()
                        
                        test_task_type = label_processor.detect_task_type(y_test)
                        print(f"🤖 y_test conversion task type: {test_task_type}")
                        
                        if test_task_type == 'classification':
                            print("❌ CRITICAL ERROR: Cannot compute regression metrics with classification labels!")
                            print(f"   Detected classification labels in y_test: {list(set([str(x) for x in y_test[:5]]))}")
                            raise ValueError(f"REGRESSION ANALYSIS FAILED: Test labels contain classification data like 'ClassC'. Please use QUALITATIVE ANALYSIS instead.")
                        else:
                            print("🔧 Regression y_test - proceeding with numeric conversion")
                            try:
                                if isinstance(y_test, pd.Series):
                                    # **CRITICAL FIX: Check for string labels before pd.to_numeric**
                                    sample_labels = [str(label) for label in y_test[:5]]
                                    has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                           for label in sample_labels if label.strip())
                                    
                                    if has_string_labels:
                                        raise ValueError(f"String labels detected in y_test: {sample_labels[:3]}")
                                    
                                    y_test_numeric = pd.to_numeric(y_test, errors='coerce')
                                    # Check if conversion failed
                                    if pd.isna(y_test_numeric).any():
                                        failed_labels = y_test[pd.isna(y_test_numeric)][:3].tolist()
                                        raise ValueError(f"Cannot convert test labels to numeric: {failed_labels}. Use Qualitative Analysis for classification tasks.")
                                else:
                                    y_test_numeric = np.array([safe_float_convert(val) for val in y_test])
                                    # Check for failed conversions
                                    if np.isnan(y_test_numeric).any():
                                        failed_indices = np.where(np.isnan(y_test_numeric))[0]
                                        failed_labels = [str(y_test[i]) for i in failed_indices[:3]]
                                        raise ValueError(f"Cannot convert test labels to numeric: {failed_labels}")
                            except Exception as conv_error:
                                print(f"❌ y_test numeric conversion failed: {conv_error}")
                                raise ValueError(f"REGRESSION TASK FAILED: Cannot convert test labels to numeric. Detected non-numeric labels suggest this is a CLASSIFICATION task.")
                        
                        # Convert y_pred to numeric if possible
                        if isinstance(y_pred, pd.Series):
                            y_pred_numeric = pd.to_numeric(y_pred, errors='coerce')
                        else:
                            y_pred_numeric = np.array([safe_float_convert(val) for val in y_pred])
                        
                        # Check if conversions were successful
                        if np.isnan(y_test_numeric).any() or np.isnan(y_pred_numeric).any():
                            raise ValueError("Cannot compute regression metrics with non-numeric values")
                        
                        rmse = np.sqrt(mean_squared_error(y_test_numeric, y_pred_numeric))
                        r2 = r2_score(y_test_numeric, y_pred_numeric)
                        mae = mean_absolute_error(y_test_numeric, y_pred_numeric)
                        
                        results.update({
                            'test_rmse': rmse,
                            'test_r2': r2,
                            'test_mae': mae,
                            'task_type': 'regression'    
                        })
                    except Exception as e:
                        print(f"Warning: Could not compute regression metrics: {e}")
                        results.update({
                            'error': f"Regression metrics computation failed: {str(e)}",
                            'task_type': 'regression'
                        })
                  
                if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                    self.controller.modeling_controller.modeling_model.trained_model = analyzer.model
                    self.controller.modeling_controller.modeling_model.set_method(method)
                    self.controller.modeling_controller.modeling_model.set_evaluation_results(results)
                # For built-in models called via modeling_service, also set the task_type
                if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_service'):
                    # Store original train_model method
                    original_train_model = self.controller.modeling_controller.modeling_service.train_model
                    # Create wrapper that forces regression
                    def force_regression_train_model(model_name, params, X, y):
                        if params is None:
                            params = {}
                        # Force regression task type
                        params['task_type'] = 'regression'
                        print("EXPLICITLY FORCING REGRESSION MODEL via params override")
                        return original_train_model(model_name, params, X, y)
                    # Replace with our wrapper
                    self.controller.modeling_controller.modeling_service.train_model = force_regression_train_model
            else:  # Qualitative analysis
                self.progress_dialog.update_progress(25, "Performing qualitative analysis...")  # Update progress to 25%
                # Set analysis type to qualitative
                if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                    self.controller.modeling_controller.modeling_model.set_analysis_type('qualitative')
                # Intelligent task type detection
                # **CRITICAL FIX: Use enhanced label processor for consistent task type detection**
                from app.utils.label_processor import EnhancedLabelProcessor
                label_processor = EnhancedLabelProcessor()
                if y_train is not None:
                    detected_task_type = label_processor.detect_task_type(y_train)
                    is_classification = (detected_task_type == 'classification')
                    print(f"🤖 Enhanced task detection result: {detected_task_type} ({'Classification task' if is_classification else 'Regression task'})")
                    if not is_classification:
                        # If detected as regression task, automatically handle it
                        print("Warning: Data is more suitable for regression analysis, but user chose qualitative analysis")
                        print("Will perform binning on continuous data to adapt for classification analysis...")
                        # Perform binning on continuous values
                        try:
                            from sklearn.preprocessing import KBinsDiscretizer
                            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                            
                            # å®å¨çæ°å¼è½¬æ¢ï¼é¿åå¯¹å­ç¬¦ä¸²æ ç­¾å¼ºå¶è½¬æ¢
                            def safe_numeric_conversion(labels):
                                """Safely convert labels to numeric, skip conversion if string labels"""
                                try:
                                    # Check if already string labels
                                    sample_labels = [str(label) for label in labels[:5]]
                                    if any(not label.replace('.', '').replace('-', '').isdigit() for label in sample_labels):
                                        print("Detected string labels, skipping numeric conversion")
                                        return None  # Return None to skip conversion
                                    # Try numeric conversion
                                    return pd.to_numeric(labels, errors='coerce')
                                except Exception as e:
                                    print(f"Numeric conversion failed {e}")
                                    return None
                            
                            if isinstance(y_train, pd.Series):
                                y_train_numeric = safe_numeric_conversion(y_train)
                                if y_train_numeric is not None and not y_train_numeric.isna().all():
                                    y_train_binned = discretizer.fit_transform(y_train_numeric.values.reshape(-1, 1)).flatten()
                                    y_train = pd.Series([f"Bin_{int(bin_id)}" for bin_id in y_train_binned], index=y_train.index)
                                else:
                                    print("Keep original string labels, no binning")
                            else:
                                y_train_numeric = safe_numeric_conversion(y_train)
                                if y_train_numeric is not None and not pd.isna(y_train_numeric).all():
                                    y_train_binned = discretizer.fit_transform(y_train_numeric.reshape(-1, 1)).flatten()
                                    y_train = np.array([f"Bin_{int(bin_id)}" for bin_id in y_train_binned])
                                else:
                                    print("Keep original string labels, no binning")
                            
                            # Process test set similarly
                            if y_test is not None:
                                if isinstance(y_test, pd.Series):
                                    y_test_numeric = safe_numeric_conversion(y_test)
                                    if y_test_numeric is not None and not y_test_numeric.isna().all():
                                        y_test_binned = discretizer.transform(y_test_numeric.values.reshape(-1, 1)).flatten()
                                        y_test = pd.Series([f"Bin_{int(bin_id)}" for bin_id in y_test_binned], index=y_test.index)
                                    else:
                                        print("Keep original test labels, no binning")
                                else:
                                    y_test_numeric = safe_numeric_conversion(y_test)
                                    if y_test_numeric is not None and not pd.isna(y_test_numeric).all():
                                        y_test_binned = discretizer.transform(y_test_numeric.reshape(-1, 1)).flatten()
                                        y_test = np.array([f"Bin_{int(bin_id)}" for bin_id in y_test_binned])
                                    else:
                                        print("Keep original test labels, no binning")
                            print(f"Binning completed, generated {len(np.unique([label for label in y_train]))} categories")
                        except Exception as e:
                            print(f"Binning failed: {e}, will use original data")
                    # Already handled by label standardization above
                analyzer = QualitativeAnalyzer(  # Create qualitative analyzer
                    method=method  # Set analysis method
                )
                # Train model
                self.progress_dialog.update_progress(40, "Training qualitative model...")
                try:
                    # **SIMPLIFIED: y_train should already be properly formatted from above**
                    print("=== MAIN WINDOW: Using pre-processed labels for qualitative analysis ===")
                    y_train_safe = y_train  # Use the already processed labels
                    print(f"Labels ready: {len(set(y_train_safe))} unique labels")
                    print(f"Sample labels: {list(y_train_safe[:5])}")
                    print(f"Label dtype: {y_train_safe.dtype}")
                      
                    print(f"\n{'='*80}")
                    print("DEBUG: Target reached")
                    print(f"{'='*80}")
                    print(f"y_train_safe shape: {y_train_safe.shape}")
                    print(f"y_train_safe sample values: {y_train_safe[:5]}")
                    print(f"y_train_safe dtype: {y_train_safe.dtype}")
                    print(f"y_train_safe unique values: {len(set(y_train_safe))}")
                    print(f"y_train_safe type check: {type(y_train_safe)}")
                      
                    sample_types = set(type(v).__name__ for v in y_train_safe[:20])
                    if len(sample_types) > 1:
                        print(f"Mixed data types detected: {sample_types}")
                        print("Converting all values to string for consistency")
                        # Convert all to string for consistency
                        y_train_safe = np.array([str(v) for v in y_train_safe])
                        print(f"After string conversion: {y_train_safe.dtype}")
                    else:
                        print(f"Uniform data type: {sample_types}")
                    # Ensure y_train is suitable for classification
                    analyzer.fit(X_train, y_train_safe)
                except ImportError as e:
                    # Handle import errors - if required libraries are not installed
                    error_msg = f"Required library not installed: {str(e)}. Please install it and try again."
                    self.display_error(error_msg)
                    self.progress_dialog.hide()
                    return
                except Exception as e:
                    # Handle other training errors
                    error_msg = f"Error training model: {str(e)}"
                    self.display_error(error_msg)
                    self.progress_dialog.hide()
                    return
                # Update progress bar to avoid getting stuck at 40%
                self.progress_dialog.update_progress(50, "Model training completed...")
                # Cross-validation
                self.progress_dialog.update_progress(60, "Performing cross-validation...")
                # Use the same standardized labels for cross-validation
                print("=== MAIN WINDOW: Cross-validation using pre-processed labels ===")
                print(f"Cross-validation labels: {y_train_safe.dtype}, samples: {y_train_safe[:5]}")
                results = analyzer.cross_validate(X_train, y_train_safe, cv=hyperopt_params['cv_folds'])
                # Make predictions
                if X_test is not None and y_test is not None:
                    self.progress_dialog.update_progress(70, "Making predictions...")
                      
                    if analysis_type.startswith("Qualitative"):  # Qualitative analysis
                        print(f"Qualitative analysis data type check - y_test type: {type(y_test)}, dtype: {y_test.dtype if hasattr(y_test, 'dtype') else 'N/A'}")
                        print(f"Qualitative analysis label unique values: {np.unique(y_test)}")
                          
                        # Test labels already standardized to string type above
                          
                        if hasattr(y_test, 'dtype') and y_test.dtype.kind == 'f':
                              
                              
                            n_bins = min(10, len(np.unique(y_test)))    
                            print(f"Detected continuous value labels, will discretize into{n_bins} categories")
                            try:
                                  
                                import pandas as pd
                                y_test_bins = pd.cut(y_test, bins=n_bins, labels=False)
                                print(f"Discretized label unique values: {np.unique(y_test_bins)}")
                                  
                                y_test_eval = y_test_bins
                            except Exception as e:
                                print(f"Discretization failed: {str(e)}, will fall back to safe conversion")
                                # **CRITICAL FIX: Safe conversion that handles string labels**
                                try:
                                    # Check if labels are string-based before attempting conversion
                                    sample_labels = [str(label) for label in y_test[:5]]
                                    has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                           for label in sample_labels if label.strip())
                                    
                                    if has_string_labels:
                                        print(f"Detected string labels, keeping as classification labels: {sample_labels[:3]}")
                                        y_test_eval = y_test  # Keep original string labels
                                    else:
                                        # **CRITICAL FIX: Safe label handling without forced conversion**
                                        try:
                                            # Check if labels are string-based (like Verde, ClassA, etc.)
                                            sample_labels = [str(label) for label in y_test[:5]]
                                            has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                                   for label in sample_labels if label.strip())
                                            
                                            if has_string_labels:
                                                print(f"🔧 Detected string labels (like Verde): {sample_labels[:3]}. Keeping as classification labels.")
                                                y_test_eval = y_test  # Keep original string labels for classification
                                            else:
                                                # Only convert to int if truly numeric
                                                y_test_eval = y_test.astype(int)
                                        except Exception as e:
                                            print(f"Integer conversion failed: {e}")
                                            # **FINAL FALLBACK: Always use original labels for classification**
                                            print("🔧 Using original labels as classification targets")
                                            y_test_eval = y_test
                                except Exception as e:
                                    print(f"Numeric conversion failed: {e}")
                                    y_test_eval = y_test  # Keep original labels
                        elif hasattr(y_test, 'dtype') and y_test.dtype.kind in 'iu':    
                            print("Integer type labels, no processing needed")
                            y_test_eval = y_test
                        else:    
                            print("Other type labels, keep as is")
                            y_test_eval = y_test
                    else:  # Quantitative analysis - regression problem, should keep continuous values unchanged
                        print(f"Quantitative analysis data type check - y_test type: {type(y_test)}, dtype: {y_test.dtype if hasattr(y_test, 'dtype') else 'N/A'}")
                        # **CRITICAL FIX: Safe numeric conversion for quantitative analysis**
                        if hasattr(y_test, 'dtype') and y_test.dtype.kind not in 'iuf':  # Non-numeric type
                            try:
                                # **CRITICAL FIX: Check if labels are string-based before attempting conversion**
                                sample_labels = [str(label) for label in y_test[:5]]
                                has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                       for label in sample_labels if label.strip())
                                
                                if has_string_labels:
                                    print(f"Detected string labels in quantitative analysis: {sample_labels[:3]}")
                                    print("Skipping numeric conversion for string-based classification labels")
                                    y_test_eval = y_test  # Keep original string labels
                                else:
                                    # **CRITICAL FIX: Use enhanced label processor for safe evaluation**
                                    from app.utils.label_processor import EnhancedLabelProcessor
                                    label_processor = EnhancedLabelProcessor()
                                    
                                    # Detect task type first
                                    test_task_type = label_processor.detect_task_type(y_test)
                                    print(f"🤖 Test evaluation task type: {test_task_type}")
                                    
                                    if test_task_type == 'classification':
                                        print("🔧 Classification evaluation - using string labels")
                                        y_test_eval = y_test  # Keep original string labels
                                    else:
                                        print("🔧 Regression evaluation - attempting numeric conversion")
                                        try:
                                            y_test_eval = pd.to_numeric(y_test, errors='coerce')
                                            # Check if conversion failed
                                            if pd.isna(y_test_eval).any():
                                                failed_labels = y_test[pd.isna(y_test_eval)][:3].tolist()
                                                print(f"Cannot convert test labels to numeric: {failed_labels}. Treating as classification.")
                                                y_test_eval = y_test  # Keep original labels
                                        except (ValueError, TypeError):
                                            print("❌ Numeric conversion failed - treating as classification")
                                            y_test_eval = y_test
                                
                                # Check if conversion produced too many NaN values
                                nan_count = pd.isna(y_test_eval).sum()
                                if nan_count > len(y_test_eval) * 0.5:  # If >50% are NaN
                                    print(f"Warning: {nan_count}/{len(y_test_eval)} labels became NaN after conversion")
                                    print("This suggests the data is better suited for classification analysis")
                                    y_test_eval = y_test  # Keep original labels
                                    
                            except Exception as e:
                                print(f"Numeric conversion failed: {e}")
                                y_test_eval = y_test  # Keep original labels
                        else:
                            y_test_eval = y_test
                      
                    y_pred = analyzer.predict(X_test)
                    from app.services.evaluation_service import EvaluationService
                    eval_service = EvaluationService()
                      
                    metrics_results = eval_service.eval_classification_metrics(
                        y_test_eval, y_pred, ['accuracy', 'precision', 'recall', 'f1']
                    )
                    results.update({
                        'test_accuracy': metrics_results.get('accuracy', 0.0),
                        'test_precision': metrics_results.get('precision', 0.0),
                        'test_recall': metrics_results.get('recall', 0.0),
                        'test_f1': metrics_results.get('f1', 0.0),
                        'task_type': 'classification'
                    })
                  
                if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                    self.controller.modeling_controller.modeling_model.trained_model = analyzer.model
                    self.controller.modeling_controller.modeling_model.set_method(method)
                    self.controller.modeling_controller.modeling_model.set_evaluation_results(results)
                    
                    # **New: Auto-save ONNX model**
                    self.auto_save_onnx_model_from_analysis(analyzer.model, X_train, method, results)
            # Display results
            self.progress_dialog.update_progress(75, "Generating analysis results...")  # Update progress to 75%
            msg = "Analysis Results:\n"  # Initialize result text
            for metric, value in results.items():  # Iterate through results
                if isinstance(value, float):
                    msg += f"{metric}: {value:.4f}\n"  # Add result information
                else:
                    msg += f"{metric}: {value}\n"  # Add result information
              
            self.progress_dialog.update_progress(90, "Updating visualization...")
            # Update visualization components - add code based on specific implementation
            if hasattr(self, 'visualization_widget'):
                if analysis_type.startswith("Quantitative") and X_test is not None and y_test is not None:
                    # Display prediction result scatter plot
                    self.visualization_widget.clear()
                    self.visualization_widget.plot_scatter(y_test, y_pred, 
                                                         title="Actual vs Predicted", 
                                                         xlabel="Actual Values", 
                                                         ylabel="Predicted Values")
                elif X_test is not None and y_test is not None:
                    # Display confusion matrix
                    from sklearn.metrics import confusion_matrix
                    
                    print(f"🔍 Preparing confusion matrix visualization...")
                    print(f"y_test data type: {y_test.dtype if hasattr(y_test, 'dtype') else 'N/A'}")
                    print(f"y_pred data type: {y_pred.dtype if hasattr(y_pred, 'dtype') else 'N/A'}")
                    print(f"y_test unique values: {np.unique(y_test)}")
                    print(f"y_pred unique values: {np.unique(y_pred)}")
                    
                    # **CRITICAL FIX: Fix y_test and y_pred type mismatch issue**
                    print("🔧 Checking and fixing label type mismatch...")
                    
                    # Get original label mapping from data model
                    data_model = self.controller.data_controller.data_model
                    original_label_mapping = None
                    
                    if hasattr(data_model, 'label_mapping') and data_model.label_mapping:
                        original_label_mapping = data_model.label_mapping
                        print(f"📋 Found label mapping in data model: {original_label_mapping}")
                    
                    # **Key fix: Detect type differences between y_test and y_pred**
                    y_test_is_string = hasattr(y_test, 'dtype') and y_test.dtype.kind in ['U', 'S', 'O']
                    y_pred_is_numeric = hasattr(y_pred, 'dtype') and y_pred.dtype.kind in ['i', 'f']
                    
                    print(f"🔍 y_testis string type: {y_test_is_string}")
                    print(f"🔍 y_predis numeric type: {y_pred_is_numeric}")
                    
                    if y_test_is_string and y_pred_is_numeric:
                        print("⚡ Detected type mismatch: y_test is string, y_pred is numeric")
                        
                        # Get real class names (from y_test)
                        unique_test_labels = np.unique(y_test)
                        unique_pred_indices = np.unique(y_pred)
                        
                        print(f"📊 Unique labels in test set: {unique_test_labels}")
                        print(f"📊 Unique indices in prediction results: {unique_pred_indices}")
                        
                        # Create correct mapping: index -> label names
                        # Assume prediction indices correspond to alphabetically sorted labels
                        sorted_labels = sorted(unique_test_labels)
                        
                        # Create index to label mapping
                        index_to_label = {i: label for i, label in enumerate(sorted_labels)}
                        label_to_index = {label: i for i, label in enumerate(sorted_labels)}
                        
                        print(f"🗺️ Created index to label mapping: {index_to_label}")
                        print(f"🗺️ Created label to index mapping: {label_to_index}")
                        
                        # Convert y_test to indices
                        y_test_indices = np.array([label_to_index.get(str(label), 0) for label in y_test])
                        
                        # Ensure y_pred indices are within valid range
                        max_valid_index = len(sorted_labels) - 1
                        y_pred_indices = np.array([min(int(pred), max_valid_index) if pred <= max_valid_index else 0 for pred in y_pred])
                        
                        print(f"✅ Converted y_test index range: {np.min(y_test_indices)} - {np.max(y_test_indices)}")
                        print(f"✅ Converted y_pred index range: {np.min(y_pred_indices)} - {np.max(y_pred_indices)}")
                        
                        # Use real label names as class_names
                        class_names = sorted_labels
                        
                    else:
                        print("📝 Using existing label mapping logic...")
                        
                        # If no mapping found, create clean mapping based on current data
                        if original_label_mapping is None:
                            print("🆕 No existing label mapping found, creating new mapping...")
                            
                            # Get all unique labels, but only use current analysis labels
                            all_unique_values = np.unique(np.concatenate([
                                np.array(y_test).flatten(),
                                np.array(y_pred).flatten()
                            ]))
                            
                            # Filter and clean labels
                            clean_labels = []
                            for val in all_unique_values:
                                str_val = str(val).strip()
                                # Skip obviously incorrect labels
                                if str_val not in ['nan', 'None', '', 'False', 'True'] and not str_val.startswith('category'):
                                    clean_labels.append(str_val)
                            
                            # If no valid labels after cleaning, use original labels
                            if not clean_labels:
                                clean_labels = [str(val) for val in all_unique_values]
                            
                            print(f"🧹 Cleaned labels: {clean_labels}")
                            
                            # Create mapping
                            label_to_index = {str(val): idx for idx, val in enumerate(clean_labels)}
                            index_to_label = {idx: str(val) for idx, val in enumerate(clean_labels)}
                            
                        else:
                            # Extract mapping dictionary
                            index_to_label = original_label_mapping.get('index_to_label', {})
                            label_to_index = original_label_mapping.get('label_to_index', {})
                        
                        # Use original conversion logic
                        def safe_convert_to_index(labels, label_to_index):
                            """Safely convert labels to indices"""
                            indices = []
                            for label in labels:
                                str_label = str(label).strip()
                                if str_label in label_to_index:
                                    indices.append(label_to_index[str_label])
                                else:
                                    indices.append(0)  # default mapping to0
                            return np.array(indices)
                        
                        y_test_indices = safe_convert_to_index(y_test, label_to_index)
                        y_pred_indices = safe_convert_to_index(y_pred, label_to_index)
                        
                        # Create class names list
                        max_index = max(max(y_test_indices), max(y_pred_indices))
                        class_names = []
                        for i in range(max_index + 1):
                            if i in index_to_label:
                                class_names.append(str(index_to_label[i]))
                            else:
                                class_names.append(f"Class_{i}")
                    
                    print(f"📈 Final test label index unique values: {np.unique(y_test_indices)}")
                    print(f"📈 Final prediction label index unique values: {np.unique(y_pred_indices)}")
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(y_test_indices, y_pred_indices)
                    print(f"📊 Confusion matrix shape: {cm.shape}")
                    
                    print(f"🏷️ Final class names ({len(class_names)}): {class_names}")
                    
                    # Verify data consistency
                    if len(class_names) == cm.shape[0] == cm.shape[1]:
                        print("✅ Label mapping and confusion matrix dimensions fully matched")
                    else:
                        print(f"❌ Dimension mismatch: class_names={len(class_names)}, cm_shape={cm.shape}")
                        
                        # Force fixDimension mismatch
                        matrix_size = cm.shape[0]
                        if len(class_names) != matrix_size:
                            if len(class_names) > matrix_size:
                                class_names = class_names[:matrix_size]
                            else:
                                while len(class_names) < matrix_size:
                                    class_names.append(f"Class_{len(class_names)}")
                            print(f"🔧 Fixed class names: {class_names}")
                    
                    # Display confusion matrix
                    print("🎨 Starting confusion matrix plotting...")
                    self.visualization_widget.clear()
                    self.visualization_widget.plot_confusion_matrix(cm, 
                                                                 class_names=class_names,
                                                                 title="Confusion Matrix")
                    print("✅ Confusion matrix display completed")
              
            self.progress_dialog.update_progress(95, "Finalizing results...")
              
            if hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                if hasattr(self.controller.modeling_controller.modeling_model, 'evaluation_results'):
                      
                    final_results = self.controller.modeling_controller.modeling_model.evaluation_results
                    if final_results:
                        results = final_results
              
            self.result_table.setRowCount(0)
              
            if self.result_table.columnCount() < 5:
                  
                self.result_table.setColumnCount(5)
                self.result_table.setHorizontalHeaderLabels(["Method", "R²/Accuracy", "RMSE/F1", "MAE/Precision", "Recall"])
              
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)
              
            method_item = QTableWidgetItem(method)
            self.result_table.setItem(row_position, 0, method_item)
              
            is_quantitative = analysis_type.startswith("Quantitative")
            task_type = results.get('task_type', 'unknown')
            print(f"Displaying results - Analysis type: {analysis_type}, Task type: {task_type}")
              
            if is_quantitative or task_type == 'regression':
                  
                r2_value = results.get('r2_mean', results.get('test_r2', 0.0))
                r2_item = QTableWidgetItem(f"{r2_value:.4f}")
                self.result_table.setItem(row_position, 1, r2_item)
                rmse_value = results.get('rmse_mean', results.get('test_rmse', 0.0)) 
                rmse_item = QTableWidgetItem(f"{rmse_value:.4f}")
                self.result_table.setItem(row_position, 2, rmse_item)
                  
                mae_value = results.get('mae_mean', results.get('test_mae', 0.0))
                mae_item = QTableWidgetItem(f"{mae_value:.4f}")
                self.result_table.setItem(row_position, 3, mae_item)
                  
                self.result_table.setItem(row_position, 4, QTableWidgetItem("-"))
            else:
                  
                acc_value = results.get('accuracy', results.get('test_accuracy', 0.0))
                acc_item = QTableWidgetItem(f"{acc_value:.4f}")
                self.result_table.setItem(row_position, 1, acc_item)
                f1_value = results.get('f1_score', results.get('test_f1', 0.0))
                f1_item = QTableWidgetItem(f"{f1_value:.4f}")
                self.result_table.setItem(row_position, 2, f1_item)
                precision_value = results.get('precision', results.get('test_precision', 0.0))
                precision_item = QTableWidgetItem(f"{precision_value:.4f}")
                self.result_table.setItem(row_position, 3, precision_item)
                recall_value = results.get('recall', results.get('test_recall', 0.0)) 
                recall_item = QTableWidgetItem(f"{recall_value:.4f}")
                self.result_table.setItem(row_position, 4, recall_item)
              
            self.result_table.resizeColumnsToContents()
              
            self.progress_dialog.update_progress(100, "Analysis complete")    
            self.progress_dialog.close()    
            QMessageBox.information(self, "Analysis Results", msg)
        except Exception as e:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
    def generate_report(self):
        """Generate analysis report"""
        if self.current_spectra is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        try:
            # Show progress dialog
            self.progress_dialog = ProgressDialog("Generating Report", self)
            self.progress_dialog.show()
            # Prepare report data
            self.progress_dialog.update_progress(25, "Preparing report data...")
            spectra_data = {
                'shape': self.current_spectra.shape,
                'wavelength_range': f"{self.wavelengths[0]:.2f} - {self.wavelengths[-1]:.2f}",
                'n_samples': self.current_spectra.shape[0],
                'n_features': self.current_spectra.shape[1]
            }
            # Get actual preprocessing methods applied
            preprocessing_results = {}
            if hasattr(self, 'controller') and hasattr(self.controller, 'preprocessing_controller'):
                  
                if hasattr(self.controller.preprocessing_controller, 'preprocessing_model'):
                      
                    print(f"Looking for preprocessing methods: found preprocessing_model")
                    if hasattr(self.controller.preprocessing_controller.preprocessing_model, 'applied_methods'):
                          
                        print(f"Found applied_methods, containing {len(self.controller.preprocessing_controller.preprocessing_model.applied_methods)} methods")
                        for method, params in self.controller.preprocessing_controller.preprocessing_model.applied_methods.items():
                            preprocessing_results[method] = params
                    else:
                        print("preprocessing_model exists but has no applied_methods attribute")
                        preprocessing_results = {
                            'No preprocessing': 'No preprocessing methods applied yet'
                        }
                else:
                    print("Cannot find preprocessing_model attribute")
                    preprocessing_results = {
                        'No preprocessing': 'No preprocessing methods applied yet'
                    }
            else:
                print("Cannot find preprocessing_controller")
                preprocessing_results = {
                    'No preprocessing': 'No preprocessing methods applied yet'
                }
            # Get actual feature selection methods applied
            feature_extraction_results = {}
            if hasattr(self, 'controller') and hasattr(self.controller, 'feature_selection_controller') and hasattr(self.controller.feature_selection_controller, 'feature_selection_model'):
                if hasattr(self.controller.feature_selection_controller.feature_selection_model, 'selected_method'):
                    method = self.controller.feature_selection_controller.feature_selection_model.selected_method
                    num_features = len(self.controller.feature_selection_controller.feature_selection_model.selected_features) if hasattr(self.controller.feature_selection_controller.feature_selection_model, 'selected_features') else 0
                    feature_extraction_results[method] = f"Selected {num_features} features"
                else:
                    feature_extraction_results = {
                        'No feature selection': 'No feature selection methods applied yet'
                    }
            else:
                feature_extraction_results = {
                    'No feature selection': 'No feature selection methods applied yet'
                }
            # Get actual analysis/modeling results
            analysis_results = {}
            if hasattr(self, 'controller') and hasattr(self.controller, 'modeling_controller') and hasattr(self.controller.modeling_controller, 'modeling_model'):
                if hasattr(self.controller.modeling_controller.modeling_model, 'evaluation_results'):
                    analysis_results = self.controller.modeling_controller.modeling_model.evaluation_results
                elif hasattr(self.controller.modeling_controller.modeling_model, 'trained_model'):
                    analysis_results = {
                        'Model': str(type(self.controller.modeling_controller.modeling_model.trained_model).__name__),
                        'Training status': 'Trained but not evaluated'
                    }
                else:
                    analysis_results = {
                        'No analysis': 'No modeling or analysis has been performed yet'
                    }
            else:
                analysis_results = {
                    'No analysis': 'No modeling or analysis has been performed yet'
                }
            # Generate report
            self.progress_dialog.update_progress(75, "Generating PDF report...")
            report_generator = SpectrumReportGenerator()
            report_path = report_generator.generate_report(
                spectra_data,
                preprocessing_results,
                feature_extraction_results,
                analysis_results,
                title="Spectral Analysis Report"
            )
            # Complete
            self.progress_dialog.update_progress(100, "Report generation complete")
            self.progress_dialog.close()
            self.statusBar().showMessage(f"Report generated: {report_path}")
            QMessageBox.information(self, "Success", "Analysis report generated successfully")
        except Exception as e:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Report generation failed: {str(e)}")

    def preview_preprocessing(self):
        """Preview the preprocessing effects on spectral data"""
        global pd
        if self.current_spectra is None:    
            QMessageBox.warning(self, "Warning", "Please load data first")    
            return
        try:
              
            preview_mode = self.preview_mode.currentText()    
              
            # **CRITICAL FIX: Fix parameter name mapping to match SpectrumPreprocessor expected parameter names**
            params = {
                'baseline_correction': self.baseline_check.isChecked(),
                'poly_order': self.poly_order.value(),  # Fix parameter name: polynomial_order -> poly_order
                'smoothing': self.smooth_check.isChecked(),
                'window_length': self.window_size.value(),
                'normalization': self.norm_method.currentText().lower() if self.norm_check.isChecked() else 'none',  # Fix parameter format
                'outlier_detection': self.outlier_check.isChecked(),
                'threshold': self.outlier_threshold.value(),
                # Temporarily comment out parameters not supported by SpectrumPreprocessor
                # 'scatter_correction': self.scatter_check.isChecked(),
                # 'scatter_method': self.scatter_method.currentText(),
                # 'standardization': self.standardize_check.isChecked(),
                # 'standardization_method': self.standardize_method.currentText(),
                # 'derivative': self.derivative_check.isChecked(),
                # 'derivative_order': self.derivative_order.value(),
                # 'denoising': self.denoise_check.isChecked(),
                # 'denoising_strength': self.denoise_strength.value(),
                # 'alignment': self.alignment_check.isChecked(),
                # 'alignment_method': self.alignment_method.currentText(),
                # 'alignment_reference': self.reference_method.currentText()
            }
              
            input_data = self.spectra_data
            if not hasattr(self, 'original_spectra') or self.original_spectra is None:
                self.original_spectra = self.spectra_data.copy()
            else:
                input_data = self.original_spectra
              
            data = pd.DataFrame(input_data)    
              
            preprocessor = SpectrumPreprocessor()    
            processed_data = preprocessor.apply(data, params)    
              
            if preview_mode == "Original":    
                self.visualization_widget.plot_spectra(self.wavelengths, self.original_spectra, labels=self.sample_labels)    
            elif preview_mode == "Processed":    
                self.visualization_widget.plot_spectra(self.wavelengths, processed_data.values, labels=self.sample_labels)    
            else:  # Comparison mode
                self.visualization_widget.plot_spectra_comparison(
                    self.wavelengths, 
                    self.original_spectra, 
                    processed_data.values,
                    title="Original vs Processed Spectra"
                )
        except Exception as e:    
            QMessageBox.critical(self, "Error", f"Preview failed: {str(e)}")    
    def reset_preprocessing_params(self):
        """Reset preprocessing parameters to default values"""
        
        # Reset despiking parameters
        self.despiking_check.setChecked(False)
        self.despiking_method.setCurrentIndex(0)  # MAD
        self.despiking_window.setValue(7)  # More conservative default
        self.despiking_threshold.setValue(3.0)  # More conservative threshold
          
        self.baseline_check.setChecked(False)
        self.baseline_method.setCurrentIndex(0)  # Polynomial
        self.scatter_check.setChecked(False)    
        self.smooth_check.setChecked(False)    
        self.derivative_check.setChecked(False)    
        self.standardize_check.setChecked(False)    
        self.norm_check.setChecked(False)    
        self.outlier_check.setChecked(False)    
        self.denoise_check.setChecked(False)    
        self.alignment_check.setChecked(False)
        
        # Reset Raman specific parameters
        self.wavelength_calib_check.setChecked(False)
        self.intensity_calib_check.setChecked(False)    
          
        self.poly_order.setValue(2)  # Better default for most spectra    
        self.scatter_method.setCurrentIndex(0)    
        self.smooth_method.setCurrentIndex(0)    
        self.window_size.setValue(11)  # Better odd number for smoothing    
        self.derivative_order.setValue(1)    
        self.standardize_method.setCurrentIndex(0)    
        self.norm_method.setCurrentIndex(0)    
        self.outlier_threshold.setValue(3.0)    
        self.denoise_strength.setValue(0.5)    
        self.alignment_method.setCurrentIndex(0)    
        self.reference_method.setCurrentIndex(0)    
          
        self.preview_mode.setCurrentIndex(0)    
          
        if hasattr(self, 'original_spectra') and self.original_spectra is not None:    
            self.current_spectra = self.original_spectra.copy()    
            self.visualization_widget.plot_spectra(self.wavelengths, self.original_spectra, labels=self.sample_labels)    
        elif self.spectra_data is not None:    
            self.visualization_widget.plot_spectra(self.wavelengths, self.spectra_data, labels=self.sample_labels)    

    def show_preprocessing_help(self):
        """Show comprehensive preprocessing algorithms help"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QTextEdit, QPushButton
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Preprocessing Algorithms Guide")
        help_dialog.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        
        # Create tab widget for different algorithm categories
        tabs = QTabWidget()
        
        # Spike Removal Tab
        spike_tab = QWidget()
        spike_layout = QVBoxLayout()
        spike_text = QTextEdit()
        spike_text.setReadOnly(True)
        spike_content = """
<h1><b>Spike Removal (Despiking)</b></h1>

<p>Spike removal is primarily used to eliminate cosmic ray spikes and instrumental artifacts from Raman spectra. These spikes typically appear as abnormally high intensity values at single or few data points, which can severely affect subsequent spectral analysis.</p>

<h2><b>MAD (Median Absolute Deviation)</b></h2>
<p>The MAD method identifies spikes by calculating the deviation of data points from the median within a local window. For each data point, the algorithm takes a window around it, calculates the median and median absolute deviation within that window, and if the current point's deviation exceeds the threshold, it is considered a spike and replaced with the median value. This method is particularly effective for detecting sharp cosmic ray spikes in Raman spectroscopy because it uses robust statistics that are less affected by the spikes themselves.</p>

<p>Window size is typically set between 7-15 points, with smaller windows (7-9 points) being suitable for detecting sharp cosmic ray spikes, while larger windows provide more stability for noisy spectra. The threshold parameter controls detection sensitivity, with values of 3-4 being conservative and suitable for most situations. Setting the threshold too low may remove normal data points, while too high may miss genuine spikes. After processing, check the spike statistics in reports to evaluate effectiveness.</p>

<h2><b>Local Z-score</b></h2>
<p>The Local Z-score method detects anomalous points using local mean and standard deviation. The algorithm calculates local statistics around each data point, normalizes the difference between the current point and local mean by the local standard deviation to obtain a standardized score, and points exceeding the threshold are considered spikes. This method is more sensitive to broader anomalous features and suitable for handling cases beyond simple single-point spikes, particularly when baseline fluctuations are significant.</p>

<p>Recommended window size is 9-11 points, with thresholds typically set at 4-5, which is higher than MAD due to different Z-score distribution characteristics. In practice, start with the MAD method using conservative parameters (window=7, threshold=3) for testing and observe results. Increase threshold if too many normal data points are removed, or increase window size if spectra are noisy.</p>
        """
        spike_text.setHtml(spike_content)
        spike_layout.addWidget(spike_text)
        spike_tab.setLayout(spike_layout)
        tabs.addTab(spike_tab, "Spike Removal")
        
        # Baseline Correction Tab
        baseline_tab = QWidget()
        baseline_layout = QVBoxLayout()
        baseline_text = QTextEdit()
        baseline_text.setReadOnly(True)
        baseline_content = """
<h1><b>Baseline Correction</b></h1>

<p>Baseline correction removes baseline drift and background fluorescence interference from spectra. Baseline issues are common in various spectroscopic techniques, manifesting as slow changes in overall intensity or background signal overlay, affecting accurate peak intensity determination and quantitative spectral analysis.</p>

<h2><b>Polynomial</b></h2>
<p>Polynomial baseline correction estimates the baseline by fitting a polynomial function and then subtracts it from the original spectrum. This is the most straightforward method, suitable for smooth, gradually varying baselines. The polynomial order is the key parameter, with orders 1-5 being selectable.</p>

<p>Low-order polynomials (1-2) are suitable for simple linear or quadratic baseline drift, while high-order polynomials (3-5) can handle more complex baseline shapes but may overfit and affect genuine spectral features. Orders 2-3 are appropriate choices for most cases. Too low order may not completely remove baseline, while too high may weaken or distort genuine peak signals.</p>

<h2><b>ALS (Asymmetric Least Squares)</b></h2>
<p>The ALS method uses an iterative algorithm that assigns greater weight to points below the baseline and lower weight to peak regions, enabling better distinction between baseline and peak signals. This method is particularly suitable for spectra with obvious peak signals above the baseline, as it can automatically identify and protect peak regions from baseline fitting interference. Compared to polynomial methods, ALS better maintains peak shapes and intensities, especially for peak-dense spectra. The algorithm automatically adjusts weight distribution without requiring manual setting of complex parameters, but computation time is relatively longer.</p>

<h2><b>airPLS (Adaptive Iteratively Reweighted Penalized Least Squares)</b></h2>
<p>airPLS is an improved version of the ALS method that employs adaptive weight adjustment strategies. The algorithm can automatically identify baseline and peak regions, dynamically adjusting the weight of each data point, providing stronger adaptability to complex and variable baselines. This method performs excellently when processing multi-component baselines, irregular backgrounds, or spectra with significant baseline shape variations. The advantage of airPLS lies in its strong robustness, providing good results for different types of spectra, though with the highest computational complexity.</p>

<p>When selecting baseline correction methods, start with 2nd-order polynomial. If the baseline is complex or has obvious peak signals, use ALS. For particularly difficult baselines, try airPLS. After processing, always visually inspect results to ensure peak shapes are not destroyed and baselines are effectively removed.</p>
        """
        baseline_text.setHtml(baseline_content)
        baseline_layout.addWidget(baseline_text)
        baseline_tab.setLayout(baseline_layout)
        tabs.addTab(baseline_tab, "Baseline Correction")
        
        # Scatter Correction Tab
        scatter_tab = QWidget()
        scatter_layout = QVBoxLayout()
        scatter_text = QTextEdit()
        scatter_text.setReadOnly(True)
        scatter_content = """
<h1><b>Scatter Correction</b></h1>

<p>Scatter correction is primarily used in near-infrared and mid-infrared spectroscopy to eliminate multiplicative scattering effects caused by physical properties of samples such as particle size, density, and surface roughness. These effects cause samples with identical chemical composition to produce different spectral intensities, affecting quantitative analysis accuracy.</p>

<h2><b>SNV (Standard Normal Variate)</b></h2>
<p>SNV performs centering and standardization on each spectrum individually, using the formula (X-mean)/standard deviation. This method assumes that scattering effects have the same influence across the entire spectral range, correcting scattering by eliminating intensity differences between spectra. SNV processing is simple and fast, with good results for most near-infrared transmission spectra, particularly suitable when scattering differences between samples are small. However, this method may amplify noise and should be used cautiously for spectra with low signal-to-noise ratios.</p>

<h2><b>MSC (Multiplicative Scatter Correction)</b></h2>
<p>MSC uses the average spectrum of all samples as a reference, correcting scattering by aligning each spectrum with the reference spectrum through linear regression. This method maintains relative intensity relationships between spectra and is suitable for reflectance spectra processing. MSC assumes scattering is systematic and works well for sample sets with regular scattering variation patterns. Compared to SNV, MSC better preserves original spectral shape features, but requires representative reference spectra and is more sensitive to abnormal samples.</p>

<h2><b>EMSC (Extended Multiplicative Scatter Correction)</b></h2>
<p>EMSC adds polynomial baseline correction to MSC, enabling simultaneous handling of multiplicative scattering and additive baseline effects. This method is particularly suitable for complex spectra with both scattering problems and baseline drift. EMSC simultaneously estimates scattering coefficients and baseline polynomial coefficients through multivariate linear regression, providing more comprehensive spectral correction. Processing effects are usually superior to using MSC or baseline correction alone, but computational complexity is higher and parameter settings require more experience.</p>

<h2><b>RNV (Robust Normal Variate)</b></h2>
<p>RNV is an improved version of SNV that uses median and median absolute deviation instead of mean and standard deviation for standardization, providing stronger resistance to outliers and noise. When outliers or significant noise exist in the data, RNV is more stable and reliable than SNV. This method maintains the simplicity of SNV while improving robustness, making it suitable for processing spectral data of varying quality.</p>

<h2><b>OSC (Orthogonal Signal Correction)</b></h2>
<p>OSC is specifically designed for supervised learning scenarios, using principal component analysis to identify and remove variation components orthogonal (unrelated) to target variables. This method requires known target variable information and can selectively remove scattering and other interferences unrelated to analysis objectives. OSC is valuable in chemometric modeling but is only applicable when there are clear analysis targets and cannot be used for exploratory data analysis.</p>

<p>When selecting scatter correction methods, SNV is preferred for near-infrared transmission spectra, MSC is better for reflectance spectra, EMSC is used when both scattering and baseline issues exist, RNV is considered for poor data quality, and OSC is only used for supervised learning requirements.</p>
        """
        scatter_text.setHtml(scatter_content)
        scatter_layout.addWidget(scatter_text)
        scatter_tab.setLayout(scatter_layout)
        tabs.addTab(scatter_tab, "Scatter Correction")
        
        # Smoothing Tab
        smooth_tab = QWidget()
        smooth_layout = QVBoxLayout()
        smooth_text = QTextEdit()
        smooth_text.setReadOnly(True)
        smooth_content = """
<h1><b>Smoothing</b></h1>

<p>The purpose of smoothing is to reduce spectral noise while preserving genuine spectral features as much as possible. Noise typically manifests as high-frequency random fluctuations, while genuine spectral signals are relatively smooth. Appropriate smoothing can improve signal-to-noise ratio, but excessive smoothing leads to peak broadening, intensity reduction, or even feature loss.</p>

<h2><b>Savitzky-Golay Filter</b></h2>
<p>The Savitzky-Golay method fits polynomials within local windows and replaces the center point with the fitted value, enabling both noise smoothing and good preservation of peak shapes. Window size must be odd, typically ranging from 5-21 points. Smaller windows (9-11 points) preserve more details but have limited smoothing effects, while larger windows provide stronger smoothing but may be excessive. Polynomial order controls fitting complexity, with orders 2-3 suitable for most spectra. Too low order may result in insufficient fitting, while too high may introduce false peaks. This method's advantage is its ability to maintain peak positions and relative intensities well, making it the preferred method for spectral smoothing.</p>

<h2><b>Moving Average</b></h2>
<p>Moving average is the simplest smoothing method, directly averaging data points within a window. This method has strong suppression effects on high-frequency noise and fast computation speed, but significantly broadens peak shapes and reduces peak heights. Moving average is suitable for situations with very high noise and low requirements for peak shape. Larger windows provide stronger smoothing effects but more severe peak broadening. It is generally used only in special situations with extremely low signal-to-noise ratios or when strong smoothing is needed.</p>

<h2><b>Gaussian Smoothing</b></h2>
<p>Gaussian smoothing uses Gaussian functions as weights for weighted averaging of data, with maximum weight at the center point and decreasing weights with distance. This method produces natural smoothing effects and better peak shape preservation than moving average. The standard deviation parameter controls smoothing strength, with larger values providing stronger smoothing effects. Gaussian smoothing effectively reduces noise while maintaining overall spectral shape, suitable for applications with certain requirements for peak shape.</p>

<h2><b>Median Filter</b></h2>
<p>Median filtering replaces the center point with the median of data within the window. This method is particularly effective for impulse noise and spikes, capable of smoothing while maintaining edges and sharp features. Median filtering does not blur peak boundaries like other methods, making it useful for situations requiring both denoising and peak sharpness preservation. This method is often combined with other smoothing methods, first using median filtering to remove spikes, then using other methods for further smoothing.</p>

<p>In practical applications, Savitzky-Golay filtering is recommended as the first choice, starting with small windows (9-11 points) and low orders (2-3). For spectra with particularly high noise, try Gaussian smoothing. For situations requiring spike removal, consider median filtering. After smoothing, always check whether peaks are excessively broadened and whether signals are distorted.</p>
        """
        smooth_text.setHtml(smooth_content)
        smooth_layout.addWidget(smooth_text)
        smooth_tab.setLayout(smooth_layout)
        tabs.addTab(smooth_tab, "Smoothing")
        
        # Advanced Methods Tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_text = QTextEdit()
        advanced_text.setReadOnly(True)
        advanced_content = """
<h1><b>Advanced Preprocessing Methods</b></h1>

<h2><b>Normalization</b></h2>
<p>Normalization scales spectral data to specific ranges, eliminating intensity differences between different samples. Min-Max normalization scales data to the [0,1] interval, suitable for maintaining relative data distribution. Unit vector normalization standardizes each spectrum's vector length to 1, emphasizing spectral shape rather than intensity. Peak normalization scales based on the highest peak intensity, suitable for emphasizing relative peak intensity relationships. Method selection depends on analysis objectives: use Min-Max for absolute intensity focus, unit vector for spectral shape focus, and peak normalization for relative peak intensity focus.</p>

<h2><b>Standardization</b></h2>
<p>Standardization converts data to a distribution with mean 0 and standard deviation 1. Z-Score standardization uses arithmetic mean and standard deviation, suitable for data distributions close to normal. Robust standardization uses median and median absolute deviation, providing stronger resistance to outliers. Standardization is mainly used for data preprocessing before multivariate analysis, ensuring different variables have similar scales and preventing certain variables from dominating analysis results due to large numerical ranges.</p>

<h2><b>Derivative</b></h2>
<p>Derivative transformation enhances peak resolution and eliminates baseline interference by calculating first or second derivatives of spectra. First derivative converts original peaks to positive-negative peak pairs, effectively eliminating constant baselines and linear drift while improving resolution of overlapping peaks. Second derivative further enhances resolution, converting original peaks to negative peaks, particularly effective for severely overlapping peaks. Derivative transformation amplifies noise, so smoothing processing is usually required first. This method is valuable in qualitative analysis but changes peak shapes and intensity relationships.</p>

<h2><b>Outlier Detection</b></h2>
<p>Outlier detection is used to identify and process abnormal spectra in datasets. Statistical methods judge outliers based on distance from mean, simple and intuitive but with requirements for data distribution. Isolation forest uses random forest concepts, identifying easily isolated abnormal points by constructing decision trees. Local outlier factor is based on local density concepts, comparing each point's density with its neighbors to identify outliers. Outliers may be measurement errors, sample contamination, or truly special samples, requiring manual judgment after detection on whether to remove them.</p>

<h2><b>Denoising</b></h2>
<p>Denoising methods are more complex than simple smoothing and can better distinguish between signals and noise. Wavelet denoising separates different frequency components through multi-resolution analysis, maintaining signal details while denoising. Total variation denoising is based on signal gradient characteristics, maintaining edges while smoothing regions. Non-local means utilizes repetitive patterns in images for denoising, effective for structured noise. These methods are computationally complex but superior to traditional smoothing, suitable for applications with high signal quality requirements.</p>

<h2><b>Peak Alignment</b></h2>
<p>Peak alignment corrects peak shifts between different spectra, which may be caused by temperature changes, instrument drift, and other factors. Cross-correlation alignment finds optimal alignment positions by maximizing correlations between spectra, suitable for overall shifts. Dynamic time warping allows non-linear alignment, handling situations with different degrees of shift in different regions. The Icoshift method divides spectra into multiple intervals for separate alignment, balancing efficiency and flexibility. Peak alignment is important for multivariate analysis but may change relative position relationships of peaks.</p>
        """
        advanced_text.setHtml(advanced_content)
        advanced_layout.addWidget(advanced_text)
        advanced_tab.setLayout(advanced_layout)
        tabs.addTab(advanced_tab, "Advanced Methods")
        
        # Best Practices Tab (shortened for space)
        practices_tab = QWidget()
        practices_layout = QVBoxLayout()
        practices_text = QTextEdit()
        practices_text.setReadOnly(True)
        practices_content = """
<h1><b>Best Practices & Workflow</b></h1>

<h2><b>Preprocessing Order Recommendations</b></h2>
<p>The order of spectral preprocessing is important, as incorrect order may lead to poor processing effects or even introduce new problems. The recommended processing order is: first perform despiking (Raman spectra only), as spikes affect all subsequent processing steps; then perform baseline correction to remove baseline drift and background interference; next perform scatter correction (mainly for near-infrared and mid-infrared) to eliminate physical scattering effects; finally perform smoothing, as smoothing affects other correction methods and should be placed last.</p>

<h2><b>Processing Strategies for Different Spectral Techniques</b></h2>
<p><b>Raman Spectra</b> typically require despiking, recommended using MAD method to remove cosmic ray spikes, then 2-3 order polynomial for baseline correction. Raman spectra generally do not require scatter correction. Smoothing should be cautious, recommended using small window (7-9 points) Savitzky-Golay filtering to avoid excessive smoothing causing peak broadening.</p>

<p><b>Near-Infrared Spectra</b> do not require despiking. Scatter correction is a key step, with SNV preferred for transmission spectra and MSC for reflectance spectra. Baseline correction is selected as needed, typically 1-2 order polynomials suffice. Smoothing can be moderate, with 9-15 point window Savitzky-Golay filtering being suitable.</p>

<p><b>Mid-Infrared/FTIR Spectra</b> also do not require despiking. Baseline correction is important, with polynomial or ALS methods available. For complex samples, consider EMSC method to simultaneously handle scattering and baseline issues. Smoothing should be moderate, recommended 9-11 point window Savitzky-Golay filtering.</p>

<h2><b>Common Errors and Considerations</b></h2>
<p>Over-processing is the most common problem, with many users tending to use all available preprocessing methods, but appropriate processing is often minimal processing. Incorrect processing order is also common, such as smoothing before baseline correction, which affects baseline detection accuracy. Excessive smoothing leading to spectral resolution loss is another common issue, especially when using overly large smoothing windows. Ignoring visual inspection is dangerous; automated processing results must be checked for reasonableness. Finally, different samples may require different processing parameters and cannot be generalized.</p>

<h2><b>Quality Control Recommendations</b></h2>
<p>After each processing step, use preview functions to check effects and compare spectral changes before and after processing. Pay special attention to whether peak shapes, positions, and intensities undergo unreasonable changes. Processing parameters should be recorded in reports to ensure reproducibility. For important analysis projects, validate processing effects using known standard samples. Establish standard operating procedures, use consistent processing parameters for similar samples, but maintain flexibility to adjust according to specific situations.</p>
        """
        practices_text.setHtml(practices_content)
        practices_layout.addWidget(practices_text)
        practices_tab.setLayout(practices_layout)
        tabs.addTab(practices_tab, "Best Practices")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.close)
        layout.addWidget(close_btn)
        
        help_dialog.setLayout(layout)
        help_dialog.exec_()

    def display_error(self, message, title="Error"):
        """Display an error message to the user"""
        QMessageBox.critical(self, title, message)

    def apply_feature_selection(self):
        """Apply feature selection with intelligent recommendations"""
        try:
            # Get current parameters
            method = self.feature_method.currentText()
            param = self.feature_param_spin.value()
            # **CRITICAL: Intelligent feature count recommendation**
            if hasattr(self.controller, 'data_controller') and hasattr(self.controller.data_controller, 'data_model'):
                data_model = self.controller.data_controller.data_model
                if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                    y_train = data_model.y_train
                    n_classes = len(np.unique(y_train))
                    # Calculate recommended features
                    min_features_needed = max(n_classes * 3, 30)
                    if param < min_features_needed:
                        # Show intelligent recommendation dialog
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Feature Selection Recommendation")
                        msg.setText(f"Current setting: {param} features\n"
                                  f"Detected classes: {n_classes}\n"
                                  f"Recommended: {min_features_needed}+ features\n\n"
                                  f"Low feature count may result in poor accuracy.\n"
                                  f"Would you like to use the recommended value?")
                        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                        msg.setDefaultButton(QMessageBox.Yes)
                        result = msg.exec_()
                        if result == QMessageBox.Yes:
                            # Use recommended value
                            self.feature_param_spin.setValue(min_features_needed)
                            param = min_features_needed
                            print(f"Updated feature count to recommended value: {param}")
                        elif result == QMessageBox.Cancel:
                            return
                        # If No, continue with user setting
            # Continue with existing feature selection logic
            # Get method mapping and parameters
            method_mapping = {
                "PCA": "Principal Component Analysis (PCA)",
                "PLSR": "PLSR",  
                "Peak Detection": "Peak Detection",
                "Wavelet": "Wavelet"
            }
            service_method = method_mapping.get(method, method)
            params = {'k': param}
            # Get training data
            data_model = self.controller.data_controller.data_model
            
            # **CRITICAL FIX: Use preprocessed data if available for feature selection**
            print("🔍 Checking for preprocessed data in feature selection...")
            
            if (hasattr(data_model, 'preprocessed_data') and data_model.preprocessed_data is not None and
                hasattr(data_model, 'X_train') and data_model.X_train is not None):
                
                print(f"✅ Found preprocessed data: shape={data_model.preprocessed_data.shape}")
                print(f"   Original X_train shape: {data_model.X_train.shape}")
                print(f"   Original X_test shape: {data_model.X_test.shape if data_model.X_test is not None else 'None'}")
                
                # Calculate expected dimensions
                X_train_rows = data_model.X_train.shape[0]
                X_test_rows = data_model.X_test.shape[0] if data_model.X_test is not None else 0
                expected_total_rows = X_train_rows + X_test_rows
                
                # Check if preprocessed data matches expected dimensions
                if (data_model.preprocessed_data.shape[0] == expected_total_rows and
                    data_model.preprocessed_data.shape[1] == data_model.X_train.shape[1]):
                    
                    print("🔧 Using PREPROCESSED data for feature selection")
                    # Split preprocessed data back into train/test
                    X_train = data_model.preprocessed_data[:X_train_rows]
                    if X_test_rows > 0:
                        X_test = data_model.preprocessed_data[X_train_rows:X_train_rows + X_test_rows]
                    else:
                        X_test = data_model.X_test
                        
                    print(f"   ✅ Preprocessed X_train shape: {X_train.shape}")
                    print(f"   ✅ Preprocessed X_test shape: {X_test.shape if X_test is not None else 'None'}")
                else:
                    print("⚠️  Preprocessed data dimensions don't match, using original data")
                    X_train = data_model.X_train
                    X_test = data_model.X_test
            else:
                print("⚠️  No preprocessed data found, using original data")
                print("   💡 Suggestion: Apply preprocessing before feature selection for better results")
                X_train = data_model.X_train
                X_test = data_model.X_test
                
            y_train = data_model.y_train
            # Apply feature selection using the controller's apply_method
            result = self.controller.feature_selection_controller.apply_method(
                X_train, y_train, X_test, service_method, params
            )
            # Update data model with results
            data_model.X_train_selected = result['X_train_selected']
            if X_test is not None:
                data_model.X_test_selected = result['X_test_selected']
            QMessageBox.information(self, "Success", 
                f"Feature selection completed. Selected {len(result['selected_features'])} features.")
            # Auto-popup feature importance visualization window
            if 'feature_importance' in result and result['feature_importance'] is not None:
                from app.utils.visualization_window import VisualizationWindow
                # **CRITICAL FIX: Keep window reference to prevent garbage collection**
                if not hasattr(self, 'feature_vis_windows'):
                    self.feature_vis_windows = []
                
                vis_window = VisualizationWindow(title="Feature Selection Result")
                vis_window.plot_feature_importance(
                    result['feature_importance'],
                    result['selected_features']
                )
                vis_window.show()
                
                # Keep window reference
                self.feature_vis_windows.append(vis_window)
                
                # Clean old window references (keep latest 5)
                if len(self.feature_vis_windows) > 5:
                    old_window = self.feature_vis_windows.pop(0)
                    try:
                        old_window.close()
                    except:
                        pass
        except Exception as e:
            self.display_error(f"Feature selection failed: {str(e)}")

    def partition_data_with_params(self):
        """Apply data partitioning with parameters from the data partitioning view"""
          
        if not self.controller.check_data_ready():
            self.display_message("English text", "English text")
            return
        try:
              
            method = self.data_partitioning_view.get_selected_method()
            params = self.data_partitioning_view.get_parameters()
            # **CRITICAL FIX: Force classification mode for qualitative analysis**
            current_analysis_type = self.analysis_type.currentText() if hasattr(self, 'analysis_type') else "Qualitative"
            if current_analysis_type == "Qualitative":
                params['force_classification'] = True
                print(f"Forcing classification mode for qualitative analysis")
            print(f"Data partitioning - method: {method}")
            print(f"Data partitioning - parameters: {params}")
              
            self.controller.data_controller.apply_data_partitioning(method, params)
        except Exception as e:
            self.display_error(f"English text")
            import traceback
            traceback.print_exc()

    def apply_data_partitioning(self):
        """Apply data partitioning - keep backward compatibility"""
          
        self.partition_data_with_params()

    def feature_selection(self):
        # This method is now empty as the feature selection logic has been moved to the controller
        pass

    def on_analysis_type_change(self, analysis_type):
        """Handle analysis type change event"""
        print(f"Analysis type changed to: {analysis_type}")
        # Update interface prompts
        if analysis_type.startswith("Quantitative"):
            self.analysis_desc.setText("Quantitative Analysis: Predict continuous values (Regression analysis)")
            # Check if the data is suitable for quantitative analysis
            if hasattr(self, 'controller') and hasattr(self.controller, 'data_controller'):
                data_model = self.controller.data_controller.data_model
                if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                    y_train = data_model.y_train
                    # Check if it's a classification label (classification features usually have fewer unique values)
                    if hasattr(y_train, 'nunique'):
                        unique_count = y_train.nunique()
                        if unique_count <= 10:  # If unique values are less than 10
                            self.display_message("Warning: The current target variable has few unique values ({}), which may be more suitable for qualitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")
                    elif hasattr(y_train, 'shape'):
                        unique_count = len(np.unique(y_train))
                        if unique_count <= 10:
                            self.display_message("Warning: The current target variable has few unique values ({}), which may be more suitable for qualitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")
        else:
            self.analysis_desc.setText("Qualitative Analysis: Classification prediction (Category analysis)")
            # Check if the data is suitable for qualitative analysis
            if hasattr(self, 'controller') and hasattr(self.controller, 'data_controller'):
                data_model = self.controller.data_controller.data_model
                if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                    y_train = data_model.y_train
                    # Check if it's continuous value (continuous values usually have more unique values)
                    if hasattr(y_train, 'nunique'):
                        unique_count = y_train.nunique()
                        if unique_count > 10 and unique_count / len(y_train) > 0.1:  # Many unique values with high ratio
                            self.display_message("Warning: The current target variable has many unique values ({}), which may be more suitable for quantitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")
                    elif hasattr(y_train, 'shape'):
                        unique_count = len(np.unique(y_train))
                        if unique_count > 10 and unique_count / len(y_train) > 0.1:
                            self.display_message("Warning: The current target variable has many unique values ({}), which may be more suitable for quantitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")

    def create_analysis_controls(self):
        """Create analysis control area"""
        analysis_group = QGroupBox("Analysis Settings")
        layout = QVBoxLayout()
        # Add task type selector
        task_type_layout = QHBoxLayout()
        self.task_type_label = QLabel("Task Type:")
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Classification Task", "Regression Task"])
        self.task_type_combo.currentIndexChanged.connect(self.on_task_type_changed)
        task_type_layout.addWidget(self.task_type_label)
        task_type_layout.addWidget(self.task_type_combo)
        layout.addLayout(task_type_layout)
        # Analysis type selection
        analysis_layout = QHBoxLayout()
        self.analysis_type_label = QLabel("Analysis Type:")
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Qualitative", "Quantitative"])
        self.analysis_type.currentTextChanged.connect(self.on_analysis_type_changed)
        analysis_layout.addWidget(self.analysis_type_label)
        analysis_layout.addWidget(self.analysis_type)
        layout.addLayout(analysis_layout)
        # Analysis method selection
        method_layout = QHBoxLayout()
        self.analysis_method_label = QLabel("Analysis Method:")
        self.analysis_method = QComboBox()
        method_layout.addWidget(self.analysis_method_label)
        method_layout.addWidget(self.analysis_method)
        layout.addLayout(method_layout)

        # Evaluation metric selection
        eval_layout = QHBoxLayout()
        self.evaluation_metric_label = QLabel("Evaluation Metric:")
        self.evaluation_metric = QComboBox()
        eval_layout.addWidget(self.evaluation_metric_label)
        eval_layout.addWidget(self.evaluation_metric)
        layout.addLayout(eval_layout)
        # Auto update analysis methods and evaluation metrics
        self.update_analysis_methods(self.analysis_type.currentText())
        # Analysis button
        analysis_button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Start Analysis")
        self.analyze_button.clicked.connect(self.start_analysis)
        analysis_button_layout.addWidget(self.analyze_button)
        layout.addLayout(analysis_button_layout)
        # Analysis description
        self.analysis_desc = QLabel("Choose analysis type")
        self.analysis_desc.setStyleSheet("font-style: italic; color: #666666;")
        layout.addWidget(self.analysis_desc)
        # Results table
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["Method", "R²", "RMSE", "MAE", "F1/AUC"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.result_table)
        analysis_group.setLayout(layout)
        return analysis_group
    def on_task_type_changed(self, index):
        """Handle task type change"""
        task_type = "classification" if index == 0 else "regression"
        # Synchronize task type and analysis type
        if task_type == "classification" and self.analysis_type.currentText() != "Qualitative":
            self.analysis_type.setCurrentText("Qualitative")
        elif task_type == "regression" and self.analysis_type.currentText() != "Quantitative":
            self.analysis_type.setCurrentText("Quantitative")
        # Set task_type in controller
        if hasattr(self, 'controller'):
            self.controller.task_type = task_type
            print(f"Task type updated to: {task_type}")
            # If data model exists, update its task_type
            if hasattr(self.controller, 'data_controller') and hasattr(self.controller.data_controller, 'data_model'):
                self.controller.data_controller.data_model.task_type = task_type
                print(f"Data model task type set to: {task_type}")
    def on_analysis_type_changed(self, analysis_type):
        """Handle analysis type change"""
        # Update analysis methods and evaluation metrics options
        self.update_analysis_methods(analysis_type)
        # Synchronize task type selector
        if analysis_type == "Qualitative" and self.task_type_combo.currentText() != "Classification Task":
            self.task_type_combo.setCurrentText("Classification Task")
        elif analysis_type == "Quantitative" and self.task_type_combo.currentText() != "Regression Task":
            self.task_type_combo.setCurrentText("Regression Task")
        # Update analysis description
        if analysis_type == "Qualitative":
            self.analysis_desc.setText("Qualitative Analysis: Classification Prediction (Category Analysis)")
            self.controller.analysis_type = "Qualitative"
            # Check if data is suitable for qualitative analysis
            if hasattr(self, 'controller') and hasattr(self.controller, 'data_controller'):
                data_model = self.controller.data_controller.data_model
                if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                    y_train = data_model.y_train
                    # Check if continuous values (continuous values usually have more unique values)
                    if hasattr(y_train, 'nunique'):
                        unique_count = y_train.nunique()
                        if unique_count > 10 and unique_count / len(y_train) > 0.1:  # Many unique values with high ratio
                            self.display_message("Warning: Current target variable has many unique values ({}), may be more suitable for quantitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")
                    elif hasattr(y_train, 'shape'):
                        unique_count = len(np.unique(y_train))
                        if unique_count > 10 and unique_count / len(y_train) > 0.1:
                            self.display_message("Warning: Current target variable has many unique values ({}), may be more suitable for quantitative analysis.".format(unique_count),
                                              "Analysis Type Suggestion")

    def on_scatter_method_changed(self, method):
        """Handle scatter correction method change"""
        # Enable settings button for all methods that have parameter dialogs
        if method in ["MSC", "SNV", "EMSC", "RNV", "OSC"]:
            self.settings_btn.setEnabled(True)
        else:
            self.settings_btn.setEnabled(False)
    def open_scatter_parameters(self):
        """
        A refactored method to open a clean, well-laid-out dialog 
        displaying all available parameters for the selected scatter correction method.
        """
        method = self.scatter_method.currentText()
        # Create a mapping from UI display names to actual plugin names
        method_mapping = {
            "MSC": "Multiplicative Scatter Correction (MSC)",
            "SNV": "Standard Normal Variate (SNV)",
            "EMSC": "Extended Multiplicative Scatter Correction (EMSC)",
            "RNV": "Robust Normal Variate (RNV)",
            "OSC": "Orthogonal Signal Correction (OSC)"
        }
        try:
            # BUG FIX V4: Use the method map to get the correct plugin name
            preprocessing_service = self.controller.preprocessing_controller.preprocessing_service
            plugin_name = method_mapping.get(method, method)
            param_info = None
            # First, check for a plugin
            if plugin_name in preprocessing_service.plugins:
                plugin = preprocessing_service.plugins[plugin_name]
                param_info = plugin.get_parameter_info()
            else:
                # If no plugin, provide parameters for built-in methods
                param_info = self.get_builtin_method_params(method)
            # If no configurable parameters, show an informational message
            if not param_info:
                algorithm_info = {
                    "MSC": "The Multiplicative Scatter Correction algorithm uses fixed default parameters:\n✅Reference Spectrum: Mean of all samples\n✅Correction factors are calculated automatically\n✅No manual adjustment is required.",
                    "EMSC": "The Extended MSC algorithm uses fixed default parameters:\n✅Reference Spectrum: Mean of all samples\n✅Polynomial Degree: 2\n✅Correction factors are calculated automatically.",
                    "RNV": "The Robust Normal Variate algorithm uses fixed default parameters:\n✅Robust Statistic: Median\n✅Robust standard deviation is calculated automatically.\n✅No manual adjustment is required.",
                    "OSC": "The Orthogonal Signal Correction algorithm uses fixed default parameters:\n✅Number of Components: 1\n✅Orthogonal components are calculated automatically.\n✅No manual adjustment is required."
                }
                info_text = algorithm_info.get(method, f"The {method} algorithm uses default parameters and requires no manual adjustment.")
                QMessageBox.information(self, f"{method} Algorithm Information", info_text)
                return

            dialog = QDialog(self)
            # Add detailed descriptions for each algorithm
            algorithm_descriptions = {
                "SNV": "Standard Normal Variate - Corrects multiplicative scatter effects.",
                "MSC": "Multiplicative Scatter Correction - Uses a reference spectrum to correct scatter.",
                "EMSC": "Extended MSC - Combines MSC with polynomial baseline correction.",
                "RNV": "Robust Normal Variate - A robust version of SNV for data with outliers.",
                "OSC": "Orthogonal Signal Correction - Removes variation orthogonal to the target variable."
            }
            dialog.setWindowTitle(f"{method} Parameter Settings")
            form_layout = QFormLayout(dialog)
            form_layout.setSpacing(12)
            form_layout.setContentsMargins(20, 20, 20, 20)
            # Add algorithm description label
            if method in algorithm_descriptions:
                desc_label = QLabel(algorithm_descriptions[method])
                desc_label.setStyleSheet("""
                    QLabel {
                        color: #2c3e50;
                        font-weight: bold;
                        font-size: 10pt;
                        padding: 5px;
                        background-color: #ecf0f1;
                        border-radius: 4px;
                        margin-bottom: 10px;
                    }
                """)
                form_layout.addRow(desc_label)
            widgets = {}
            # Dynamically load and display all parameters
            for param_name, info in param_info.items():
                # Use the English parameter name for the label
                label_text = info.get('name', param_name)
                widget = self.data_partitioning_view.create_widget(info)
                # Set appropriate widths and styles for different control types
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setFixedWidth(120)
                    widget.setStyleSheet("QSpinBox, QDoubleSpinBox { font-size: 10pt; }")
                elif isinstance(widget, QComboBox):
                    widget.setMinimumWidth(150)
                    widget.setStyleSheet("QComboBox { font-size: 10pt; }")
                elif isinstance(widget, QCheckBox):
                    # For checkboxes, use the English label
                    widget.setText(label_text)
                    widget.setStyleSheet("QCheckBox { font-size: 10pt; }")
                    form_layout.addRow(widget)
                    widgets[param_name] = widget
                    continue # Skip the standard addRow
                else:
                    widget.setFixedWidth(150)
                # Add parameter description as a tooltip
                if 'description' in info:
                    widget.setToolTip(info['description'])
                form_layout.addRow(f"{label_text}:", widget)
                widgets[param_name] = widget

            # Add a separator line
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("QFrame { color: #bdc3c7; margin: 10px 0; }")
            form_layout.addRow(line)
            # Create buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.button(QDialogButtonBox.Ok).setText("OK")
            button_box.button(QDialogButtonBox.Cancel).setText("Cancel")
            button_box.button(QDialogButtonBox.Ok).setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            button_box.button(QDialogButtonBox.Cancel).setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
            """)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            form_layout.addRow(button_box)
            dialog.setMinimumWidth(450)  
            dialog.setFixedHeight(dialog.sizeHint().height())

            if dialog.exec_():
                new_params = {}
                for name, widget in widgets.items():
                    if isinstance(widget, QCheckBox):
                        new_params[name] = widget.isChecked()
                    elif isinstance(widget, QComboBox):
                        new_params[name] = widget.currentText()
                    elif isinstance(widget, QLineEdit):
                        new_params[name] = widget.text()
                    else:
                        new_params[name] = widget.value()

                # If it's a plugin, use its set_parameters method
                if plugin_name in preprocessing_service.plugins:
                    plugin = preprocessing_service.plugins[plugin_name]
                    plugin.set_parameters(**new_params)
                else:
                    # If it's a built-in method, store parameters in an instance variable
                    if not hasattr(self, 'builtin_method_params'):
                        self.builtin_method_params = {}
                    self.builtin_method_params[method] = new_params
                self.display_message(f"{method} parameters updated.", "Success")

        except Exception as e:
            self.display_error(f"Failed to open {method} parameter settings: {e}")

    def _toggle_poly_order_visibility(self, method):
        """Show or hide the Poly Order row based on the selected method"""
        is_sgolay = (method == "S-Golay")
        self.poly_order_label.setVisible(is_sgolay)
        self.smooth_poly_order.setVisible(is_sgolay)
    def get_builtin_method_params(self, method):
        """Define parameter configurations for built-in methods from a technical perspective."""
        if method == "SNV":
            # SNV: Standard Normal Variate, corrects for multiplicative scatter effects.
            # Key parameters: centering, scaling, and min_std threshold.
            return {
                'center': {
                    'name': 'Center Data',
                    'type': 'bool',
                    'default': True,
                    'description': 'Subtract the mean of each spectrum (recommended).'
                },
                'scale': {
                    'name': 'Scale Data',
                    'type': 'bool',
                    'default': True,
                    'description': 'Divide by the standard deviation of each spectrum (recommended).'
                },
                'min_std': {
                    'name': 'Min Std Threshold',
                    'type': 'float',
                    'default': 1e-6,
                    'min': 1e-10,
                    'max': 1e-2,
                    'description': 'Safety threshold to avoid division by zero.'
                }
            }
        elif method == "MSC":
            # MSC: Multiplicative Scatter Correction, uses a reference spectrum.
            # Key parameter: reference spectrum calculation method.
            return {
                'reference_spectrum': {
                    'name': 'Reference Spectrum',
                    'type': 'str',
                    'default': 'mean',
                    'options': ['mean', 'median', 'first_sample'],
                    'description': 'Method to calculate the reference spectrum: mean (recommended), median, or first sample.'
                }
            }
        elif method == "EMSC":
            # EMSC: Extended MSC, includes polynomial baseline correction.
            # Key parameters: reference method and polynomial degree.
            return {
                'reference_spectrum': {
                    'name': 'Reference Spectrum',
                    'type': 'str',
                    'default': 'mean',
                    'options': ['mean', 'median', 'first_sample'],
                    'description': 'Method to calculate the reference spectrum.'
                },
                'polynomial_degree': {
                    'name': 'Polynomial Degree',
                    'type': 'int',
                    'default': 2,
                    'min': 0,
                    'max': 4,
                    'description': 'Degree of the polynomial for baseline correction (2 is often sufficient).'
                }
            }
        elif method == "RNV":
            # RNV: Robust Normal Variate, a robust version of SNV.
            # Key parameter: robust statistic method.
            return {
                'robust_statistic': {
                    'name': 'Robust Statistic',
                    'type': 'str',
                    'default': 'median',
                    'options': ['median', 'trimmed_mean_10', 'trimmed_mean_20'],
                    'description': 'Robust statistic to use: median (recommended), 10% trimmed mean, or 20% trimmed mean.'
                }
            }
        elif method == "OSC":
            # OSC: Orthogonal Signal Correction, removes orthogonal variation.
            # Key parameter: number of components to remove.
            return {
                'num_components': {
                    'name': 'Number of Components',
                    'type': 'int',
                    'default': 1,
                    'min': 1,
                    'max': 5,
                    'description': 'Number of orthogonal components to remove (1-2 is common; more may cause overfitting).'
                }
            }
        else:
            return None

    def auto_save_onnx_model_from_analysis(self, model, X_sample, method, evaluation_results):
        """
        Auto-save ONNX model from analysis process
        
        Args:
            model: trained model
            X_sample: sample data for inferring input shape
            method: training method name
            evaluation_results: evaluation results
        """
        try:
            if model is None:
                return
            
            # Auto-save as ONNX format
            print(f"🚀 Auto-saving analysis model to ONNX format...")
            
            onnx_path = self.onnx_service.auto_save_model(
                model=model,
                X_sample=X_sample,
                model_name=type(model).__name__,
                method=method,
                evaluation_results=evaluation_results
            )
            
            if onnx_path:
                # Display save success message
                print(f"🎉 Analysis model automatically saved to ONNX: {onnx_path}")
            else:
                print("ℹ️ Analysis model could not be exported to ONNX format")
                
        except Exception as e:
            print(f"Error in auto_save_onnx_model_from_analysis: {e}")
            # Do not interrupt main flow


