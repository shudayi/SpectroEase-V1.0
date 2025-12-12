from app.views.pca_explorer_dialog import PCAExplorerDialog
import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.utils.logger_config import get_logger
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QShortcut, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QToolBar, QVBoxLayout, QWidget
)
import matplotlib
matplotlib.use('Qt5Agg')

# Import UI design system
from app.config.ui_design_tokens import UIDesignTokens as DT
from app.utils.ui_helpers import (
    create_primary_button, create_secondary_button, create_compact_button,
    create_combobox, create_spinbox, create_double_spinbox, create_checkbox,
    create_label, create_form_row, create_button_row, create_standard_groupbox,
    create_checkbox_with_params, apply_design_tokens_to_button, 
    apply_design_tokens_to_layout
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app_dir = os.path.dirname(os.path.dirname(current_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from app.controllers.main_controller import MainController
from app.services.feature_extraction_service import FeatureExtractionService
from app.views.data_view import DataView, DataPartitioningView
from app.views.preprocessing_view_v2 import PreprocessingViewV2 as PreprocessingView
from app.views.feature_selection_view import FeatureSelectionView
from app.views.feature_extraction_view import FeatureExtractionView
from app.views.modeling_view import ModelingView
from app.views.evaluation_view import EvaluationView
from app.views.hyperparameter_optimization_view import HyperparameterOptimizationView
from app.views.llm_dialog import LLMDialog
from config.llm_config import LLMConfig
from app.services.llm_service import LLMService
from app.services.onnx_service import ONNXService
from app.services.custom_algorithm_cache import CustomAlgorithmCache

from plugins.preprocessing.spectrum_converter import SpectrumConverter
from plugins.preprocessing.spectrum_visualizer import SpectrumVisualizer
from plugins.preprocessing.spectrum_preprocessor import SpectrumPreprocessor
from plugins.feature_selection.spectrum_feature_extractor import SpectrumFeatureExtractor
from plugins.modeling.quantitative_analyzer import QuantitativeAnalyzer
from plugins.modeling.qualitative_analyzer import QualitativeAnalyzer
from plugins.reporting.spectrum_report_generator import SpectrumReportGenerator
from plugins.reporting.professional_spectrum_report import ProfessionalSpectrumReportGenerator
from app.views.visualization_widget import VisualizationWidget
from app.views.progress_dialog import ProgressDialog
from app.models.preprocessing_model import PreprocessingModel

class MainWindow(QMainWindow):
    def __init__(self, preprocessing_plugins, feature_selection_plugins, 
                 modeling_plugins, data_partitioning_plugins, llm_service):
        super().__init__()
        
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logo.png')
        self.setWindowIcon(QIcon(icon_path))
        
        # 🔧 Optimization #3: Initialize centralized logger
        self.logger = get_logger('main')
        self.data_logger = get_logger('data_flow')
        
        self.preprocessing_plugins = preprocessing_plugins
        self.feature_selection_plugins = feature_selection_plugins
        self.modeling_plugins = modeling_plugins
        self.data_partitioning_plugins = data_partitioning_plugins
        self.llm_service = llm_service
        self.onnx_service = ONNXService(save_directory="models")
        self.custom_algorithm_cache = CustomAlgorithmCache()  # Initialize cache service
        self.spectra_data = None
        self.wavelengths = None
        self.current_spectra = None
        
        self.feature_data_options = {
            'original': None,
            'preprocessed': None,
            'pca_features': None,
            'selected_features': None
        }
        self.current_feature_source = 'original'
        
        self.spectral_type = None
        self.spectral_config = None
        
        self.preprocessing_view = PreprocessingView(self.preprocessing_plugins)
        self.feature_selection_view = FeatureSelectionView(self.feature_selection_plugins)
        self.feature_extraction_view = FeatureExtractionView()
        self.modeling_view = ModelingView(self.modeling_plugins)
        self.evaluation_view = EvaluationView()
        self.hyperopt_view = HyperparameterOptimizationView()
        self.data_partitioning_view = DataPartitioningView(self.data_partitioning_plugins)
        
        self.set_application_style()
        self.init_ui()
        self.controller = MainController(self, None)  # translator removed
        
        # Load cached custom algorithms after UI initialization
        self.load_cached_algorithms()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.feature_extraction_service = FeatureExtractionService(
            data_model=self.controller.data_controller.data_model, 
            logger=self.logger,
            parent_view=self
        )
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
        
        # Apply DPI awareness and responsive sizing (Editor Comment 2)
        from app.utils.ui_scaling import ui_scaling_manager, apply_responsive_sizing, get_responsive_font
        
        # Apply DPI awareness
        ui_scaling_manager.apply_dpi_awareness()
        
        # Set responsive window size based on screen resolution
        apply_responsive_sizing(self, base_width=2000, base_height=1200)
        
        # Set responsive font
        app_font = get_responsive_font("Arial", 9)
        self.setFont(app_font)  # Apply this font to entire window
        
        # Print scaling information for verification
        scaling_info = ui_scaling_manager.get_scaling_info()
        print(f"🖥️ UI Scaling Applied:")
        print(f"   Screen: {scaling_info['screen_resolution']} @ {scaling_info['screen_dpi']} DPI")
        print(f"   Scale factors: UI={scaling_info['ui_scale_factor']:.2f}, Font={scaling_info['font_scale_factor']:.2f}")
        print(f"   Window size: {scaling_info['recommended_window_size']}")
        
        # Create a horizontal splitter to separate left and right content
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)  # Set splitter as central widget of main window
        # 🎨 UI Optimization: Create container for left panel with unified design system
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(DT.SPACING_RELAXED)  # 12px between major groups
        left_layout.setContentsMargins(*DT.MARGIN_CONTAINER)  # (10, 10, 10, 10)
        
        # Create a scroll area to wrap left content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(left_container)
        scroll_area.setMinimumWidth(DT.WIDTH_LEFT_PANEL_MIN)  # 580px
        scroll_area.setFrameShape(QScrollArea.NoFrame)    
        # 🎨 UI Optimization: Data Management Group
        self.data_group, data_layout = create_standard_groupbox(
            "Data Management",
            height=DT.HEIGHT_GROUP_COMPACT,  # 60px
            margin='standard'  # (8,8,8,8)
        )
        
        # Import data button - using primary button style
        self.import_btn = create_primary_button("Import Data", callback=None)
        data_layout.addWidget(self.import_btn)
        
        left_layout.addWidget(self.data_group)
        # 🎨 UI Optimization: Data Partitioning Group
        self.data_partitioning_group, data_partitioning_layout = create_standard_groupbox(
            "Data Partitioning",
            height=DT.HEIGHT_GROUP_MEDIUM,  # 160px
            margin='standard'  # (8,8,8,8)
        )
        
        # Data partitioning view container
        data_partition_container = QWidget()
        data_partition_container_layout = QVBoxLayout(data_partition_container)
        data_partition_container_layout.setSpacing(DT.SPACING_STANDARD)  # 8px
        data_partition_container_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)
        data_partition_container_layout.addWidget(self.data_partitioning_view)
        # Auto height - let it size based on content
        
        data_partitioning_layout.addWidget(data_partition_container)
        
        # Apply button - using primary button style
        self.data_partition_btn = create_primary_button(
            "Partition Data",
            callback=self.partition_data_with_params
        )
        data_partitioning_layout.addWidget(self.data_partition_btn)
        
        left_layout.addWidget(self.data_partitioning_group)
        # 🎨 UI Optimization: Preprocessing Group
        self.preprocessing_group = QGroupBox("Preprocessing")
        self.preprocessing_layout = QVBoxLayout()
        self.preprocessing_layout.setSpacing(DT.SPACING_STANDARD)  # 8px
        self.preprocessing_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)
        
        # 🎨 UI Optimization: Create preprocessing tabs with responsive height
        self.preprocessing_tabs = QTabWidget()
        # 计算响应式高度：每个子组约140px，两列并排，需要至少300px
        screen_height = self.screen().size().height() if self.screen() else 1080
        if screen_height >= 1440:  # 2K及以上
            tab_min_height = 400
            preprocessing_group_height = 540  # 400 + 80 + 28 + 32(边距间距)
        elif screen_height >= 1080:  # 1080p
            tab_min_height = 350
            preprocessing_group_height = 490  # 350 + 80 + 28 + 32
        else:  # 720p或更小
            tab_min_height = 320
            preprocessing_group_height = 460  # 320 + 80 + 28 + 32
        self.preprocessing_tabs.setMinimumHeight(tab_min_height)
        self.preprocessing_tabs.setTabText(0, "Basic")  # Set text of first tab to "Basic"
        self.preprocessing_tabs.setTabText(1, "Advanced")  # Set text of second tab to "Advanced"
        self.preprocessing_tabs.setTabText(2, "Special")  # Set text of third tab to "Special"
        # 🎨 UI Optimization: Basic preprocessing tab
        basic_tab = QWidget()
        basic_layout = QHBoxLayout()
        basic_layout.setSpacing(DT.SPACING_RELAXED)  # 12px between columns
        basic_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)
        
        # Left column
        left_column = QVBoxLayout()
        left_column.setSpacing(DT.SPACING_STANDARD)  # 8px
        
        # Right column  
        right_column = QVBoxLayout()
        right_column.setSpacing(DT.SPACING_STANDARD)  # 8px
        
        # --- Despiking Group (Left Column) ---
        despiking_group = QGroupBox("Despiking")
        despiking_outer_layout = QVBoxLayout()
        despiking_outer_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        despiking_outer_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)

        self.despiking_check = QCheckBox("Enable Despiking")
        self.despiking_check.setChecked(False)  # Default off
        despiking_outer_layout.addWidget(self.despiking_check)

        self.despiking_params_container = QWidget()
        despiking_form_layout = QFormLayout(self.despiking_params_container)
        despiking_form_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        despiking_form_layout.setContentsMargins(*DT.MARGIN_FORM_FIELD)  # (0,4,0,0)
        despiking_form_layout.setLabelAlignment(Qt.AlignLeft)

        # Method selection
        self.despiking_method = QComboBox()
        self.despiking_method.addItems(["MAD", "Local Z-score"])
        self.despiking_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.despiking_method.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        despiking_form_layout.addRow("Method:", self.despiking_method)

        # Window size
        self.despiking_window = QSpinBox()
        self.despiking_window.setRange(7, 15)
        self.despiking_window.setValue(7)
        self.despiking_window.setSingleStep(2)
        self.despiking_window.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.despiking_window.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        despiking_form_layout.addRow("Window:", self.despiking_window)

        # Threshold
        self.despiking_threshold = QDoubleSpinBox()
        self.despiking_threshold.setRange(3.0, 6.0)
        self.despiking_threshold.setValue(5.0)
        self.despiking_threshold.setSingleStep(0.5)
        self.despiking_threshold.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.despiking_threshold.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
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
        baseline_outer_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        baseline_outer_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)

        self.baseline_check = QCheckBox("Enable Baseline Correction")
        baseline_outer_layout.addWidget(self.baseline_check)

        self.baseline_params_container = QWidget()
        baseline_form_layout = QFormLayout(self.baseline_params_container)
        baseline_form_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        baseline_form_layout.setContentsMargins(*DT.MARGIN_FORM_FIELD)  # (0,4,0,0)
        baseline_form_layout.setLabelAlignment(Qt.AlignLeft)

        # Method selection for baseline correction
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["Polynomial", "ALS", "airPLS"])
        self.baseline_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.baseline_method.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        baseline_form_layout.addRow("Method:", self.baseline_method)

        self.poly_order = QSpinBox()
        self.poly_order.setRange(1, 10)
        self.poly_order.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.poly_order.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        baseline_form_layout.addRow("Polynomial Order:", self.poly_order)
        baseline_outer_layout.addWidget(self.baseline_params_container)
        baseline_group.setLayout(baseline_outer_layout)

        self.baseline_check.toggled.connect(self.baseline_params_container.setEnabled)
        self.baseline_params_container.setEnabled(False)

        left_column.addWidget(baseline_group)
        # --- Scatter Correction Group (Rebuilt with QFormLayout) ---
        scatter_group = QGroupBox("Scatter Correction")
        scatter_outer_layout = QVBoxLayout()
        scatter_outer_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        scatter_outer_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)

        self.scatter_check = QCheckBox("Enable Scatter Correction")
        scatter_outer_layout.addWidget(self.scatter_check)

        self.scatter_params_container = QWidget()
        scatter_form_layout = QFormLayout(self.scatter_params_container)
        scatter_form_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        scatter_form_layout.setContentsMargins(*DT.MARGIN_FORM_FIELD)  # (0,4,0,0)
        scatter_form_layout.setLabelAlignment(Qt.AlignLeft)

        self.scatter_method = QComboBox()
        self.scatter_method.addItems(["MSC", "SNV", "EMSC", "RNV", "OSC"])
        self.scatter_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.scatter_method.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFixedHeight(DT.HEIGHT_BUTTON_SECONDARY)  # 24px
        self.settings_btn.setFixedWidth(DT.WIDTH_BUTTON_COMPACT)  # 90px
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
        smooth_outer_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        smooth_outer_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # (8,8,8,8)

          
        self.smooth_check = QCheckBox("Enable Smoothing")
        self.smooth_check.setFixedHeight(DT.HEIGHT_CHECKBOX)  # 24px
        smooth_outer_layout.addWidget(self.smooth_check)

          
        self.smoothing_params_container = QWidget()   
        smooth_form_layout = QFormLayout(self.smoothing_params_container)
        smooth_form_layout.setSpacing(DT.SPACING_TIGHT)  # 4px
        smooth_form_layout.setContentsMargins(*DT.MARGIN_FORM_FIELD)  # (0,4,0,0)
        smooth_form_layout.setLabelAlignment(Qt.AlignLeft)   

          
        self.smooth_method = QComboBox()
        self.smooth_method.addItems(["S-Golay", "Moving Avg", "Median", "Wavelet"])
        self.smooth_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.smooth_method.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        smooth_form_layout.addRow("Method:", self.smooth_method)
          
        self.window_size = QSpinBox()
        self.window_size.setRange(3, 51)
        self.window_size.setValue(11)
        self.window_size.setSingleStep(2)
        self.window_size.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.window_size.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
        smooth_form_layout.addRow("Window:", self.window_size)
          
        self.poly_order_label = QLabel("Poly:")
        self.smooth_poly_order = QSpinBox()
        self.smooth_poly_order.setRange(2, 5)
        self.smooth_poly_order.setValue(2)
        self.smooth_poly_order.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.smooth_poly_order.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)  # 180px
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
        standardize_params = QHBoxLayout()    
        standardize_params.setSpacing(1)    
        std_method_label = QLabel("Method:")    
        std_method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        std_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        standardize_params.addWidget(std_method_label)    
        self.standardize_method = QComboBox()    
        self.standardize_method.addItems(["Z-Score", "Robust", "Min-Max"])    
        self.standardize_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px    
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
        norm_params = QHBoxLayout()    
        norm_params.setSpacing(2)    
        norm_method_label = QLabel("Method:")    
        norm_method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        norm_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        norm_params.addWidget(norm_method_label)    
        self.norm_method = QComboBox()    
        self.norm_method.addItems(["Min-Max", "Vector", "Area", "Maximum"])    
        self.norm_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px    
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
        derivative_params = QHBoxLayout()    
        derivative_params.setSpacing(2)    
        derivative_order_label = QLabel("Order:")    
        derivative_order_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px    
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
        outlier_params = QHBoxLayout()    
        outlier_params.setSpacing(4)    
        outlier_threshold_label = QLabel("Threshold:")    
        outlier_threshold_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px    
        outlier_threshold_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
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
        denoise_params = QHBoxLayout()    
        denoise_params.setSpacing(4)    
        denoise_strength_label = QLabel("Strength:")    
        denoise_strength_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px    
        denoise_strength_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
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
        alignment_params = QHBoxLayout()    
        alignment_params.setSpacing(4)    
        alignment_method_label = QLabel("Method:")    
        alignment_method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        alignment_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        alignment_params.addWidget(alignment_method_label)    
        self.alignment_method = QComboBox()    
        self.alignment_method.addItems(["DTW", "COW", "ICS", "PAFFT"])    
        self.alignment_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px    
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
        reference_method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        reference_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        reference_method_layout.addWidget(reference_method_label)    
        self.reference_method = QComboBox()    
        self.reference_method.addItems(["Mean", "Med", "Max", "First"])    
        self.reference_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px    
        reference_method_layout.addWidget(self.reference_method)    
        reference_layout.addLayout(reference_method_layout)    
        reference_group.setLayout(reference_layout)    
        reference_group.setMaximumHeight(70)    
        right_column.addWidget(reference_group)    
        special_form_layout.addLayout(right_column)    
        special_layout.addLayout(special_form_layout)
        
        special_tab.setLayout(special_layout)    
        self.preprocessing_tabs.addTab(special_tab, "Special")    
        
        # V1.3.1: Custom Tab for user-defined preprocessing algorithms
        custom_tab = self._create_custom_preprocessing_tab()
        self.preprocessing_tabs.addTab(custom_tab, "Custom")
        
        self.preprocessing_layout.addWidget(self.preprocessing_tabs)    
          
        checkboxes = [
            self.despiking_check, self.baseline_check, self.scatter_check, self.smooth_check, 
            self.standardize_check, self.norm_check, self.derivative_check,
            self.outlier_check, self.denoise_check, self.alignment_check
        ]
        for checkbox in checkboxes:    
            checkbox.setStyleSheet("QCheckBox { padding: 2px; margin: 1px; }")    
          
        # 🎨 UI Optimization: Preview Group
        preview_group, preview_layout = create_standard_groupbox(
            "Preview",
            height=DT.HEIGHT_PREVIEW_GROUP,  # 80px
            margin='standard'
        )
        preview_layout_h = QHBoxLayout()
        preview_layout_h.setSpacing(DT.SPACING_STANDARD)  # 8px
          
        display_label = create_label("Display:", width=60, align='left')
        preview_layout_h.addWidget(display_label)
        
        self.preview_mode = create_combobox(
            ["Original", "Processed", "Comparison"],
            width=140,
            callback=self.on_preview_mode_changed
        )
        preview_layout_h.addWidget(self.preview_mode)
          
        preview_btn = create_secondary_button("Preview", callback=self.preview_preprocessing)
        preview_layout_h.addWidget(preview_btn)
        preview_layout_h.addStretch()
        
        preview_layout.addLayout(preview_layout_h)
        self.preprocessing_layout.addWidget(preview_group)
        
        self.update_preview_mode_availability()    
          
        # 🎨 UI Optimization: Preprocessing Button Group
        self.preprocess_btn = create_primary_button("Apply", callback=self.apply_preprocessing)
        reset_btn = create_secondary_button("Reset", callback=self.reset_preprocessing_params)
        help_btn = create_secondary_button("Help", callback=self.show_preprocessing_help)
        
        button_row = create_button_row([self.preprocess_btn, reset_btn, help_btn], spacing='standard')
        button_row.setContentsMargins(*DT.MARGIN_STANDARD)
        self.preprocessing_layout.addLayout(button_row)    
        self.preprocessing_group.setLayout(self.preprocessing_layout)    
        # 🎨 UI Optimization: 设置预处理组固定高度，避免下方空白
        self.preprocessing_group.setFixedHeight(preprocessing_group_height)
        left_layout.addWidget(self.preprocessing_group)    
          
        # 🎨 UI Optimization: Feature Selection Group
        self.feature_group, feature_layout = create_standard_groupbox(
            "Feature Selection",
            height=DT.HEIGHT_GROUP_SMALL,  # 100px
            margin='standard'
        )
        
        # Method selection row
        method_layout = QHBoxLayout()
        method_layout.setSpacing(DT.SPACING_STANDARD)  # 8px
        
        method_label = create_label("Method:", width=60, align='left')
        method_layout.addWidget(method_label)
        
        # V1.4.0: Dynamic feature selection methods (built-in + plugins)
        self.feature_method = create_combobox([
            "CARS", "SPA", "PLSR", "SelectKBest",
            "Mutual Information", "RFE", "PCA"
        ], width='standard')
        self._update_feature_selection_methods()  # Update with plugins
        method_layout.addWidget(self.feature_method)
        
        # Parameter input
        param_label = create_label("Components/Features:", width=140, align='left')
        method_layout.addWidget(param_label)
        
        self.feature_param_spin = create_spinbox(1, 100, 10, width=DT.WIDTH_CONTROL_COMPACT)
        method_layout.addWidget(self.feature_param_spin)
        method_layout.addStretch()
        
        feature_layout.addLayout(method_layout)
        
        # Apply button
        self.feature_btn = create_primary_button("Apply", callback=self._on_feature_btn_clicked)
        feature_layout.addWidget(self.feature_btn)
        
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
        type_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        type_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        type_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        analysis_controls.addWidget(type_label)    
        self.analysis_type = QComboBox()    
        self.analysis_type.addItems(["Quantitative", "Qualitative"])    
        self.analysis_type.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px
        self.analysis_type.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        analysis_controls.addWidget(self.analysis_type)    
        method_label = QLabel("Method:")    
        method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        method_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        analysis_controls.addWidget(method_label)    
        self.analysis_method = QComboBox()    
        self.analysis_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px
        self.analysis_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px    
        analysis_controls.addWidget(self.analysis_method)    
        analysis_layout.addLayout(analysis_controls)    
          
        hyperparameter_group = QGroupBox("Hyperparameter Optimization")
        hyperparameter_group.setFixedHeight(90)    
        hyperparameter_layout = QVBoxLayout()    
        hyperparameter_layout.setContentsMargins(5, 5, 5, 5)    
        hyperparameter_layout.setSpacing(5)    
          
        opt_method_row = QHBoxLayout()    
        opt_method_label = QLabel("Method:")    
        opt_method_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        opt_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        opt_method_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        opt_method_row.addWidget(opt_method_label)    
        self.optimization_method = QComboBox()    
        self.optimization_method.addItems(["Grid Search", "Random Search", "Bayesian"])    
        self.optimization_method.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px
        self.optimization_method.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px    
        opt_method_row.addWidget(self.optimization_method)    
        hyperparameter_layout.addLayout(opt_method_row)    
          
        metric_cv_row = QHBoxLayout()    
        metric_label = QLabel("Metric:")    
        metric_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        metric_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        metric_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        metric_cv_row.addWidget(metric_label)    
        self.evaluation_metric = QComboBox()    
        self.evaluation_metric.addItems(["RMSE", "R²", "Accuracy", "F1"])    
        self.evaluation_metric.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)  # 220px
        self.evaluation_metric.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px    
        metric_cv_row.addWidget(self.evaluation_metric)    
        hyperparameter_layout.addLayout(metric_cv_row)    
          
        cv_iter_row = QHBoxLayout()    
        cv_label = QLabel("CV Folds:")    
        cv_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        cv_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        cv_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px    
        cv_iter_row.addWidget(cv_label)    
        self.cv_folds = QSpinBox()    
        self.cv_folds.setRange(2, 10)    
        self.cv_folds.setValue(5)    
        self.cv_folds.setMinimumWidth(50)    
        self.cv_folds.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        cv_iter_row.addWidget(self.cv_folds)    
        iter_label = QLabel("Max Iter:")    
        iter_label.setMinimumWidth(40)    
        iter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)    
        iter_label.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        cv_iter_row.addWidget(iter_label)    
        self.max_iterations = QSpinBox()    
        self.max_iterations.setRange(10, 1000)    
        self.max_iterations.setValue(100)    
        self.max_iterations.setSingleStep(10)    
        self.max_iterations.setMinimumWidth(50)    
        self.max_iterations.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px    
        cv_iter_row.addWidget(self.max_iterations)    
        hyperparameter_layout.addLayout(cv_iter_row)    
        hyperparameter_group.setLayout(hyperparameter_layout)    
        hyperparameter_group.setFixedHeight(95)    
        analysis_layout.addWidget(hyperparameter_group)    
          
        # 🎨 UI Optimization: Start Analysis Button
        self.analyze_btn = create_primary_button("Start Analysis", callback=self.start_analysis)
        analysis_layout.addWidget(self.analyze_btn)    
        
        self.analysis_group.setLayout(analysis_layout)    
        left_layout.addWidget(self.analysis_group)    
          
        # 🎨 UI Optimization: Set Size Policies
        self.data_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.preprocessing_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)    
        self.data_partitioning_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.feature_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)    
        self.analysis_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)    
          
        # 🎨 UI Optimization: Generate Report Button
        report_btn = create_primary_button("Generate Report", callback=self.generate_report)
        left_layout.addWidget(report_btn)    
          
        # 🎨 UI Optimization: Right Panel
        right_panel = QWidget()    
        right_layout = QVBoxLayout(right_panel)    
        right_layout.setSpacing(DT.SPACING_STANDARD)  # 8px
        right_layout.setContentsMargins(*DT.MARGIN_CONTAINER)  # (10,10,10,10)
          
        self.visualization_widget = VisualizationWidget()    
        self.visualization_widget.setMinimumHeight(DT.HEIGHT_VISUALIZATION_MIN)  # 400px
        right_layout.addWidget(self.visualization_widget)    
          
        # 🎨 UI Optimization: Result Table
        self.result_table = QTableWidget()
        self.result_table.setMinimumHeight(DT.HEIGHT_RESULT_TABLE_MIN)  # 180px
        self.result_table.setMaximumHeight(DT.HEIGHT_RESULT_TABLE_MAX)  # 300px
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
          
        # 🎨 UI Optimization: Add panels to splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(right_panel)
          
        # 🎨 UI Optimization: Configure splitter  
        self.splitter = splitter  # Store for resize event
        splitter.setChildrenCollapsible(False)    
        splitter.setHandleWidth(DT.SPLITTER_HANDLE_WIDTH)  # 4px
        splitter.setStretchFactor(0, 1)    
        splitter.setStretchFactor(1, 3)    
        
        # Apply responsive splitter sizes
        self.apply_responsive_splitter_sizes()    
          
        self.setup_toolbar()    
          
        self.setup_status_bar()    
          
        self.setup_shortcuts()    
          
        self.setup_operation_history()    
          
        self.update_analysis_methods(self.analysis_type.currentText())    
    
    def apply_responsive_splitter_sizes(self):
        """🎨 UI Optimization: Apply responsive splitter sizes based on window width"""
        if hasattr(self, 'splitter'):
            total_width = self.width()
            left_width = int(total_width * DT.SPLITTER_LEFT_RATIO)  # 25%
            right_width = int(total_width * DT.SPLITTER_RIGHT_RATIO)  # 75%
            self.splitter.setSizes([left_width, right_width])
    
    def resizeEvent(self, event):
        """🎨 UI Optimization: Handle window resize to adjust splitter"""
        super().resizeEvent(event)
        self.apply_responsive_splitter_sizes()    

    def setup_toolbar(self):
        """Setup the main toolbar with action buttons"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))    
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)    
        self.addToolBar(toolbar)
          
        llm_action = QAction("AI Assistant", self)
        if self.llm_service is None:
            llm_action.setStatusTip("AI Assistant (Not Configured - Click AI Config to set up)")
            llm_action.setToolTip("AI Assistant (Not Configured)\n\nClick 'AI Config' to set up your API key")
        else:
            llm_action.setStatusTip("AI Assistant - Convert algorithm code with AI")
            llm_action.setToolTip("AI Assistant\n\nAutomatically convert algorithm code to system format")
        llm_action.triggered.connect(self.show_llm_dialog)
        toolbar.addAction(llm_action)
          
        llm_config_action = QAction("AI Config", self)
        llm_config_action.setStatusTip("Configure AI Assistant API settings")
        llm_config_action.setToolTip("AI Configuration\n\nSet up your API key for AI Assistant")
        llm_config_action.triggered.connect(self.show_llm_config)
        toolbar.addAction(llm_config_action)
        
        manage_algorithms_action = QAction("Manage Algorithms", self)
        manage_algorithms_action.setStatusTip("Manage saved custom algorithms")
        manage_algorithms_action.setToolTip("View and manage custom algorithms")
        manage_algorithms_action.triggered.connect(self.show_algorithm_manager)
        toolbar.addAction(manage_algorithms_action)
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
          
        # Load Data button removed from DataPartitioningView - use Import Data button instead
          
        # Note: preprocess_btn is already connected via callback in create_primary_button
        # Don't connect again here to avoid duplicate calls
        # self.preprocess_btn.clicked.connect(self.apply_preprocessing)  # REMOVED: Already connected in init_ui
          
        # This is a duplicate connection, the correct one is in init_ui.
        # self.feature_btn.clicked.connect(self.apply_feature_selection)
        
        # Connect feature extraction signal (Editor Comment 4)
        self.feature_extraction_view.extraction_requested.connect(self.handle_feature_extraction)
          
          
        # self.analyze_btn.clicked.connect(    
        #     lambda: self.controller.check_data_split() and 
        #     self.controller.modeling_controller.train_model()
        # )
          
        self.analysis_type.currentTextChanged.connect(self.update_analysis_methods)    
    
    def extract_features_unsupervised(self):
        """Handle unsupervised feature extraction (Editor Comment 4)"""
        if self.current_spectra is None:
            QMessageBox.warning(self, "Warning", "Please load spectral data first")
            return
        
        method = self.extraction_method.currentText()
        n_components = self.extraction_param_spin.value()
        
        params = {
            'n_components': n_components,
            'cev_threshold': 0.95,
            'standardize': True,
            'show_loadings': True
        }
        
        self.handle_feature_extraction(method.lower(), params)
    
    def handle_feature_extraction(self, method, params):
        """Handle feature extraction request (Editor Comment 4)"""
        try:
            if self.current_spectra is None:
                QMessageBox.warning(self, "Warning", "Please load spectral data first")
                return
            
            from app.services.feature_extraction_service import feature_extraction_service
            
            # Prepare data (exclude label column if present)
            X = self.current_spectra.copy()
            if hasattr(self, 'target_column_index') and self.target_column_index is not None:
                if self.target_column_index < X.shape[1]:
                    X = X.drop(X.columns[self.target_column_index], axis=1)
            
            print(f"🔬 Starting {method.upper()} feature extraction...")
            print(f"   Input data shape: {X.shape}")
            
            if method == 'pca':
                results = feature_extraction_service.extract_pca_features(
                    X, 
                    n_components=params.get('n_components'),
                    cev_threshold=params.get('cev_threshold', 0.95)
                )
                
                self.feature_data_options['pca_features'] = results['transformed_data']
                self.pca_results = results
                
                print(f"✅ Feature extraction completed: {results['n_components']} components extracted")
                print(f"   Original spectra preserved: {self.current_spectra.shape}")
                print(f"   PCA features stored: {self.feature_data_options['pca_features'].shape}")
                
                # Display results in feature extraction view
                self.feature_extraction_view.display_pca_results(results)
                
                reply = QMessageBox.question(
                    self, "Use PCA Features for Modeling?",
                    f"Successfully extracted {results['n_components']} principal components "
                    f"explaining {results['cumulative_explained_variance'][-1]:.1%} of variance.\n\n"
                    f"Would you like to use these PCA features for subsequent modeling?\n\n"
                    f"• Yes: Use {results['n_components']} PCA components for modeling\n"
                    f"• No: Keep using original {self.current_spectra.shape[1]} wavelengths",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.current_feature_source = 'pca_features'
                    QMessageBox.information(
                        self, "PCA Features Selected",
                        f"Now using {results['n_components']} PCA components for modeling.\n"
                        f"You can proceed with data partitioning and modeling."
                    )
                else:
                    QMessageBox.information(
                        self, "Original Data Retained",
                        f"PCA features are available but original data will be used for modeling.\n"
                        f"You can switch to PCA features later if needed."
                    )
                
            else:
                QMessageBox.warning(self, "Warning", f"Method {method} not implemented yet")
                
        except Exception as e:
            error_msg = f"Feature extraction failed: {str(e)}"
            print(f"❌ {error_msg}")
            QMessageBox.critical(self, "Feature Extraction Error", error_msg)
    
    def get_current_feature_data(self):
        """Get the currently selected feature data for modeling"""
        if self.current_feature_source == 'selected_features' and self.feature_data_options['selected_features'] is not None:
            selected_data = self.feature_data_options['selected_features']
            print(f"🎯 Using selected features for analysis:")
            print(f"   Train: {selected_data['X_train'].shape}")
            print(f"   Test: {selected_data['X_test'].shape if selected_data['X_test'] is not None else 'None'}")
            print(f"   Selected features: {len(selected_data['selected_features'])}")
            return selected_data
        elif self.current_feature_source == 'pca_features' and self.feature_data_options['pca_features'] is not None:
            print(f"🔬 Using PCA features for analysis: {self.feature_data_options['pca_features'].shape}")
            return self.feature_data_options['pca_features']
        elif self.current_feature_source == 'preprocessed' and self.feature_data_options['preprocessed'] is not None:
            print(f"🔧 Using preprocessed data for analysis: {self.feature_data_options['preprocessed'].shape}")
            return self.feature_data_options['preprocessed']
        else:
            print(f"📊 Using original spectral data for analysis: {self.current_spectra.shape}")
            return self.current_spectra    
          
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
        """Update data view and auto-visualize"""
        try:
            # Add data view component
            data_view = DataView()
            data_view_widget = data_view.update_data(data)
            if data_view_widget is not None:
                self.data_group.layout().addWidget(data_view_widget)
            
            # Auto-update visualization
            if hasattr(self, 'visualization_widget') and data is not None:
                wavelengths = None
                spectra_data = None
                labels = None

                # Correctly extract wavelengths and data from the incoming DataFrame
                if data.shape[1] > 1:
                    # Assume first column is labels, rest are spectral data
                    wavelengths = data.columns[1:]
                    spectra_data = data.iloc[:, 1:].values
                    labels = data.iloc[:, 0].values
                else:
                    # No label column
                    wavelengths = data.columns
                    spectra_data = data.values
                    labels = None
                
                # Convert wavelengths to a numeric numpy array
                try:
                    numeric_wavelengths = pd.to_numeric(wavelengths).to_numpy()
                except (ValueError, TypeError):
                    print("⚠️ Could not convert wavelengths to numeric. Using as is.")
                    numeric_wavelengths = wavelengths.to_numpy()

                # Set instance attributes for use throughout the application
                self.original_spectra = spectra_data.copy()
                self.current_spectra = spectra_data.copy()
                self.sample_labels = labels
                self.wavelengths = numeric_wavelengths
                
                print(f"🔍 Data loading verification:")
                print(f"   Original data shape: {self.original_spectra.shape}")
                print(f"   Original data range: [{self.original_spectra.min():.2f}, {self.original_spectra.max():.2f}]")
                import hashlib
                original_hash = hashlib.md5(self.original_spectra.tobytes()).hexdigest()
                print(f"   Original data hash: {original_hash[:16]}... (on load)")
                
                # Pass spectral type to visualization component
                if hasattr(self, 'spectral_type') and self.spectral_type:
                    self.visualization_widget.spectral_type = self.spectral_type
                if hasattr(self, 'spectral_config') and self.spectral_config:
                    self.visualization_widget.spectral_config = self.spectral_config

                # Use unified update_preprocessing_preview method
                self.visualization_widget.update_preprocessing_preview(
                    data_model=self.controller.data_controller.data_model,
                    wavelengths=self.wavelengths,
                    preview_mode='Original',
                    methods_applied=[]
                )
                print(f"✅ Loading complete, display mode: Original")

                # print("✅ Data visualization auto-updated")  # Too frequent, commented out
        except Exception as e:
            print(f"⚠️ Data view update failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: at least add data view
            try:
                data_view = DataView()
                data_view_widget = data_view.update_data(data)
                if data_view_widget is not None:
                    self.data_group.layout().addWidget(data_view_widget)
            except:
                pass

    def update_split_view(self, train_data, test_data):
        """Update the view with training and testing data partitions"""
          
        pass

    def _create_custom_preprocessing_tab(self):
        """V1.3.1: Create Custom Tab for user-defined preprocessing algorithms"""
        custom_tab = QWidget()
        custom_layout = QVBoxLayout()
        custom_layout.setSpacing(3)
        custom_layout.setContentsMargins(5, 5, 5, 5)
        
        # Simple description
        info_label = QLabel("Custom preprocessing algorithms (added via Tools → Algorithm Conversion)")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        custom_layout.addWidget(info_label)
        
        # Custom algorithm checkbox container
        self.custom_preproc_layout = QVBoxLayout()
        self.custom_preproc_layout.setSpacing(2)
        custom_layout.addLayout(self.custom_preproc_layout)
        
        # No algorithm prompt
        self.no_custom_preproc_label = QLabel("No custom algorithms")
        self.no_custom_preproc_label.setStyleSheet("color: #999; padding: 5px;")
        self.custom_preproc_layout.addWidget(self.no_custom_preproc_label)
        
        custom_layout.addStretch()
        custom_tab.setLayout(custom_layout)
        
        # Initialize storage
        if not hasattr(self, 'custom_preprocessing_algorithms'):
            self.custom_preprocessing_algorithms = {}
        if not hasattr(self, 'custom_preprocessing_checkboxes'):
            self.custom_preprocessing_checkboxes = {}
        
        return custom_tab
    
    def _add_custom_preprocessing_to_ui(self, code: str):
        """V1.3.1: Add custom preprocessing algorithm to Custom Tab UI"""
        print(f"🔍 DEBUG: _add_custom_preprocessing_to_ui called")
        print(f"🔍 DEBUG: no_custom_preproc_label exists: {hasattr(self, 'no_custom_preproc_label')}")
        print(f"🔍 DEBUG: custom_preproc_layout exists: {hasattr(self, 'custom_preproc_layout')}")
        
        try:
            import types
            import pandas as pd
            import numpy as np
            from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
            
            # Dynamically load algorithm
            mod = types.ModuleType('custom_algo')
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'PreprocessingAlgorithm': PreprocessingAlgorithm,
                'Dict': __import__('typing').Dict,
                'Any': __import__('typing').Any,
                'List': __import__('typing').List,
                'Tuple': __import__('typing').Tuple,
            })
            
            # V1.3.6: Add scipy support (if available)
            try:
                import scipy
                from scipy.signal import savgol_filter
                mod.__dict__.update({
                    'scipy': scipy,
                    'savgol_filter': savgol_filter,
                })
            except ImportError:
                pass
            
            # V1.3.6: Add sklearn support (if available)
            try:
                from sklearn.preprocessing import StandardScaler
                mod.__dict__.update({
                    'StandardScaler': StandardScaler,
                })
            except ImportError:
                pass
            
            exec(code, mod.__dict__)
            print(f"✅ Code executed successfully")
            
            # Find algorithm class (must be a subclass of PreprocessingAlgorithm)
            for item in mod.__dict__.values():
                if isinstance(item, type):
                    # Must be a subclass of PreprocessingAlgorithm, not the abstract base class itself
                    if (issubclass(item, PreprocessingAlgorithm) and 
                        item.__name__ not in ['PreprocessingAlgorithm', 'FeatureSelectionAlgorithm', 
                                              'ModelingAlgorithm', 'DataPartitioningAlgorithm']):
                        try:
                            algorithm = item()
                            algorithm_name = algorithm.get_name()
                            print(f"✅ Found algorithm: {algorithm_name}")
                            
                            # Store algorithm
                            self.custom_preprocessing_algorithms[algorithm_name] = algorithm
                            
                            # Also add to preprocessing_view plugins so preprocessing service can find it
                            if hasattr(self, 'preprocessing_view') and self.preprocessing_view:
                                if self.preprocessing_view.plugins is None:
                                    self.preprocessing_view.plugins = {}
                                self.preprocessing_view.plugins[algorithm_name] = algorithm
                                # Also add to custom_algorithms dict
                                self.preprocessing_view.custom_algorithms[algorithm_name] = algorithm
                                print(f"✅ Also added to preprocessing_view plugins: {algorithm_name}")
                            
                            # Hide "no algorithm" prompt
                            if hasattr(self, 'no_custom_preproc_label') and self.no_custom_preproc_label.isVisible():
                                self.no_custom_preproc_label.hide()
                                print(f"✅ Hidden 'no algorithm' label")
                            
                            # V1.3.5: Use concise custom identifier
                            checkbox = QCheckBox(f"[Custom Preproc] {algorithm_name}")
                            checkbox.setStyleSheet("QCheckBox { padding: 2px; margin: 1px; }")
                            self.custom_preprocessing_checkboxes[algorithm_name] = checkbox
                            self.custom_preproc_layout.addWidget(checkbox)
                            
                            print(f"✅ Custom preprocessing algorithm '{algorithm_name}' added to Custom Tab")
                            print(f"🔍 DEBUG: Total checkboxes in custom_preproc_layout: {self.custom_preproc_layout.count()}")
                            break
                        except Exception as e:
                            print(f"⚠️ Failed to instantiate {item.__name__}: {e}")
                            continue
                    
        except Exception as e:
            print(f"❌ Error adding custom preprocessing to UI: {e}")
            import traceback
            traceback.print_exc()

    def show_llm_dialog(self):
        """Show the LLM (Large Language Model) dialog"""
        # Check if LLM service is available
        if self.llm_service is None:
            QMessageBox.information(
                self,
                "AI Assistant Not Available",
                "<b>The AI Assistant requires an API key to function.</b><br><br>"
                "To enable this feature:<br>"
                "1. Click <b>'AI Config'</b> in the toolbar<br>"
                "2. Enter your DeepSeek API key<br>"
                "3. Get your API key at: <a href='https://platform.deepseek.com'>https://platform.deepseek.com</a><br><br>"
                "<i>Note: This is an optional feature. You can still use all core features without it.</i>",
                QMessageBox.Ok
            )
            return
        
        dialog = LLMDialog(self, self.llm_service)    
        dialog.exec_()    
    
    def load_cached_algorithms(self):
        """Load cached custom algorithms from storage"""
        try:
            # 🔧 EXE-Compatible: Clean up old temporary algorithm files
            from app.utils.dynamic_loader import get_loader
            loader = get_loader()
            loader.cleanup_old_files(keep_recent=20)  # Keep 20 most recent files
            
            print("Loading cached custom algorithms...")
            
            # Algorithm type mapping
            type_mapping = {
                'preprocessing': ('Preprocessing', self.preprocessing_view),
                'feature_selection': ('Feature Selection', self.feature_selection_view),
                'modeling': ('Modeling', self.modeling_view),
                'data_partitioning': ('Data Partitioning', self.data_partitioning_view)
            }
            
            total_loaded = 0
            errors = []
            
            # Load algorithms for each type
            for cache_type, (display_type, view) in type_mapping.items():
                try:
                    algorithms = self.custom_algorithm_cache.load_algorithms(cache_type)
                    
                    if algorithms:
                        print(f"Loading {len(algorithms)} {display_type} algorithm(s)...")
                        
                        for algo_data in algorithms:
                            try:
                                name = algo_data.get('name', 'Unknown')
                                code = algo_data.get('code', '')
                                
                                if code:
                                    # V1.3.1: For preprocessing, also add to main_window's Custom Tab
                                    if cache_type == 'preprocessing':
                                        print(f"🔧 Adding preprocessing algorithm to Custom Tab: {name}")
                                        self._add_custom_preprocessing_to_ui(code)
                                    # Auto-replace existing algorithms during startup (no dialog)
                                    if cache_type == 'preprocessing':
                                        view.add_custom_algorithm(code, auto_replace=True)
                                    else:
                                        view.add_custom_algorithm(code)
                                    total_loaded += 1
                                    print(f"  Loaded: {name}")
                                else:
                                    errors.append(f"{name}: No code found")
                                    
                            except Exception as e:
                                error_msg = f"{algo_data.get('name', 'Unknown')}: {str(e)}"
                                errors.append(error_msg)
                                print(f"  Failed to load {algo_data.get('name')}: {e}")
                                
                except Exception as e:
                    error_msg = f"Failed to load {display_type} algorithms: {str(e)}"
                    errors.append(error_msg)
                    print(f"Error loading {cache_type}: {e}")
            
            # Show result
            if total_loaded > 0:
                print(f"✅ Successfully loaded {total_loaded} custom algorithm(s)")
                
                # V1.4.0: Update dropdowns after loading all algorithms
                self._update_feature_selection_methods()
                if hasattr(self, 'analysis_type'):
                    current_analysis_type = self.analysis_type.currentText()
                    self.update_analysis_methods(current_analysis_type)
            else:
                print(f"ℹ️  No cached custom algorithms found")
            
            if errors:
                print(f"⚠️  Encountered {len(errors)} error(s) while loading algorithms:")
                for error in errors[:5]:  # Only show first 5 errors
                    print(f"   - {error}")
                
        except Exception as e:
            print(f"❌ Fatal error loading cached algorithms: {e}")
            import traceback
            traceback.print_exc()

    def add_converted_algorithm(self, algorithm_type: str, code: str, task_type: str = None):
        """
        Add a converted algorithm to the appropriate plugin list and save to cache
        Uses EXE-compatible loading method
        
        Args:
            algorithm_type: Algorithm type (Preprocessing/Feature Selection/Modeling/Data Partitioning)
            code: Algorithm code
            task_type: Task type (only for Modeling: 'classification'/'regression'/'both')
        """
        try:
            from app.utils.dynamic_loader import get_loader
            
            # 🔧 EXE-Compatible: Use dynamic loader with fallback
            loader = get_loader()
            
            # Extract algorithm name first for better file naming
            algorithm_name = self._extract_algorithm_name(code)
            
            try:
                # Validate code by attempting to load it (EXE compatible)
                print(f"🔍 Validating algorithm code for '{algorithm_name}'...")
                test_instance = loader.load_from_code(code, algorithm_name)
                print(f"✅ Algorithm validation successful: {test_instance.get_name()}")
            except Exception as e:
                raise Exception(f"Algorithm validation failed: {str(e)}")
            
            # Add to appropriate view (they will use their own loading method)
            if algorithm_type == "Preprocessing":
                self._add_custom_preprocessing_to_ui(code)  # V1.3.1: Add to main_window's Custom Tab
                self.preprocessing_view.add_custom_algorithm(code)  # Also add to view instance (for getting algorithms)    
            elif algorithm_type == "Feature Selection":
                self.feature_selection_view.add_custom_algorithm(code)
                # V1.4.0: Update feature selection dropdown
                self._update_feature_selection_methods()    
            elif algorithm_type == "Modeling":
                # V1.3.2: Pass task type
                print(f"📥 MainWindow: Adding modeling algorithm with task_type='{task_type}'")
                self.modeling_view.add_custom_algorithm(code, task_type=task_type)
                # V1.4.0: Update modeling method dropdown
                current_analysis_type = self.analysis_type.currentText()
                print(f"🔄 MainWindow: Updating analysis methods for '{current_analysis_type}'")
                self.update_analysis_methods(current_analysis_type)    
            elif algorithm_type == "Data Partitioning":
                self.data_partitioning_view.add_custom_algorithm(code)  # V1.3.1: Fix! Use view instead of plugins    
            
            # Map algorithm type to cache key
            cache_type_mapping = {
                'Preprocessing': 'preprocessing',
                'Feature Selection': 'feature_selection',
                'Modeling': 'modeling',
                'Data Partitioning': 'data_partitioning'
            }
            
            cache_type = cache_type_mapping.get(algorithm_type)
            
            if cache_type and algorithm_name:
                # Save to cache
                success = self.custom_algorithm_cache.save_algorithm(
                    cache_type,
                    algorithm_name,
                    code
                )
                
                if success:
                    print(f"Saved custom algorithm '{algorithm_name}' to cache")
                else:
                    print(f"Warning: Failed to save algorithm '{algorithm_name}' to cache")
            
            # V1.3.1: Simplified success message
            QMessageBox.information(self, 'Success', 'Successfully added to system')
            
        except Exception as e:    
            QMessageBox.critical(self, 'Error', f'Error adding algorithm: {str(e)}')
    
    def _extract_algorithm_name(self, code: str) -> str:
        """
        Extract algorithm name from code by executing it and calling get_name()
        V1.3.4: Fixed - Provide necessary imports for code execution
        """
        try:
            import types
            import pandas as pd
            import numpy as np
            from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
            from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
            from interfaces.modeling_algorithm import ModelingAlgorithm
            from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
            
            # Create a temporary module to execute code
            mod = types.ModuleType('temp_module')
            
            # V1.3.6: Provide comprehensive imports for scientific computing
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'PreprocessingAlgorithm': PreprocessingAlgorithm,
                'FeatureSelectionAlgorithm': FeatureSelectionAlgorithm,
                'ModelingAlgorithm': ModelingAlgorithm,
                'DataPartitioningAlgorithm': DataPartitioningAlgorithm,
                'Dict': __import__('typing').Dict,
                'List': __import__('typing').List,
                'Tuple': __import__('typing').Tuple,
                'Any': __import__('typing').Any,
            })
            
            # V1.3.6: Add scipy support (if available)
            try:
                import scipy
                import scipy.signal
                from scipy.signal import savgol_filter
                mod.__dict__.update({
                    'scipy': scipy,
                    'savgol_filter': savgol_filter,
                })
            except ImportError:
                pass
            
            # V1.3.6: Add sklearn support (if available)
            try:
                import sklearn
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                mod.__dict__.update({
                    'sklearn': sklearn,
                    'train_test_split': train_test_split,
                    'LinearRegression': LinearRegression,
                    'StandardScaler': StandardScaler,
                })
            except ImportError:
                pass
            
            exec(code, mod.__dict__)
            
            # Find the class and get its name
            for item in mod.__dict__.values():
                if isinstance(item, type) and hasattr(item, 'get_name'):
                    try:
                        # Try to instantiate and get name
                        instance = item()
                        name = instance.get_name()
                        print(f"✅ Successfully extracted algorithm name: {name}")
                        return name
                    except Exception as e:
                        print(f"⚠️  Failed to instantiate {item.__name__}: {e}")
                        continue
            
            # Fallback: try to extract class name from code
            print("⚠️  Could not instantiate algorithm, extracting class name from code...")
            import re
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                print(f"📝 Extracted class name: {class_name}")
                return class_name
            
            print("❌ Failed to extract any name, using default")
            return "CustomAlgorithm"
            
        except Exception as e:
            print(f"❌ Failed to extract algorithm name: {e}")
            import traceback
            traceback.print_exc()
            return "CustomAlgorithm"    

    def show_llm_config(self):
        """Show the LLM configuration dialog"""
        from app.views.llm_config_dialog import LLMConfigDialog    
        dialog = LLMConfigDialog(self)    
        if dialog.exec_() == QDialog.Accepted:    
            # 🔧 FIX: Pass all configured parameters to LLMService
            self.llm_service = LLMService(
                api_key=LLMConfig.API_KEY,
                base_url=LLMConfig.API_BASE_URL,
                model_name=LLMConfig.MODEL_NAME
            )
    
    def show_algorithm_manager(self):
        """Show the custom algorithm manager dialog"""
        from app.views.custom_algorithm_manager_dialog import CustomAlgorithmManagerDialog
        dialog = CustomAlgorithmManagerDialog(
            cache_service=self.custom_algorithm_cache,
            reload_callback=self.load_cached_algorithms,
            remove_callback=self.remove_algorithm_from_ui,  # V1.4.1: Add remove callback
            parent=self
        )
        dialog.exec_()
    
    def remove_algorithm_from_ui(self, algorithm_type: str, algorithm_name: str):
        """
        V1.4.1: Remove algorithm from UI immediately after deletion
        
        Args:
            algorithm_type: Type of algorithm (preprocessing/feature_selection/modeling/data_partitioning)
            algorithm_name: Name of the algorithm to remove
        """
        try:
            print(f"🗑️  Removing algorithm '{algorithm_name}' from UI (type: {algorithm_type})")
            
            # Map cache type to view
            type_mapping = {
                'preprocessing': self.preprocessing_view,
                'feature_selection': self.feature_selection_view,
                'modeling': self.modeling_view,
                'data_partitioning': self.data_partitioning_view
            }
            
            view = type_mapping.get(algorithm_type)
            if not view:
                print(f"⚠️  Unknown algorithm type: {algorithm_type}")
                return
            
            # Remove from view.plugins
            if hasattr(view, 'plugins') and algorithm_name in view.plugins:
                del view.plugins[algorithm_name]
                print(f"  ✅ Removed from view.plugins")
            
            # Remove from UI checkboxes (if exists)
            if hasattr(view, 'method_checkboxes') and algorithm_name in view.method_checkboxes:
                checkbox = view.method_checkboxes[algorithm_name]
                parent_frame = checkbox.parent()
                if parent_frame:
                    parent_frame.deleteLater()
                del view.method_checkboxes[algorithm_name]
                print(f"  ✅ Removed from UI checkboxes")
            
            # Remove from dropdown menus
            if algorithm_type == 'feature_selection':
                self._update_feature_selection_methods()
                print(f"  ✅ Updated feature selection dropdown")
            elif algorithm_type == 'modeling':
                current_analysis_type = self.analysis_type.currentText() if hasattr(self, 'analysis_type') else None
                if current_analysis_type:
                    self.update_analysis_methods(current_analysis_type)
                    print(f"  ✅ Updated modeling dropdown")
            elif algorithm_type == 'data_partitioning':
                # Refresh data partitioning dropdown
                self._update_data_partitioning_methods()
                print(f"  ✅ Updated data partitioning dropdown")
            elif algorithm_type == 'preprocessing':
                # Preprocessing doesn't use dropdown, but refresh if needed
                if hasattr(view, 'refresh_algorithm_list'):
                    view.refresh_algorithm_list()
                    print(f"  ✅ Refreshed preprocessing algorithm list")
            
            # Remove from task types (for modeling)
            if algorithm_type == 'modeling' and hasattr(view, 'algorithm_task_types'):
                if algorithm_name in view.algorithm_task_types:
                    del view.algorithm_task_types[algorithm_name]
                    print(f"  ✅ Removed from task types")
            
            print(f"✅ Successfully removed '{algorithm_name}' from UI")
            
        except Exception as e:
            print(f"❌ Error removing algorithm from UI: {e}")
            import traceback
            traceback.print_exc()    

    def import_data(self):
        """Load spectral data with spectral type configuration"""
        # Call the data controller's load_data method which includes spectral configuration
        if hasattr(self, 'controller') and self.controller:
            success = self.controller.data_controller.load_data()
            if success:
                # Data loaded successfully, controller handles the rest
                print("Data loaded successfully with spectral configuration")
        else:
            QMessageBox.warning(self, "Error", "Controller not initialized")
    
    def update_ui_after_data_load(self):
        """Update UI components after data is loaded"""
        # This method can be called by the controller if needed
        pass
    

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
                    },
                    # P0 fix: Add wavelength/wavenumber info for spectral-specific algorithms
                    'wavelengths': self.wavelengths.tolist() if hasattr(self, 'wavelengths') and self.wavelengths is not None else None,
                'spectral_type': self.spectral_type if hasattr(self, 'spectral_type') else None
            }
            # Extract enabled methods
            methods = []
            for method_name, params in params_dict.items():
                # Skip non-dict parameters (e.g., wavelengths, spectral_type)
                if isinstance(params, dict) and params.get('enabled', False):
                    methods.append(method_name)
            
            # Add custom algorithms from preprocessing view AND main window Custom Tab
            has_custom_algorithms_available = False
            
            # Check main window Custom Tab checkboxes first
            if hasattr(self, 'custom_preprocessing_checkboxes') and len(self.custom_preprocessing_checkboxes) > 0:
                has_custom_algorithms_available = True
                print(f"🔍 DEBUG: Checking main window Custom Tab: {len(self.custom_preprocessing_checkboxes)} checkboxes")
                for name, checkbox in self.custom_preprocessing_checkboxes.items():
                    if checkbox and checkbox.isChecked():
                        print(f"✅ Found selected algorithm in main window Custom Tab: {name}")
                        if name not in methods:
                            methods.append(name)
                            # Get algorithm from main window storage
                            if hasattr(self, 'custom_preprocessing_algorithms') and name in self.custom_preprocessing_algorithms:
                                algorithm = self.custom_preprocessing_algorithms[name]
                                # Get default parameters
                                params = {}
                                try:
                                    if hasattr(algorithm, 'get_params_info'):
                                        param_info = algorithm.get_params_info()
                                        for k, v in param_info.items():
                                            params[k] = v.get('default', 0)
                                except:
                                    pass
                                
                                if name not in params_dict:
                                    params_dict[name] = {}
                                params_dict[name].update(params)
                                print(f"✅ Added custom algorithm from main window: {name}")
            
            # Also check preprocessing view Custom Tab
            if hasattr(self, 'preprocessing_view') and self.preprocessing_view:
                try:
                    # Check if there are any custom algorithms available (regardless of selection)
                    if hasattr(self.preprocessing_view, 'custom_algorithms') and len(self.preprocessing_view.custom_algorithms) > 0:
                        has_custom_algorithms_available = True
                    
                    # Debug: Check checkbox states before calling get_selected_custom_algorithms
                    if hasattr(self.preprocessing_view, 'custom_algo_checkboxes'):
                        print(f"🔍 DEBUG: Checking preprocessing view Custom Tab: {list(self.preprocessing_view.custom_algo_checkboxes.keys())}")
                        for name, cb in self.preprocessing_view.custom_algo_checkboxes.items():
                            print(f"   - {name}: isChecked={cb.isChecked()}, exists={cb is not None}")
                    
                    custom_algorithms = self.preprocessing_view.get_selected_custom_algorithms()
                    print(f"🔍 Found {len(custom_algorithms)} selected custom algorithms from preprocessing view")
                    for name, algo_info in custom_algorithms.items():
                        if name not in methods:
                            methods.append(name)
                            # Add custom algorithm params to params_dict
                            if name not in params_dict:
                                params_dict[name] = {}
                            # Merge algorithm params (but don't override if already set)
                            algo_params = algo_info.get('params', {})
                            for k, v in algo_params.items():
                                if k not in params_dict[name] or params_dict[name][k] is None or params_dict[name][k] == []:
                                    params_dict[name][k] = v
                            print(f"✅ Added custom algorithm from preprocessing view: {name} with params: {list(params_dict[name].keys())}")
                except Exception as e:
                    print(f"⚠️  Error getting custom algorithms from preprocessing view: {e}")
                    import traceback
                    traceback.print_exc()
            
            if methods:
                print(f"✅ Applied preprocessing: {', '.join(methods)}")
            elif has_custom_algorithms_available:
                # If custom algorithms are available but none selected, show helpful message
                QMessageBox.information(
                    self, 
                    "No Methods Selected", 
                    "Please select at least one preprocessing method from the Custom Tab.\n\n"
                    "Tip: Check the checkbox next to your custom algorithm to enable it."
                )
                self.progress_dialog.close()
                return
            
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
                
                # V1.2.1 critical fix: Verify Original data is not accidentally modified
                import hashlib
                original_hash_before = hashlib.md5(self.original_spectra.tobytes()).hexdigest()
                print(f"🔍 Preprocessing verification:")
                print(f"   Original data hash: {original_hash_before[:16]}... (before preprocessing)")
                
                # V1.2.1 critical fix: Get actual preprocessed data from X_processed
                data_model = self.controller.data_controller.data_model
                
                # Check if preprocessed data exists
                if hasattr(data_model, 'X_processed') and data_model.X_processed is not None:
                    # Correct path: Get preprocessed data from X_processed
                    self.processed_spectra = data_model.X_processed.values
                    print(f"✅ Retrieved preprocessed data: {self.processed_spectra.shape}")
                    print(f"   Data source: data_model.X_processed")
                    
                    # V1.2.1 critical verification: Confirm Original data is not modified
                    original_hash_after = hashlib.md5(self.original_spectra.tobytes()).hexdigest()
                    print(f"🔍 Post-preprocessing verification:")
                    print(f"   Original data hash: {original_hash_after[:16]}... (after preprocessing)")
                    if original_hash_before == original_hash_after:
                        print(f"   ✅ Original data unchanged!")
                    else:
                        print(f"   ❌ Critical error: Original data was modified!")
                        print(f"      Hash before preprocessing: {original_hash_before[:16]}...")
                        print(f"      Hash after preprocessing: {original_hash_after[:16]}...")
                    
                    processed_hash = hashlib.md5(self.processed_spectra.tobytes()).hexdigest()
                    print(f"   Processed data hash: {processed_hash[:16]}...")
                    if original_hash_after != processed_hash:
                        print(f"   ✅ Processed data differs from Original (preprocessing effective)")
                    else:
                        print(f"   ⚠️  Warning: Processed data same as Original (preprocessing may be ineffective)")
                    
                    # V1.2.1: Detailed verification of preprocessing effect
                    X_original = data_model.get_X()
                    if X_original is not None:
                        # Check if data is completely identical
                        is_identical = np.array_equal(self.processed_spectra, X_original.values)
                        
                        if not is_identical:
                            # Calculate difference magnitude
                            diff = np.abs(self.processed_spectra - X_original.values)
                            mean_diff = np.mean(diff)
                            max_diff = np.max(diff)
                            pct_changed = np.sum(diff > 1e-10) / diff.size * 100
                            
                            print(f"   ✅ Data changed:")
                            print(f"      Mean difference: {mean_diff:.4f}")
                            print(f"      Max difference: {max_diff:.4f}")
                            print(f"      Change percentage: {pct_changed:.2f}%")
                            
                            # If difference is small, warn user
                            if mean_diff < 0.01:
                                print(f"      ⚠️  Warning: Preprocessing effect is weak, mean difference only {mean_diff:.6f}")
                                print(f"      Suggestion: Check if preprocessing parameters are appropriate")
                        else:
                            print(f"   ❌ Warning: Preprocessed data completely unchanged!")
                            print(f"   Possible reasons:")
                            print(f"   1. Preprocessing algorithm not executed")
                            print(f"   2. Parameter settings caused no change")
                            print(f"   3. Data flow error")
                else:
                    # If no X_processed, fall back to original data
                    print("⚠️  X_processed not found, using original data")
                    X_original = data_model.get_X()
                    if X_original is not None:
                        self.processed_spectra = X_original.values
                    else:
                        self.processed_spectra = None
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
                            # Update feature data management system
                            self.feature_data_options['preprocessed'] = self.processed_spectra.copy()
                            
                            # If currently using original data, automatically switch to preprocessed data
                            if self.current_feature_source == 'original':
                                self.current_feature_source = 'preprocessed'
                                self.statusBar().showMessage("Switched to using preprocessed data for subsequent analysis.", 5000)
                                print("🔧 Switched to using preprocessed data for modeling")
                            
                            # V1.2.1 critical fix: Refresh current preview mode after preprocessing, instead of forcing processed display
                            # This maintains the user's currently selected view mode (Original/Processed/Comparison)
                            current_mode = self.preview_mode.currentText()
                            print(f"🔧 Preprocessing complete, refreshing current preview mode: {current_mode}")
                            
                            # V1.2.1 fix: Update preview mode button status after successful preprocessing
                            self.update_preview_mode_availability()
                            
                            # Refresh current preview mode display
                            self.preview_preprocessing()
                    except Exception as viz_error:
                        print(f"Visualization update failed: {viz_error}")
                        import traceback
                        traceback.print_exc()
                # Complete
                self.progress_dialog.update_progress(100, "Preprocessing complete")
                self.progress_dialog.close()
                self.statusBar().showMessage("Preprocessing complete")
                if len(methods) > 0:
                    QMessageBox.information(self, "Success", f"Successfully applied {len(methods)} preprocessing method(s): {', '.join(methods)}")
                # Note: Empty methods case is already handled above with custom algorithm check
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
        """V1.4.0: Update available analysis methods based on analysis type (including plugin algorithms)"""
        self.analysis_method.clear()    
        if analysis_type == "Quantitative":    
            # Built-in regression methods
            built_in_methods = [    
                "PLSR",
                "SVR",
                "RF",
                "NN",
                "GPR",
                "XGBoost",
                "LightGBM"
            ]
            self.analysis_method.addItems(built_in_methods)
            
            # V1.4.0: Add custom regression algorithms from plugins
            if hasattr(self, 'modeling_view') and hasattr(self.modeling_view, 'algorithm_task_types'):
                print(f"🔍 Quantitative: Found {len(self.modeling_view.algorithm_task_types)} custom algorithms")
                for method_name, task_type in self.modeling_view.algorithm_task_types.items():
                    print(f"  - {method_name}: task_type='{task_type}'")
                    if task_type in ['regression', 'both']:
                        print(f"    ✅ Adding to Quantitative dropdown")
                        self.analysis_method.addItem(method_name)
                    else:
                        print(f"    ❌ Skipping (not regression)")
            
            self.evaluation_metric.clear()    
            self.evaluation_metric.addItems([    
                "RMSE",
                "MAE",
                "R²",
                "R",
                "RE"
            ])
        else:    
            # Built-in classification methods
            built_in_methods = [    
                "SVM",
                "RF",
                "KNN",
                "DT",
                "NN",
                "XGBoost",
                "LightGBM"
            ]
            self.analysis_method.addItems(built_in_methods)
            
            # V1.4.0: Add custom classification algorithms from plugins
            if hasattr(self, 'modeling_view') and hasattr(self.modeling_view, 'algorithm_task_types'):
                print(f"🔍 Qualitative: Found {len(self.modeling_view.algorithm_task_types)} custom algorithms")
                for method_name, task_type in self.modeling_view.algorithm_task_types.items():
                    print(f"  - {method_name}: task_type='{task_type}'")
                    if task_type in ['classification', 'both']:
                        print(f"    ✅ Adding to Qualitative dropdown")
                        self.analysis_method.addItem(method_name)
                    else:
                        print(f"    ❌ Skipping (not classification)")
            
            self.evaluation_metric.clear()    
            self.evaluation_metric.addItems([    
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "AUC"
            ])
    
    def _update_feature_selection_methods(self):
        """V1.4.0: Update feature selection dropdown with custom algorithms from plugins"""
        if not hasattr(self, 'feature_method'):
            return
        
        # Get current selection to restore it later
        current_selection = self.feature_method.currentText()
        
        # Clear and re-populate with built-in methods
        self.feature_method.clear()
        built_in_methods = [
            "CARS", "SPA", "PLSR", "SelectKBest",
            "Mutual Information", "RFE", "PCA"
        ]
        self.feature_method.addItems(built_in_methods)
        
        # Add custom algorithms from feature_selection_view
        if hasattr(self, 'feature_selection_view') and hasattr(self.feature_selection_view, 'plugins'):
            for method_name in self.feature_selection_view.plugins.keys():
                if method_name not in built_in_methods:
                    self.feature_method.addItem(method_name)
        
        # Restore previous selection if it still exists
        index = self.feature_method.findText(current_selection)
        if index >= 0:
            self.feature_method.setCurrentIndex(index)
    
    def _update_data_partitioning_methods(self):
        """V1.4.4: Update data partitioning dropdown with custom algorithms from plugins"""
        if not hasattr(self, 'data_partitioning_view') or not hasattr(self.data_partitioning_view, 'method_combo'):
            return
        
        # Get current selection to restore it later
        current_selection = self.data_partitioning_view.method_combo.currentText()
        
        # Clear and re-populate with built-in methods
        self.data_partitioning_view.method_combo.clear()
        built_in_methods = ["Train-Test Split", "K-Fold", "LOGO", "Random", "Stratified"]
        self.data_partitioning_view.method_combo.addItems(built_in_methods)
        
        # Add custom algorithms from data_partitioning_view plugins
        if hasattr(self.data_partitioning_view, 'plugins'):
            for method_name in self.data_partitioning_view.plugins.keys():
                if method_name not in built_in_methods:
                    self.data_partitioning_view.method_combo.addItem(method_name)
        
        # Restore previous selection if it still exists
        index = self.data_partitioning_view.method_combo.findText(current_selection)
        if index >= 0:
            self.data_partitioning_view.method_combo.setCurrentIndex(index)
        else:
            # If previous selection was deleted, select first item
            self.data_partitioning_view.method_combo.setCurrentIndex(0)
    
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
        
        # V1.4.0: Remove custom algorithm indicator if present
        # Icon prefix removed - no longer needed
          
        method_mapping = {
            "Neural Network": "nn", 
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Random Forest": "rf"
        }
          
        if method in method_mapping:
            method = method_mapping[method]
        else:
            # V1.4.0: Only convert to lowercase for built-in methods
            # Custom algorithms should keep their original case
            if not any(method in self.modeling_view.plugins for method in [method, method.lower()]):
                method = method.lower()
        try:
              
            self.progress_dialog = ProgressDialog("Analysis", self)    
            self.progress_dialog.show()    
              
            if not hasattr(self, 'controller') or not hasattr(self.controller, 'data_controller'):
                raise ValueError("Data controller not initialized")
            data_model = self.controller.data_controller.data_model
            
            # Use unified data retrieval method - simplify 3 paths to 1 method call
            modeling_data = data_model.get_modeling_data(prefer_selected=True)
            X_train = modeling_data['X_train']
            X_test = modeling_data['X_test']
            y_train = modeling_data['y_train']
            y_test = modeling_data['y_test']
            data_source = modeling_data['source']
            
            self.data_logger.info(f"Modeling data source: {data_source}")
            self.data_logger.info(f"Train shape: {X_train.shape if X_train is not None else 'None'}, "
                                 f"Test shape: {X_test.shape if X_test is not None else 'None'}")
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
            # Check feature count - lower minimum feature requirement to accommodate feature selection
            if X_train.shape[1] < 2:
                error_msg = f"Insufficient features: Only {X_train.shape[1]} features available. At least 2 features are required for modeling."
                self.display_error(error_msg.strip(), "Insufficient Features")
                self.progress_dialog.hide()
                return
            elif X_train.shape[1] < 5:
                print(f"Warning: Only {X_train.shape[1]} features available. Results may be limited with very few features.")
            elif X_train.shape[1] < 10:
                print(f"Notice: {X_train.shape[1]} features available. This is acceptable for modeling, especially after feature selection.")
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
                
                # V1.4.2: Check if this is a custom algorithm first
                is_custom_algorithm = False
                if hasattr(self, 'modeling_view') and hasattr(self.modeling_view, 'plugins'):
                    # Check if method exists in custom algorithms (case-insensitive)
                    method_lower = method.lower()
                    for algo_name in self.modeling_view.plugins.keys():
                        if algo_name.lower() == method_lower:
                            is_custom_algorithm = True
                            print(f"✅ Detected custom algorithm: {algo_name}")
                            break
                
                if is_custom_algorithm:
                    # Use modeling_service for custom algorithms
                    print(f"🔧 Using modeling_service for custom algorithm: {method}")
                    try:
                        # Convert to DataFrame/Series if needed
                        if not isinstance(X_train, pd.DataFrame):
                            X_train = pd.DataFrame(X_train)
                        if not isinstance(y_train, pd.Series):
                            y_train = pd.Series(y_train)
                        
                        # Prepare parameters
                        model_params = {'task_type': 'regression'}
                        
                        # Use modeling_service to train custom algorithm
                        model = self.controller.modeling_controller.modeling_service.train_model(
                            model_name=method,
                            params=model_params,
                            X=X_train,
                            y=y_train
                        )
                        
                        # Create a wrapper analyzer for compatibility
                        class CustomAlgorithmWrapper:
                            def __init__(self, model, algorithm_name):
                                self.model = model
                                self.algorithm_name = algorithm_name
                                self.task_type = 'regression'
                            
                            def predict(self, X):
                                if not isinstance(X, pd.DataFrame):
                                    X = pd.DataFrame(X)
                                # Get algorithm from modeling_view
                                if hasattr(self, '_algorithm'):
                                    return self._algorithm.predict(self.model, X)
                                else:
                                    # Try to get algorithm from modeling_view
                                    if hasattr(self, '_modeling_view') and self._modeling_view:
                                        algo = self._modeling_view.plugins.get(self.algorithm_name)
                                        if algo and hasattr(algo, 'predict'):
                                            return algo.predict(self.model, X)
                                    raise ValueError(f"Cannot find algorithm '{self.algorithm_name}' for prediction")
                            
                            def cross_validate(self, X, y, cv=5):
                                # Simple cross-validation wrapper
                                from sklearn.model_selection import KFold
                                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                
                                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                                rmse_scores = []
                                r2_scores = []
                                mae_scores = []
                                
                                if not isinstance(X, pd.DataFrame):
                                    X = pd.DataFrame(X)
                                if not isinstance(y, pd.Series):
                                    y = pd.Series(y)
                                
                                # Get algorithm reference
                                algo = None
                                if hasattr(self, '_algorithm'):
                                    algo = self._algorithm
                                elif hasattr(self, '_modeling_view') and self._modeling_view:
                                    algo = self._modeling_view.plugins.get(self.algorithm_name)
                                    if not algo:
                                        # Try case-insensitive match
                                        for name, alg in self._modeling_view.plugins.items():
                                            if name.lower() == self.algorithm_name.lower():
                                                algo = alg
                                                break
                                
                                if not algo or not hasattr(algo, 'predict'):
                                    raise ValueError(f"Cannot find algorithm '{self.algorithm_name}' for cross-validation")
                                
                                for train_idx, val_idx in kf.split(X):
                                    X_train_fold = X.iloc[train_idx]
                                    y_train_fold = y.iloc[train_idx]
                                    X_val_fold = X.iloc[val_idx]
                                    y_val_fold = y.iloc[val_idx]
                                    
                                    # Train model on fold
                                    fold_model = self.controller.modeling_controller.modeling_service.train_model(
                                        model_name=self.algorithm_name,
                                        params={'task_type': 'regression'},
                                        X=X_train_fold,
                                        y=y_train_fold
                                    )
                                    
                                    # Predict
                                    y_pred = algo.predict(fold_model, X_val_fold)
                                    
                                    # Calculate metrics
                                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                                    r2 = r2_score(y_val_fold, y_pred)
                                    mae = mean_absolute_error(y_val_fold, y_pred)
                                    
                                    rmse_scores.append(rmse)
                                    r2_scores.append(r2)
                                    mae_scores.append(mae)
                                
                                return {
                                    'rmse': np.mean(rmse_scores),
                                    'r2_score': np.mean(r2_scores),
                                    'mae': np.mean(mae_scores),
                                    'task_type': 'regression',
                                    'cv_folds': cv
                                }
                        
                        # Get algorithm reference for prediction
                        algo = self.modeling_view.plugins.get(method)
                        if not algo:
                            # Try case-insensitive match
                            for name, alg in self.modeling_view.plugins.items():
                                if name.lower() == method.lower():
                                    algo = alg
                                    break
                        
                        wrapper = CustomAlgorithmWrapper(model, method)
                        wrapper._algorithm = algo
                        wrapper._modeling_view = self.modeling_view
                        wrapper.controller = self.controller
                        analyzer = wrapper
                        
                        print(f"✅ Custom algorithm wrapper created for: {method}")
                        
                    except Exception as e:
                        error_msg = f"Error training custom algorithm '{method}': {str(e)}"
                        print(f"❌ {error_msg}")
                        import traceback
                        traceback.print_exc()
                        self.display_error(error_msg)
                        self.progress_dialog.hide()
                        return
                else:
                    # Use QuantitativeAnalyzer for built-in algorithms
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
                    
                    # V1.4.2: Skip fit for custom algorithms (already trained)
                    if not is_custom_algorithm:
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
                if is_custom_algorithm:
                    # For custom algorithms, convert to DataFrame/Series if needed
                    if not isinstance(X_train, pd.DataFrame):
                        X_train_cv = pd.DataFrame(X_train)
                    else:
                        X_train_cv = X_train
                    if not isinstance(y_train_safe, pd.Series):
                        y_train_cv = pd.Series(y_train_safe)
                    else:
                        y_train_cv = y_train_safe
                    results = analyzer.cross_validate(X_train_cv, y_train_cv)
                else:
                    results = analyzer.cross_validate(X_train, y_train_safe)
                # Make predictions
                if X_test is not None and y_test is not None:
                    self.progress_dialog.update_progress(70, "Making predictions...")
                    if is_custom_algorithm:
                        # For custom algorithms, convert to DataFrame if needed
                        if not isinstance(X_test, pd.DataFrame):
                            X_test_pred = pd.DataFrame(X_test)
                        else:
                            X_test_pred = X_test
                        y_pred = analyzer.predict(X_test_pred)
                    else:
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
            # Display results
            self.progress_dialog.update_progress(75, "Generating analysis results...")  # Update progress to 75%
            
            # Debug: Print results dictionary content
            print(f"🔍 DEBUG: results dictionary keys: {list(results.keys())}")
            print(f"🔍 DEBUG: results dictionary content:")
            for k, v in results.items():
                print(f"   {k}: {type(v).__name__} = {v if not isinstance(v, np.ndarray) else f'array shape {v.shape}'}")
            
            msg = "Analysis Results:\n"
            for metric, value in results.items():
                if isinstance(value, float):
                    msg += f"{metric}: {value:.3f}\n"
                elif isinstance(value, np.ndarray):
                    # Format array: each number with 3 decimal places
                    formatted_array = np.array2string(value, precision=3, separator=', ', suppress_small=True, max_line_width=80)
                    msg += f"{metric}: {formatted_array}\n"
                else:
                    msg += f"{metric}: {value}\n"
            
            print(f"🔍 DEBUG: Final msg length: {len(msg)}, content preview: {msg[:200]}")
              
            self.progress_dialog.update_progress(90, "Updating visualization...")
            # V1.2.1 fix: Completely clear before drawing new plot
            if hasattr(self, 'visualization_widget'):
                self.visualization_widget.clear()
                
                if analysis_type.startswith("Quantitative") and X_test is not None and y_test is not None:
                    # Display prediction result scatter plot
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
            
            # Show results first (non-blocking)
            QMessageBox.information(self, "Analysis Results", msg)
            
            # **New: Auto-save ONNX model AFTER showing results**
            # Use longer delay to ensure dialog is closed and UI is responsive
            if 'analyzer' in locals() and hasattr(analyzer, 'model'):
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, lambda: self.auto_save_onnx_model_from_analysis(analyzer.model, X_train, method, results))
        except Exception as e:
            # Enhanced error handling to prevent crashes
            import traceback
            error_traceback = traceback.format_exc()
            print(f"❌ CRITICAL ERROR in start_analysis: {str(e)}")
            print(f"Full traceback:\n{error_traceback}")
            
            # Close progress dialog safely
            try:
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.progress_dialog.close()
            except:
                pass
            
            # Display error message with more details
            error_msg = f"分析失败: {str(e)}\n\n详细错误信息已输出到控制台。"
            try:
                QMessageBox.critical(self, "错误", error_msg)
            except:
                # If QMessageBox also fails, at least print to console
                print(f"ERROR: {error_msg}")
    def generate_report(self):
        """Generate professional spectral analysis report"""
        if self.current_spectra is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        try:
            # Show progress dialog
            self.progress_dialog = ProgressDialog("Generating Professional Report", self)
            self.progress_dialog.show()
            
            # Prepare comprehensive report data
            self.progress_dialog.update_progress(20, "Collecting data information...")
            
            # Enhanced spectra data
            data_model = self.controller.data_controller.data_model
            # Safely get file_path, handling None values
            file_path = getattr(data_model, 'file_path', None)
            if file_path and isinstance(file_path, str):
                # Handle both Unix and Windows path separators
                filename = file_path.replace('\\', '/').split('/')[-1]
                filepath = file_path
            else:
                filename = 'N/A'
                filepath = 'N/A'
            
            spectra_data = {
                'shape': self.current_spectra.shape,
                'wavelength_range': f"{self.wavelengths[0]:.2f} - {self.wavelengths[-1]:.2f} nm",
                'n_samples': self.current_spectra.shape[0],
                'n_features': self.current_spectra.shape[1],
                'spectral_type': getattr(data_model, 'spectral_type', 'Unknown'),
                'missing_values': 0,  # Can add actual check
                'filename': filename,
                'filepath': filepath,
            }
            
            # Add class distribution for classification tasks
            if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                try:
                    if hasattr(data_model, 'task_type') and data_model.task_type == 'classification':
                        y_all = pd.concat([data_model.y_train, data_model.y_test]) if data_model.y_test is not None else data_model.y_train
                        class_counts = y_all.value_counts().to_dict()
                        spectra_data['class_distribution'] = class_counts
                except:
                    pass
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
            # Get feature selection results
            self.progress_dialog.update_progress(40, "Collecting feature selection info...")
            feature_selection_results = {}
            if hasattr(self, 'controller') and hasattr(self.controller, 'feature_selection_controller'):
                fs_model = self.controller.feature_selection_controller.feature_selection_model
                if hasattr(fs_model, 'selected_method') and fs_model.selected_method:
                    feature_selection_results['method'] = fs_model.selected_method
                    if hasattr(fs_model, 'selected_features') and fs_model.selected_features:
                        feature_selection_results['selected_features_count'] = len(fs_model.selected_features)
                        feature_selection_results['selected_features'] = fs_model.selected_features
            
            # Get modeling results
            self.progress_dialog.update_progress(60, "Collecting model performance...")
            model_results = {}
            if hasattr(self, 'controller') and hasattr(self.controller, 'modeling_controller'):
                model_model = self.controller.modeling_controller.modeling_model
                if hasattr(model_model, 'trained_model') and model_model.trained_model:
                    # Model type
                    model_results['model_type'] = type(model_model.trained_model).__name__
                    
                    # Sample counts
                    if hasattr(data_model, 'X_train') and data_model.X_train is not None:
                        model_results['n_train_samples'] = len(data_model.X_train)
                    if hasattr(data_model, 'X_test') and data_model.X_test is not None:
                        model_results['n_test_samples'] = len(data_model.X_test)
                    
                    # Evaluation results
                    if hasattr(model_model, 'evaluation_results') and model_model.evaluation_results:
                        eval_results = model_model.evaluation_results
                        
                        # Extract metrics based on task type
                        task_type = getattr(data_model, 'task_type', 'unknown')
                        
                        if task_type == 'classification':
                            # Classification metrics
                            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                                if f'train_{metric}' in eval_results:
                                    model_results[f'train_{metric}'] = eval_results[f'train_{metric}']
                                if f'test_{metric}' in eval_results:
                                    model_results[f'test_{metric}'] = eval_results[f'test_{metric}']
                        else:
                            # Regression metrics
                            for metric in ['r2', 'rmse', 'mae']:
                                if f'train_{metric}' in eval_results:
                                    model_results[f'train_{metric}'] = eval_results[f'train_{metric}']
                                if f'test_{metric}' in eval_results:
                                    model_results[f'test_{metric}'] = eval_results[f'test_{metric}']
            
            # Generate professional report
            self.progress_dialog.update_progress(80, "Generating professional PDF report...")
            report_generator = ProfessionalSpectrumReportGenerator()
            report_path = report_generator.generate_report(
                spectra_data=spectra_data,
                preprocessing_results=preprocessing_results,
                feature_selection_results=feature_selection_results,
                model_results=model_results,
                title="Spectral_Analysis_Report"
            )
            
            # Complete
            self.progress_dialog.update_progress(100, "Report generation complete")
            self.progress_dialog.close()
            self.statusBar().showMessage(f"Professional report generated: {report_path}")
            
            # Show success with option to open
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText("Professional spectral analysis report generated successfully!")
            msg_box.setInformativeText(f"Report saved to:\n{report_path}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
        except Exception as e:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Report generation failed: {str(e)}")

    def on_preview_mode_changed(self, mode):
        """V1.2.1: Check data availability when preview mode changes"""
        # Update button status
        self.update_preview_mode_availability()
    
    def update_preview_mode_availability(self):
        """V1.2.1: Update preview_mode available options based on data status"""
        if not hasattr(self, 'preview_mode'):
            return
        
        # Check if preprocessed data exists
        has_processed = False
        if hasattr(self, 'controller') and hasattr(self.controller, 'data_controller'):
            data_model = self.controller.data_controller.data_model
            has_processed = (
                hasattr(data_model, 'X_processed') and 
                data_model.X_processed is not None
            )
        
        # Get current option
        current_mode = self.preview_mode.currentText()
        
        # Enable/disable options based on data availability
        model = self.preview_mode.model()
        
        # Original is always available (index 0)
        item = model.item(0)
        item.setEnabled(True)
        
        # Processed and Comparison are only available when preprocessed data exists (indices 1, 2)
        for i in [1, 2]:
            item = model.item(i)
            item.setEnabled(has_processed)
            if not has_processed:
                item.setToolTip("Please apply preprocessing first")
        
        # If currently selected mode is unavailable, automatically switch to Original
        if not has_processed and current_mode in ['Processed', 'Comparison']:
            self.preview_mode.blockSignals(True)
            self.preview_mode.setCurrentText('Original')
            self.preview_mode.blockSignals(False)
            print("⚠️  Preprocessed data unavailable, auto-switching to Original mode")
        
        status_msg = "✓ Preprocessed data available" if has_processed else "✗ Not preprocessed"
        logging.info(f"Preview mode availability: {status_msg}")

    def preview_preprocessing(self):
        """Preview the preprocessing effects on the main visualization widget."""
        if not self.controller.check_data_ready():
            QMessageBox.warning(self, "Warning", "Please load data first")
            return

        try:
            preview_mode = self.preview_mode.currentText()
            print(f"📺 Refresh preview: mode={preview_mode}")
            
            # V1.2.1 fix: Check data availability
            data_model = self.controller.data_controller.data_model
            has_processed = (
                hasattr(data_model, 'X_processed') and 
                data_model.X_processed is not None
            )
            
            # If Processed/Comparison is selected but no preprocessed data, show warning
            if preview_mode in ['Processed', 'Comparison'] and not has_processed:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "预处理数据不可用\n\n请先应用预处理方法，然后才能查看预处理效果。"
                )
                return
            
            methods_applied = []
            if hasattr(self.controller.preprocessing_controller, 'preprocessing_model') and \
               hasattr(self.controller.preprocessing_controller.preprocessing_model, 'applied_methods'):
                methods_applied = list(self.controller.preprocessing_controller.preprocessing_model.applied_methods.keys())

            # The visualization widget now handles all preview logic based on the mode.
            self.visualization_widget.update_preprocessing_preview(
                data_model=self.controller.data_controller.data_model,
                wavelengths=self.wavelengths,
                preview_mode=preview_mode,
                methods_applied=methods_applied
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate preprocessing preview: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
          
        # V1.2.1 fix: Use unified update_preprocessing_preview
        if hasattr(self, 'original_spectra') and self.original_spectra is not None:    
            self.current_spectra = self.original_spectra.copy()
            self.visualization_widget.update_preprocessing_preview(
                data_model=self.controller.data_controller.data_model,
                wavelengths=self.wavelengths,
                preview_mode='Original',
                methods_applied=[]
            )
            print(f"✅ Reset complete, display mode: Original")
        else:
            print(f"⚠️  Warning: No original data to reset")    

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
<h1><b>Despiking</b></h1>

<p>Despiking is primarily used to eliminate cosmic ray spikes and instrumental artifacts from Raman spectra. These spikes typically appear as abnormally high intensity values at single or few data points, which can severely affect subsequent spectral analysis.</p>

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
        tabs.addTab(spike_tab, "Despiking")
        
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

<p>In practical applications, Savitzky-Golay filtering is recommended as the first choice, starting with small windows (9-11 points) and low orders (2-3). For spectra with particularly high noise, try Gaussian smoothing. For situations requiring despiking, consider median filtering. After smoothing, always check whether peaks are excessively broadened and whether signals are distorted.</p>
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

    def _on_feature_btn_clicked(self):
        """
        Handles the 'Apply' button click in the 'Feature Selection' group.
        It intelligently delegates to the correct workflow (exploratory PCA, modeling PCA, or other feature selection).
        """
        method = self.feature_method.currentText()
        
        # V1.4.0: Remove custom algorithm indicator if present
        # Icon prefix removed - no longer needed
        
        params = {'n_components': self.feature_param_spin.value()}
        
        print(f"Feature selection requested for method: {method} with params: {params}")

        if "PCA" in method:
            # Delegate the entire PCA workflow to the service.
            # The service will handle checking for partitioned data and launching the correct UI.
            try:
                self.feature_extraction_service.run_feature_extraction(method, params)
            except Exception as e:
                error_msg = f"PCA workflow failed: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "PCA Error", error_msg)
        else:
            # For all other methods, use the supervised feature selection logic.
            print(f"Redirecting to feature selection for method: {method}")
            self.apply_feature_selection()

    def apply_feature_selection_from_quick_panel(self):
        """Unified feature selection processing method - intelligent pipeline selects processing path based on algorithm type"""
        try:
            method = self.feature_method.currentText()
            
            # V1.4.0: Remove custom algorithm indicator if present
            if method.startswith("🔧 "):
                method = method.replace("🔧 ", "", 1)  # Remove "🔧 " prefix
            
            param = self.feature_param_spin.value()
            
            print(f"🔧 Applying feature selection: {method} with parameter: {param}")
            
            # Intelligent pipeline: Select different processing paths based on algorithm type
            if "PCA" in method:
                # Check if data is partitioned
                data_model = self.controller.data_controller.data_model
                is_partitioned = hasattr(data_model, 'X_train') and data_model.X_train is not None

                if not is_partitioned:
                    # --- Exploratory Analysis Path ---
                    print("🔬 Data is not partitioned. Running PCA in exploratory mode.")
                    self._run_exploratory_pca(param)
                else:
                    # --- Modeling Workflow Path ---
                    print("🎯 Data is partitioned. Running PCA in modeling mode (fit on train, transform all).")
                    self._run_modeling_pca(param)

            elif method in ["SelectKBest", "Recursive Feature Elimination (RFE)", "LASSO", "Feature Importance", "Mutual Information"]:
                # Supervised feature selection path - requires data partitioning
                self._apply_supervised_feature_selection(method, param)
            else:
                QMessageBox.warning(self, "Warning", f"Unknown method: {method}")
                
        except Exception as e:
            error_msg = f"Feature selection failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Feature Selection Error", error_msg)

    def _run_exploratory_pca(self, n_components):
        """Run exploratory PCA for unpartitioned data and display specialized dialog."""
        if self.current_spectra is None:
            QMessageBox.warning(self, "Warning", "Please load spectral data first")
            return

        try:
            from app.services.feature_extraction_service import feature_extraction_service
            import pandas as pd

            # Use current complete spectral data
            # Ensure we're processing data without labels
            data_to_process = self.get_current_feature_data()
            if isinstance(data_to_process, pd.DataFrame):
                 X = data_to_process.copy()
            else:
                 X = pd.DataFrame(data_to_process)

            print(f"🔬 Calling PCA service in exploratory mode with data shape: {X.shape}")

            # Call service with exploratory_mode=True
            # This service will call unsupervised_pca plugin, which internally creates and displays PCAExplorerDialog
            feature_extraction_service.extract_pca_features(
                X, n_components=n_components, exploratory_mode=True, wavelengths=self.wavelengths
            )
            
        except Exception as e:
            error_msg = f"Exploratory PCA failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Exploratory PCA Error", error_msg)

    def _run_modeling_pca(self, n_components):
        """Run PCA for partitioned data to prevent data leakage and provide visualization."""
        data_model = self.controller.data_controller.data_model
        if not (hasattr(data_model, 'X_train') and data_model.X_train is not None):
            QMessageBox.critical(self, "Error", "Data not partitioned. Cannot run PCA in modeling mode.")
            return

        try:
            from sklearn.decomposition import PCA
            import pandas as pd

            X_train = data_model.X_train
            X_test = data_model.X_test
            
            print(f"🎯 Fitting PCA on training data (shape: {X_train.shape}) with n_components={n_components}")

            # Check if requested number of components is greater than sample count or feature count
            max_components = min(X_train.shape)
            if n_components > max_components:
                print(f"⚠️ Requested components ({n_components}) is greater than max possible ({max_components}). Adjusting to {max_components}.")
                n_components = max_components
                self.feature_param_spin.setValue(n_components)


            pca = PCA(n_components=n_components)
            
            # Fit on training data ONLY
            pca.fit(X_train)
            
            # Transform both train and test data
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test) if X_test is not None else None
            
            print(f"   Transformed X_train shape: {X_train_pca.shape}")
            if X_test_pca is not None:
                print(f"   Transformed X_test shape: {X_test_pca.shape}")

            # Update data model for subsequent steps
            data_model.X_train_selected = X_train_pca
            data_model.X_test_selected = X_test_pca
            
            # Update feature management system
            self.feature_data_options['selected_features'] = {
                'X_train': X_train_pca,
                'X_test': X_test_pca,
                'selected_features': [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]
            }
            self.current_feature_source = 'selected_features'

            # Notify user
            explained_variance = sum(pca.explained_variance_ratio_)
            self.statusBar().showMessage(f"PCA for modeling complete. Now using {pca.n_components_} PCA features.", 5000)

            # Display results using PCAExplorerDialog
            dialog = PCAExplorerDialog(parent=self)
            dialog.set_data(
                pca_instance=pca,
                loadings=pca.components_,
                explained_variance_ratio=pca.explained_variance_ratio_,
                cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
                X_scores=X_train_pca,
                wavelengths=self.wavelengths,
                y=data_model.y_train, # Pass labels for coloring score plot
                is_modeling_mode=True
            )
            dialog.show() 
            if not hasattr(self, 'pca_dialogs'):
                self.pca_dialogs = []
            self.pca_dialogs.append(dialog)

        except Exception as e:
            error_msg = f"Modeling PCA failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Modeling PCA Error", error_msg)
    
    def _apply_unsupervised_feature_extraction(self, method, param):
        """Unsupervised feature extraction (e.g., PCA) - does not require labels and data partitioning"""
        print(f"🔬 Applying unsupervised feature extraction: {method}")
        
        if self.current_spectra is None:
            QMessageBox.warning(self, "Warning", "Please load spectral data first")
            return
        
        # Use current spectral data for unsupervised feature extraction
        from app.services.feature_extraction_service import feature_extraction_service
        
        # Convert numpy array to DataFrame for processing
        import pandas as pd
        if isinstance(self.current_spectra, np.ndarray):
            X = pd.DataFrame(self.current_spectra)
        else:
            X = self.current_spectra
        
        if "PCA" in method:  # "Principal Component Analysis (PCA)"
            results = feature_extraction_service.extract_pca_features(
                X, n_components=param, cev_threshold=0.95
            )
            
            # Store PCA results
            self.feature_data_options['pca_features'] = results['transformed_data'].values
            
            # Ask user if they want to use PCA features
            reply = QMessageBox.question(
                self, "Use PCA Features",
                f"PCA completed! Extracted {results['n_components']} components.\n"
                f"Explained variance: {results['cumulative_explained_variance'][-1]:.1%}\n\n"
                f"Would you like to use these PCA features for subsequent modeling?\n\n"
                f"• Yes: Use {results['n_components']} PCA components for modeling\n"
                f"• No: Keep original data for modeling",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.current_feature_source = 'pca_features'
                QMessageBox.information(
                    self, "PCA Features Activated",
                    f"Now using {results['n_components']} PCA components for modeling.\n"
                    f"You can proceed with data partitioning and modeling."
                )
            else:
                QMessageBox.information(
                    self, "PCA Features Available",
                    f"PCA features are available but original data will be used for modeling.\n"
                    f"You can switch to PCA features later if needed."
                )
    
    def _apply_supervised_feature_selection(self, method, param):
        """Supervised feature selection - requires labels and data partitioning"""
        print(f"🎯 Applying supervised feature selection: {method}")
        
        # Special handling: If PCA, use correct modeling path
        if "PCA" in method:
            print("🔄 PCA detected in supervised path, redirecting to correct modeling PCA path...")
            self._run_modeling_pca(param)
            return
        
        # Check if data partitioning has been performed
        if not hasattr(self.controller, 'data_controller') or not hasattr(self.controller.data_controller, 'data_model'):
            QMessageBox.warning(self, "Warning", "Please perform data partitioning first")
            return
        
        data_model = self.controller.data_controller.data_model
        if not hasattr(data_model, 'X_train') or data_model.X_train is None:
            QMessageBox.warning(self, "Warning", "Please perform data partitioning first")
            return
        
        # Call original feature selection logic
        self.apply_feature_selection()
    
    def _apply_specialized_feature_method(self, method, param):
        """Specialized feature engineering methods"""
        print(f"🔧 Applying specialized feature method: {method}")
        
        if method == "PLSR":
            # PLSR requires labels, use supervised path
            self._apply_supervised_feature_selection("PLSR", param)
        elif method in ["Wavelet", "Peak Detection"]:
            # These should be implemented in preprocessing, temporarily inform user
            QMessageBox.information(
                self, "Method Note",
                f"{method} is typically used in preprocessing step.\n"
                f"Please check the preprocessing options for {method} functionality."
            )
        else:
            QMessageBox.warning(self, "Warning", f"Method {method} not yet implemented")
    
    def apply_feature_selection(self):
        """Apply feature selection with intelligent recommendations"""
        try:
            # Get current parameters
            method = self.feature_method.currentText()
            
            # V1.4.0: Remove custom algorithm indicator if present
            if method.startswith("🔧 "):
                method = method.replace("🔧 ", "", 1)  # Remove "🔧 " prefix
            
            param = self.feature_param_spin.value()
            
            # Special handling: If user directly calls this method to select PCA, redirect to unsupervised path
            if "PCA" in method:  # "Principal Component Analysis (PCA)"
                print("🔄 PCA detected, redirecting to unsupervised extraction...")
                self._apply_unsupervised_feature_extraction(method, param)
                return
            
            recommendation_msg = "" # Initialize empty message
            # **CRITICAL: Intelligent feature count recommendation**
            if hasattr(self.controller, 'data_controller') and hasattr(self.controller.data_controller, 'data_model'):
                data_model = self.controller.data_controller.data_model
                if hasattr(data_model, 'y_train') and data_model.y_train is not None:
                    y_train = data_model.y_train
                    n_classes = len(np.unique(y_train))
                    # Calculate recommended features
                    min_features_needed = max(n_classes * 3, 30)
                    if param < min_features_needed:
                        # Instead of a popup, prepare a message for the results window.
                        recommendation_msg = (f"<b>Warning:</b> The current setting is <b>{param}</b> features, "
                                              f"but for <b>{n_classes}</b> classes, at least <b>{min_features_needed}</b> are recommended for better accuracy. "
                                              f"The analysis was run with {param} features.")
                        print(f"Feature selection recommendation: {recommendation_msg}")
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
            
            # Critical fix: Feature selection always starts from complete original data
            self.data_logger.debug("Preparing feature selection data...")
            self.data_logger.debug(f"Current feature source: {getattr(self, 'current_feature_source', 'None')}")
            
            # Prefer preprocessed complete data, fall back to original complete data
            if (hasattr(data_model, 'X_processed') and data_model.X_processed is not None and
                hasattr(data_model, 'X_train') and data_model.X_train is not None):
                
                # Use preprocessed complete data for partitioning
                self.data_logger.info("Using preprocessed full data for feature selection")
                self.data_logger.debug(f"X_processed shape: {data_model.X_processed.shape}")
                
                # Re-partition into training and test sets
                X_train_rows = data_model.X_train.shape[0]
                X_test_rows = data_model.X_test.shape[0] if data_model.X_test is not None else 0
                
                X_train = data_model.X_processed.iloc[:X_train_rows]
                if X_test_rows > 0:
                    X_test = data_model.X_processed.iloc[X_train_rows:X_train_rows + X_test_rows]
                else:
                    X_test = None
                    
                self.data_logger.debug(f"Preprocessed X_train shape: {X_train.shape}")
                self.data_logger.debug(f"Preprocessed X_test shape: {X_test.shape if X_test is not None else 'None'}")
                
            else:
                # Use original complete data (not preprocessed)
                self.data_logger.warning("Using original data (not preprocessed)")
                self.data_logger.info("Tip: Apply preprocessing before feature selection for better results")
                X_train = data_model.X_train.copy()  # Copy to avoid modifying original data
                X_test = data_model.X_test.copy() if data_model.X_test is not None else None
                
            self.data_logger.info(f"Feature selection input: X_train shape={X_train.shape}, features={X_train.shape[1]}")
                
            y_train = data_model.y_train
            # Apply feature selection using the controller's apply_method
            result = self.controller.feature_selection_controller.apply_method(
                X_train, y_train, X_test, service_method, params
            )
            # Update data model with results
            data_model.X_train_selected = result['X_train_selected']
            if X_test is not None:
                data_model.X_test_selected = result['X_test_selected']
            
            # Critical fix: Automatically use selected features after feature selection
            self.feature_data_options['selected_features'] = {
                'X_train': result['X_train_selected'],
                'X_test': result['X_test_selected'] if X_test is not None else None,
                'selected_features': result['selected_features']
            }
            self.current_feature_source = 'selected_features'
            
            # Critical fix: Do not modify original wavelength array, save selected feature wavelength info separately
            # self.wavelengths should always remain complete for visualization
            # Selected feature wavelengths are saved separately to self.selected_wavelengths
            original_wavelengths_for_viz = self.wavelengths.copy() if self.wavelengths is not None else None
            
            if 'selected_features' in result and self.wavelengths is not None and len(self.wavelengths) > 0:
                try:
                    # Get original column names
                    original_columns = X_train.columns.tolist()
                    selected_columns = result['selected_features']
                    
                    # Calculate indices of selected features
                    selected_indices = []
                    for col in selected_columns:
                        if col in original_columns:
                            selected_indices.append(original_columns.index(col))
                    
                    if len(selected_indices) > 0:
                        # Save wavelengths corresponding to selected features to separate variable
                        self.selected_wavelengths = self.wavelengths[selected_indices]
                        
                        self.data_logger.info(f"Feature selection completed: {len(selected_columns)} features selected")
                        self.data_logger.debug(f"Original wavelengths: {len(self.wavelengths)}, "
                                              f"Selected wavelengths: {len(self.selected_wavelengths)}")
                        self.data_logger.debug(f"Selected wavelength range: "
                                              f"{self.selected_wavelengths.min():.1f} - {self.selected_wavelengths.max():.1f}")
                    else:
                        self.data_logger.warning("Cannot extract selected wavelengths: no matching indices found")
                        self.selected_wavelengths = None
                except Exception as e:
                    self.data_logger.error(f"Error extracting selected wavelengths: {e}")
                    self.selected_wavelengths = None

            self.statusBar().showMessage(f"Feature selection complete: {len(result['selected_features'])} features selected. System will now use these for modeling.")
            
            self.data_logger.info(f"Feature selection activated: {len(result['selected_features'])}/{X_train.shape[1]} features")
            self.data_logger.info("System switched to using selected features for modeling")
            # Auto-popup feature importance visualization window
            if 'feature_importance' in result and result['feature_importance'] is not None:
                from app.utils.visualization_window import VisualizationWindow
                # **CRITICAL FIX: Keep window reference to prevent garbage collection**
                if not hasattr(self, 'feature_vis_windows'):
                    self.feature_vis_windows = []
                
                vis_window = VisualizationWindow(title="Feature Selection Result")
                
                # Pass the recommendation message to the window
                vis_window.set_message(recommendation_msg)
                
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
            
            # Display selected feature wavelength visualization dialog
            # NOTE: Disabled automatic display - user can manually open if needed
            # Uncomment the code below if you want to enable automatic display
            """
            try:
                if original_wavelengths_for_viz is not None and len(original_wavelengths_for_viz) > 0:
                    from app.views.feature_wavelength_visualization_dialog import FeatureWavelengthVisualizationDialog
                    
                    print(f"📊 Preparing to display feature wavelength visualization dialog...")
                    print(f"   Wavelength array length: {len(original_wavelengths_for_viz)}")
                    print(f"   Selected features count: {len(result['selected_features'])}")
                    
                    # Create and display feature wavelength visualization dialog
                    wavelength_viz_dialog = FeatureWavelengthVisualizationDialog(
                        self.controller.data_controller.data_model,
                        original_wavelengths_for_viz,
                        result['selected_features'],
                        parent=self
                    )
                    
                    # Use non-modal display, but keep reference
                    wavelength_viz_dialog.setModal(False)
                    wavelength_viz_dialog.show()
                    wavelength_viz_dialog.raise_()  # Ensure window is on top
                    wavelength_viz_dialog.activateWindow()  # Activate window
                    
                    # Save window reference to prevent garbage collection
                    if not hasattr(self, 'wavelength_viz_windows'):
                        self.wavelength_viz_windows = []
                    self.wavelength_viz_windows.append(wavelength_viz_dialog)
                    
                    # Keep only the most recent 5 windows
                    if len(self.wavelength_viz_windows) > 5:
                        old_window = self.wavelength_viz_windows.pop(0)
                        try:
                            old_window.close()
                        except:
                            pass
                    
                    print(f"✅ Feature wavelength visualization opened, showing {len(result['selected_features'])} selected bands")
                else:
                    print("⚠️  Cannot display feature wavelength visualization: wavelength info unavailable")
            except Exception as e:
                print(f"❌ Error displaying feature wavelength visualization: {e}")
                import traceback
                traceback.print_exc()
                # Does not affect main flow, continue execution
            """
        except Exception as e:
            self.display_error(f"Feature selection failed: {str(e)}")

    def partition_data_with_params(self):
        """Apply data partitioning with parameters from the data partitioning view"""
          
        if not self.controller.check_data_ready():
            self.display_message("Please load data first", "Warning")
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
            self.display_error(f"Data partitioning failed: {str(e)}")
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
    
    def update_spectral_config(self, config):
        """
        Fix: Receive and apply spectral configuration (resolves reviewer Y-axis issue)
        
        This method is called by data_controller after data import,
        to sync spectral type to main window and visualization component
        """
        if config is None:
            return
        
        # Save configuration
        self.spectral_config = config
        self.spectral_type = config.get('spectral_type', None)
        
        # Sync to visualization component
        if hasattr(self, 'visualization_widget'):
            self.visualization_widget.spectral_type = self.spectral_type
            self.visualization_widget.spectral_config = config
            
            # If data already exists, immediately refresh visualization to apply new Y-axis label
            if hasattr(self, 'wavelengths') and self.wavelengths is not None and \
               hasattr(self, 'current_spectra') and self.current_spectra is not None:
                self.visualization_widget.plot_spectra(
                    self.wavelengths,
                    self.current_spectra,
                    title="Spectral Data",
                    labels=self.sample_labels if hasattr(self, 'sample_labels') else None
                )


