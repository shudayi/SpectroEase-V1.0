# app/views/preprocessing_view.py

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QHBoxLayout, QFrame, QCheckBox, QGroupBox, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
try:
    import pyqtgraph as pg
except ImportError:
    pg = None

class PreprocessingView(QWidget):
    def __init__(self, plugins=None):
        super().__init__()
        self.plugins = plugins if plugins else {}
        self.init_ui()
        
    def init_ui(self):
        """初始化预processingUI界面"""
  
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
  
        app_font = QFont("Microsoft YaHei UI", 9)
        self.setFont(app_font)
        
  
        tabs = QTabWidget()
        tabs.setDocumentMode(True)  # 更现代的外观
        
  
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(5)
        basic_layout.setContentsMargins(5, 5, 5, 5)
        
        # 新增：去峰处理组（Raman专用，默认关闭）
        despiking_group = QGroupBox("Spike Removal (Raman推荐)")
        despiking_layout = QVBoxLayout()
        despiking_layout.setContentsMargins(5, 8, 5, 5)
        despiking_layout.setSpacing(3)
        
        self.despiking_check = QCheckBox("Enable Spike Removal")
        self.despiking_check.setChecked(False)  # 默认关闭
        despiking_layout.addWidget(self.despiking_check)
        
        # 去峰参数
        despiking_params_layout = QHBoxLayout()
        despiking_params_layout.setSpacing(8)
        
        despiking_method_label = QLabel("Method:")
        despiking_params_layout.addWidget(despiking_method_label)
        
        self.despiking_method = QComboBox()
        self.despiking_method.addItems(["mad", "localz"])
        despiking_params_layout.addWidget(self.despiking_method)
        
        despiking_window_label = QLabel("Window:")
        despiking_params_layout.addWidget(despiking_window_label)
        
        self.despiking_window = QSpinBox()
        self.despiking_window.setRange(7, 15)
        self.despiking_window.setValue(11)
        self.despiking_window.setSingleStep(2)  # 确保奇数
        despiking_params_layout.addWidget(self.despiking_window)
        
        despiking_threshold_label = QLabel("Threshold:")
        despiking_params_layout.addWidget(despiking_threshold_label)
        
        self.despiking_threshold = QDoubleSpinBox()
        self.despiking_threshold.setRange(3.0, 6.0)
        self.despiking_threshold.setValue(5.0)
        self.despiking_threshold.setSingleStep(0.5)
        despiking_params_layout.addWidget(self.despiking_threshold)
        
        despiking_layout.addLayout(despiking_params_layout)
        despiking_group.setLayout(despiking_layout)
        basic_layout.addWidget(despiking_group)
        
  
        baseline_group = QGroupBox("Baseline Correction")
        baseline_layout = QVBoxLayout()
        baseline_layout.setContentsMargins(5, 8, 5, 5)
        baseline_layout.setSpacing(3)
        
        self.baseline_check = QCheckBox("Enable Baseline Correction")
        baseline_layout.addWidget(self.baseline_check)
        
        # 基线校正方法选择
        baseline_method_layout = QHBoxLayout()
        baseline_method_layout.setSpacing(8)
        
        baseline_method_label = QLabel("Method:")
        baseline_method_layout.addWidget(baseline_method_label)
        
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["Polynomial", "ALS", "airPLS"])
        baseline_method_layout.addWidget(self.baseline_method)
        
        baseline_layout.addLayout(baseline_method_layout)
        
        baseline_params = QHBoxLayout()
        baseline_params.setSpacing(3)
        
        poly_label = QLabel("Polynomial Order:")
        baseline_params.addWidget(poly_label)
        
        self.poly_order = QSpinBox()
        self.poly_order.setRange(1, 10)
        self.poly_order.setValue(3)
        baseline_params.addWidget(self.poly_order)
        
        baseline_layout.addLayout(baseline_params)
        baseline_group.setLayout(baseline_layout)
        basic_layout.addWidget(baseline_group)
        
  
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QVBoxLayout()
        smoothing_layout.setContentsMargins(5, 20, 5, 5)
        smoothing_layout.setSpacing(30)
        
        self.smoothing_check = QCheckBox("Enable Smoothing")
        smoothing_layout.addWidget(self.smoothing_check)
        
  
        method_params = QHBoxLayout()
        method_params.setSpacing(15)
        
        method_label = QLabel("Method:")
        method_params.addWidget(method_label)
        
        self.smoothing_method = QComboBox()
        self.smoothing_method.addItems(["Savitzky-Golay", "Moving Average", "Gaussian", "Median"])
        method_params.addWidget(self.smoothing_method)
        
        smoothing_layout.addLayout(method_params)
        
  
        window_params = QHBoxLayout()
        window_params.setSpacing(15)
        
        window_label = QLabel("Window Size:")
        window_params.addWidget(window_label)
        
        self.window_size = QSpinBox()
        self.window_size.setRange(3, 51)
        self.window_size.setValue(15)
        self.window_size.setSingleStep(2)  # 确保是奇数
        window_params.addWidget(self.window_size)
        
  
        poly_sgf_label = QLabel("Poly Order:")
        window_params.addWidget(poly_sgf_label)
        
        self.poly_sgf_order = QSpinBox()
        self.poly_sgf_order.setRange(1, 5)
        self.poly_sgf_order.setValue(2)
        window_params.addWidget(self.poly_sgf_order)
        
        smoothing_layout.addLayout(window_params)
        smoothing_layout.addSpacing(10)
        smoothing_group.setLayout(smoothing_layout)
        basic_layout.addWidget(smoothing_group)
        
  
        scatter_group = QGroupBox("Scatter Correction")
        scatter_layout = QVBoxLayout()
        scatter_layout.setContentsMargins(5, 8, 5, 5)
        scatter_layout.setSpacing(3)
        
        self.scatter_check = QCheckBox("Enable Scatter Correction")
        scatter_layout.addWidget(self.scatter_check)
        
  
        scatter_method_params = QHBoxLayout()
        scatter_method_params.setSpacing(3)
        
        scatter_method_label = QLabel("Method:")
        scatter_method_params.addWidget(scatter_method_label)
        
        self.scatter_method = QComboBox()
        self.scatter_method.addItems(["MSC", "SNV", "EMSC", "RNV", "OSC"])
        scatter_method_params.addWidget(self.scatter_method)
        
        scatter_layout.addLayout(scatter_method_params)
        scatter_group.setLayout(scatter_layout)
        basic_layout.addWidget(scatter_group)
        
        basic_tab.setLayout(basic_layout)
        tabs.addTab(basic_tab, "Basic")
        
  
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(5)
        advanced_layout.setContentsMargins(5, 5, 5, 5)
        
  
        normalization_group = QGroupBox("Normalization")
        normalization_layout = QVBoxLayout()
        normalization_layout.setContentsMargins(5, 8, 5, 5)
        normalization_layout.setSpacing(3)
        
        self.normalization_check = QCheckBox("Enable Normalization")
        normalization_layout.addWidget(self.normalization_check)
        
  
        norm_method_params = QHBoxLayout()
        norm_method_params.setSpacing(3)
        
        norm_method_label = QLabel("Method:")
        norm_method_params.addWidget(norm_method_label)
        
        self.normalization_method = QComboBox()
        self.normalization_method.addItems(["Min-Max", "Vector", "Area", "Maximum"])
        norm_method_params.addWidget(self.normalization_method)
        
        normalization_layout.addLayout(norm_method_params)
        normalization_group.setLayout(normalization_layout)
        advanced_layout.addWidget(normalization_group)
        
  
        standardization_group = QGroupBox("Standardization")
        standardization_layout = QVBoxLayout()
        standardization_layout.setContentsMargins(5, 8, 5, 5)
        standardization_layout.setSpacing(3)
        
        self.standardization_check = QCheckBox("Enable Standardization")
        standardization_layout.addWidget(self.standardization_check)
        
  
        std_method_params = QHBoxLayout()
        std_method_params.setSpacing(3)
        
        std_method_label = QLabel("Method:")
        std_method_params.addWidget(std_method_label)
        
        self.standardization_method = QComboBox()
        self.standardization_method.addItems(["Z-Score", "Robust", "Min-Max"])
        std_method_params.addWidget(self.standardization_method)
        
        standardization_layout.addLayout(std_method_params)
        standardization_group.setLayout(standardization_layout)
        advanced_layout.addWidget(standardization_group)
        
  
        derivative_group = QGroupBox("Derivative")
        derivative_layout = QVBoxLayout()
        derivative_layout.setContentsMargins(5, 8, 5, 5)
        derivative_layout.setSpacing(3)
        
        self.derivative_check = QCheckBox("Enable Derivative")
        derivative_layout.addWidget(self.derivative_check)
        
  
        derivative_params = QHBoxLayout()
        derivative_params.setSpacing(3)
        
        derivative_order_label = QLabel("Order:")
        derivative_params.addWidget(derivative_order_label)
        
        self.derivative_order = QSpinBox()
        self.derivative_order.setRange(1, 3)
        self.derivative_order.setValue(1)
        derivative_params.addWidget(self.derivative_order)
        
  
        derivative_method_label = QLabel("Method:")
        derivative_params.addWidget(derivative_method_label)
        
        self.derivative_method = QComboBox()
        self.derivative_method.addItems(["Savitzky-Golay", "Finite Diff", "Gap-Segment"])
        derivative_params.addWidget(self.derivative_method)
        
        derivative_layout.addLayout(derivative_params)
        derivative_group.setLayout(derivative_layout)
        advanced_layout.addWidget(derivative_group)
        
        advanced_tab.setLayout(advanced_layout)
        tabs.addTab(advanced_tab, "Advanced")
        
  
        special_tab = QWidget()
        special_layout = QVBoxLayout()
        special_layout.setSpacing(5)
        special_layout.setContentsMargins(5, 5, 5, 5)
        
  
        outlier_group = QGroupBox("Outlier Detection")
        outlier_layout = QVBoxLayout()
        outlier_layout.setContentsMargins(5, 8, 5, 5)
        outlier_layout.setSpacing(3)
        
        self.outlier_check = QCheckBox("Enable Outlier Detection")
        outlier_layout.addWidget(self.outlier_check)
        
  
        outlier_method_params = QHBoxLayout()
        outlier_method_params.setSpacing(3)
        
        outlier_method_label = QLabel("Method:")
        outlier_method_params.addWidget(outlier_method_label)
        
        self.outlier_method = QComboBox()
        self.outlier_method.addItems(["IQR", "Z-Score", "ISO", "LOF"])
        outlier_method_params.addWidget(self.outlier_method)
        
        outlier_layout.addLayout(outlier_method_params)
        
  
        outlier_threshold_params = QHBoxLayout()
        outlier_threshold_params.setSpacing(3)
        
        outlier_threshold_label = QLabel("Threshold:")
        outlier_threshold_params.addWidget(outlier_threshold_label)
        
        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(1.0, 5.0)
        self.outlier_threshold.setValue(3.0)
        self.outlier_threshold.setSingleStep(0.1)
        outlier_threshold_params.addWidget(self.outlier_threshold)
        
        outlier_layout.addLayout(outlier_threshold_params)
        outlier_group.setLayout(outlier_layout)
        special_layout.addWidget(outlier_group)
        
  
        denoising_group = QGroupBox("Denoising")
        denoising_layout = QVBoxLayout()
        denoising_layout.setContentsMargins(5, 8, 5, 5)
        denoising_layout.setSpacing(3)
        
        self.denoising_check = QCheckBox("Enable Denoising")
        denoising_layout.addWidget(self.denoising_check)
        
  
        denoising_method_params = QHBoxLayout()
        denoising_method_params.setSpacing(3)
        
        denoising_method_label = QLabel("Method:")
        denoising_method_params.addWidget(denoising_method_label)
        
        self.denoising_method = QComboBox()
        self.denoising_method.addItems(["Wavelet", "FFT", "Wiener", "NLM"])
        denoising_method_params.addWidget(self.denoising_method)
        
        denoising_layout.addLayout(denoising_method_params)
        
  
        denoising_strength_params = QHBoxLayout()
        denoising_strength_params.setSpacing(3)
        
        denoising_strength_label = QLabel("Strength:")
        denoising_strength_params.addWidget(denoising_strength_label)
        
        self.denoising_strength = QDoubleSpinBox()
        self.denoising_strength.setRange(0.1, 2.0)
        self.denoising_strength.setValue(0.5)
        self.denoising_strength.setSingleStep(0.1)
        denoising_strength_params.addWidget(self.denoising_strength)
        
        denoising_layout.addLayout(denoising_strength_params)
        denoising_group.setLayout(denoising_layout)
        special_layout.addWidget(denoising_group)
        
  
        alignment_group = QGroupBox("Peak Alignment")
        alignment_layout = QVBoxLayout()
        alignment_layout.setContentsMargins(5, 8, 5, 5)
        alignment_layout.setSpacing(3)
        
        self.alignment_check = QCheckBox("Enable Peak Alignment")
        alignment_layout.addWidget(self.alignment_check)
        
  
        alignment_method_params = QHBoxLayout()
        alignment_method_params.setSpacing(3)
        
        alignment_method_label = QLabel("Method:")
        alignment_method_params.addWidget(alignment_method_label)
        
        self.alignment_method = QComboBox()
        self.alignment_method.addItems(["DTW", "COW", "ICS", "PAFFT"])
        alignment_method_params.addWidget(self.alignment_method)
        
        alignment_layout.addLayout(alignment_method_params)
        
  
        reference_selection_params = QHBoxLayout()
        reference_selection_params.setSpacing(3)
        
        reference_label = QLabel("Reference:")
        reference_selection_params.addWidget(reference_label)
        
        self.reference_method = QComboBox()
        self.reference_method.addItems(["Mean", "Med", "Max", "First"])
        reference_selection_params.addWidget(self.reference_method)
        
        alignment_layout.addLayout(reference_selection_params)
        alignment_group.setLayout(alignment_layout)
        special_layout.addWidget(alignment_group)
        
        # 拉曼专属预处理占位（需求C）
        raman_group = QGroupBox("Raman Specific (占位接口)")
        raman_layout = QVBoxLayout()
        raman_layout.setContentsMargins(5, 8, 5, 5)
        raman_layout.setSpacing(3)
        
        # 波长校准占位
        wavelength_calib_layout = QHBoxLayout()
        wavelength_calib_layout.setSpacing(8)
        
        self.wavelength_calib_check = QCheckBox("Enable Wavelength Calibration")
        self.wavelength_calib_check.setChecked(False)  # 默认关闭
        wavelength_calib_layout.addWidget(self.wavelength_calib_check)
        
        self.wavelength_calib_file_btn = QPushButton("Load CSV File")
        self.wavelength_calib_file_btn.setEnabled(False)
        wavelength_calib_layout.addWidget(self.wavelength_calib_file_btn)
        
        self.wavelength_calib_file_label = QLabel("No file selected")
        self.wavelength_calib_file_label.setStyleSheet("color: gray; font-style: italic;")
        wavelength_calib_layout.addWidget(self.wavelength_calib_file_label)
        
        raman_layout.addLayout(wavelength_calib_layout)
        
        # 强度校准占位
        intensity_calib_layout = QHBoxLayout()
        intensity_calib_layout.setSpacing(8)
        
        self.intensity_calib_check = QCheckBox("Enable Intensity Calibration")
        self.intensity_calib_check.setChecked(False)  # 默认关闭
        intensity_calib_layout.addWidget(self.intensity_calib_check)
        
        self.intensity_calib_file_btn = QPushButton("Load CSV File")
        self.intensity_calib_file_btn.setEnabled(False)
        intensity_calib_layout.addWidget(self.intensity_calib_file_btn)
        
        self.intensity_calib_file_label = QLabel("No file selected")
        self.intensity_calib_file_label.setStyleSheet("color: gray; font-style: italic;")
        intensity_calib_layout.addWidget(self.intensity_calib_file_label)
        
        raman_layout.addLayout(intensity_calib_layout)
        
        # 添加说明文本
        raman_info_label = QLabel("注：数据外部预处理亦可满足需求")
        raman_info_label.setStyleSheet("color: gray; font-size: 10px; font-style: italic;")
        raman_layout.addWidget(raman_info_label)
        
        raman_group.setLayout(raman_layout)
        special_layout.addWidget(raman_group)
        
        special_tab.setLayout(special_layout)
        tabs.addTab(special_tab, "Special")
        
  
        preview_tab = QWidget()
        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(5)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        
  
        preview_settings_group = QGroupBox("Preview Settings")
        preview_settings_layout = QVBoxLayout()
        preview_settings_layout.setContentsMargins(5, 8, 5, 5)
        preview_settings_layout.setSpacing(3)
        
  
        preview_type_params = QHBoxLayout()
        preview_type_params.setSpacing(3)
        
        preview_type_label = QLabel("Preview Type:")
        preview_type_params.addWidget(preview_type_label)
        
        self.preview_type = QComboBox()
        self.preview_type.addItems(["Original", "Processed", "Comparison"])
        preview_type_params.addWidget(self.preview_type)
        
        preview_settings_layout.addLayout(preview_type_params)
        
  
        preview_sample_params = QHBoxLayout()
        preview_sample_params.setSpacing(3)
        
        preview_sample_label = QLabel("Sample Index:")
        preview_sample_params.addWidget(preview_sample_label)
        
        self.preview_sample = QSpinBox()
        self.preview_sample.setRange(1, 100)
        self.preview_sample.setValue(1)
        preview_sample_params.addWidget(self.preview_sample)
        
        preview_settings_layout.addLayout(preview_sample_params)
        preview_settings_group.setLayout(preview_settings_layout)
        preview_layout.addWidget(preview_settings_group)
        
  
        preview_plot_group = QGroupBox("Preview Plot")
        preview_plot_layout = QVBoxLayout()
        preview_plot_layout.setContentsMargins(5, 8, 5, 5)
        preview_plot_layout.setSpacing(3)
        
        if pg is not None:
            self.preview_plot = pg.PlotWidget()
            self.preview_plot.setBackground('w')
            self.preview_plot.showGrid(x=True, y=True)
            self.preview_plot.setLabel('left', 'Intensity')
            self.preview_plot.setLabel('bottom', 'Wavelength (nm)')
            preview_plot_layout.addWidget(self.preview_plot)
        else:
            # Fallback when pyqtgraph is not available
            self.preview_plot = None
            placeholder_label = QLabel("Preview plot requires pyqtgraph")
            placeholder_label.setStyleSheet("color: gray; font-style: italic;")
            preview_plot_layout.addWidget(placeholder_label)
        
        preview_plot_group.setLayout(preview_plot_layout)
        preview_layout.addWidget(preview_plot_group)
        
        preview_tab.setLayout(preview_layout)
        tabs.addTab(preview_tab, "Preview")
        
        main_layout.addWidget(tabs)
        
  
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        
        self.preview_button = QPushButton("Preview")
        button_layout.addWidget(self.preview_button)
        
        self.apply_button = QPushButton("Apply")
        button_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset")
        button_layout.addWidget(self.reset_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def get_parameters(self):
        """获取所有预processingparameters"""
        params = {
            'despiking': {
                'enabled': self.despiking_check.isChecked(),
                'method': self.despiking_method.currentText() if self.despiking_check.isChecked() else None,
                'window': self.despiking_window.value() if self.despiking_check.isChecked() else None,
                'threshold': self.despiking_threshold.value() if self.despiking_check.isChecked() else None
            },
            'baseline_correction': {
                'enabled': self.baseline_check.isChecked(),
                'method': self.baseline_method.currentText() if self.baseline_check.isChecked() else None,
                'poly_order': self.poly_order.value() if self.baseline_check.isChecked() else None
            },
            'smoothing': {
                'enabled': self.smoothing_check.isChecked(),
                'method': self.smoothing_method.currentText() if self.smoothing_check.isChecked() else None,
                'window_size': self.window_size.value() if self.smoothing_check.isChecked() else None,
                'poly_order': self.poly_sgf_order.value() if self.smoothing_check.isChecked() and self.smoothing_method.currentText() == "Savitzky-Golay" else None
            },
            'scatter_correction': {
                'enabled': self.scatter_check.isChecked(),
                'method': self.scatter_method.currentText() if self.scatter_check.isChecked() else None
            },
            'normalization': {
                'enabled': self.normalization_check.isChecked(),
                'method': self.normalization_method.currentText() if self.normalization_check.isChecked() else None
            },
            'standardization': {
                'enabled': self.standardization_check.isChecked(),
                'method': self.standardization_method.currentText() if self.standardization_check.isChecked() else None
            },
            'derivative': {
                'enabled': self.derivative_check.isChecked(),
                'order': self.derivative_order.value() if self.derivative_check.isChecked() else None,
                'method': self.derivative_method.currentText() if self.derivative_check.isChecked() else None
            },
            'outlier_detection': {
                'enabled': self.outlier_check.isChecked(),
                'method': self.outlier_method.currentText() if self.outlier_check.isChecked() else None,
                'threshold': self.outlier_threshold.value() if self.outlier_check.isChecked() else None
            },
            'denoising': {
                'enabled': self.denoising_check.isChecked(),
                'method': self.denoising_method.currentText() if self.denoising_check.isChecked() else None,
                'strength': self.denoising_strength.value() if self.denoising_check.isChecked() else None
            },
            'peak_alignment': {
                'enabled': self.alignment_check.isChecked(),
                'method': self.alignment_method.currentText() if self.alignment_check.isChecked() else None,
                'reference': self.reference_method.currentText() if self.alignment_check.isChecked() else None
            },
            'raman_specific': {
                'wavelength_calibration': {
                    'enabled': self.wavelength_calib_check.isChecked(),
                    'file_path': getattr(self, 'wavelength_calib_file_path', None) if self.wavelength_calib_check.isChecked() else None
                },
                'intensity_calibration': {
                    'enabled': self.intensity_calib_check.isChecked(),
                    'file_path': getattr(self, 'intensity_calib_file_path', None) if self.intensity_calib_check.isChecked() else None
                }
            }
        }
        
  
        if hasattr(self, 'plugin_params') and self.plugin_params:
            params['plugins'] = self.plugin_params
            
        return params
    
    def reset_parameters(self):
        """重置所有parameters为默认值"""
        # 去峰参数重置
        self.despiking_check.setChecked(False)
        self.despiking_method.setCurrentIndex(0)
        self.despiking_window.setValue(11)
        self.despiking_threshold.setValue(5.0)
        
        # 基线校正参数重置
        self.baseline_check.setChecked(False)
        self.baseline_method.setCurrentIndex(0)
        self.poly_order.setValue(3)
        
  
        self.smoothing_check.setChecked(False)
        self.smoothing_method.setCurrentIndex(0)
        self.window_size.setValue(15)
        self.poly_sgf_order.setValue(2)
        
  
        self.scatter_check.setChecked(False)
        self.scatter_method.setCurrentIndex(0)
        
  
        self.normalization_check.setChecked(False)
        self.normalization_method.setCurrentIndex(0)
        
  
        self.standardization_check.setChecked(False)
        self.standardization_method.setCurrentIndex(0)
        
  
        self.derivative_check.setChecked(False)
        self.derivative_order.setValue(1)
        self.derivative_method.setCurrentIndex(0)
        
  
        self.outlier_check.setChecked(False)
        self.outlier_method.setCurrentIndex(0)
        self.outlier_threshold.setValue(3.0)
        
  
        self.denoising_check.setChecked(False)
        self.denoising_method.setCurrentIndex(0)
        self.denoising_strength.setValue(0.5)
        
  
        self.alignment_check.setChecked(False)
        self.alignment_method.setCurrentIndex(0)
        self.reference_method.setCurrentIndex(0)
        
  
        self.preview_type.setCurrentIndex(0)
        self.preview_sample.setValue(1)
        
        # 拉曼专属预处理参数重置
        self.wavelength_calib_check.setChecked(False)
        self.wavelength_calib_file_btn.setEnabled(False)
        self.wavelength_calib_file_label.setText("No file selected")
        if hasattr(self, 'wavelength_calib_file_path'):
            delattr(self, 'wavelength_calib_file_path')
        
        self.intensity_calib_check.setChecked(False)
        self.intensity_calib_file_btn.setEnabled(False)
        self.intensity_calib_file_label.setText("No file selected")
        if hasattr(self, 'intensity_calib_file_path'):
            delattr(self, 'intensity_calib_file_path')
        
  
        if hasattr(self, 'preview_plot'):
            self.preview_plot.clear()
    
    def update_preview(self, original_data, processed_data=None):
        """更新预览图表"""
        if not hasattr(self, 'preview_plot') or self.preview_plot is None:
            return
            
        self.preview_plot.clear()
        
        if original_data is None:
            return
            
  
        sample_idx = min(self.preview_sample.value() - 1, len(original_data) - 1)
        if sample_idx < 0:
            sample_idx = 0
        
  
        preview_type = self.preview_type.currentText()
        
  
        wavelengths = np.arange(original_data.shape[1])
        
        if preview_type == "Original":
  
            self.preview_plot.plot(wavelengths, original_data[sample_idx], pen='b', name='Original')
        
        elif preview_type == "Processed" and processed_data is not None:
  
            self.preview_plot.plot(wavelengths, processed_data[sample_idx], pen='r', name='Processed')
        
        elif preview_type == "Comparison" and processed_data is not None:
  
            self.preview_plot.plot(wavelengths, original_data[sample_idx], pen='b', name='Original')
            self.preview_plot.plot(wavelengths, processed_data[sample_idx], pen='r', name='Processed')
    
    def get_selected_methods(self):
        """获取选中的预processingmethod列表"""
        methods = []
        
        if self.despiking_check.isChecked():
            methods.append(f"Despiking ({self.despiking_method.currentText()})")
        
        if self.baseline_check.isChecked():
            methods.append(f"Baseline Correction ({self.baseline_method.currentText()})")
        
        if self.smoothing_check.isChecked():
            methods.append(f"{self.smoothing_method.currentText()} Smoothing")
        
        if self.scatter_check.isChecked():
            methods.append(f"{self.scatter_method.currentText()} Scatter Correction")
        
        if self.normalization_check.isChecked():
            methods.append(f"{self.normalization_method.currentText()} Normalization")
        
        if self.standardization_check.isChecked():
            methods.append(f"{self.standardization_method.currentText()} Standardization")
        
        if self.derivative_check.isChecked():
            methods.append(f"{self.derivative_order.value()} Order Derivative")
        
        if self.outlier_check.isChecked():
            methods.append(f"{self.outlier_method.currentText()} Outlier Detection")
        
        if self.denoising_check.isChecked():
            methods.append(f"{self.denoising_method.currentText()} Denoising")
        
        if self.alignment_check.isChecked():
            methods.append(f"{self.alignment_method.currentText()} Peak Alignment")
            
        return methods
    
    def display_error(self, message):
        """显示error消息"""
        QMessageBox.critical(self, "Error", message)
        
    def update_preprocessed_data(self, data):
        """更新预processing后的data显示"""
  
        pass
        
    def show_parameters(self, method):
        """显示parameters设置对话框 - 兼容旧接口"""
  
        pass
        
    def add_custom_algorithm(self, code):
        """添加自定义预processing算法 - 兼容旧接口"""
  
        pass
