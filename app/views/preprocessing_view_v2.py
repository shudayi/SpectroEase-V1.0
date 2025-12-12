# app/views/preprocessing_view_v2.py
"""
Redesigned preprocessing view - organized by spectral type and processing workflow
Meets the requirements of professional spectral analysis software
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QHBoxLayout, QFrame, QCheckBox, QGroupBox, QTabWidget, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class PreprocessingViewV2(QWidget):
    """
    Improved preprocessing view
    
    Organization principles:
    1. Tabs by spectral type: Universal / Raman / MIR-FTIR / NIR / Advanced
    2. Processing workflow order: Denoising → Baseline → Normalization → Derivative → Special processing
    3. Add professional algorithms: ModPoly, SNIP, Fluorescence removal, Atmospheric compensation, Water peak removal
    """
    
    def __init__(self, plugins=None):
        super().__init__()
        self.plugins = plugins if plugins else {}
        self.custom_algorithms = {}  # Store custom algorithms
        self.custom_algo_checkboxes = {}  # Store custom algorithm checkboxes
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Set font
        app_font = QFont("Microsoft YaHei UI", 9)
        self.setFont(app_font)
        
        # Create Tab Widget
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        
        # Tab 1: Universal preprocessing
        universal_tab = self._create_universal_tab()
        tabs.addTab(universal_tab, "Universal Preprocessing")
        
        # Tab 2: Raman specific
        raman_tab = self._create_raman_tab()
        tabs.addTab(raman_tab, "Raman Specific ⭐")
        
        # Tab 3: MIR-FTIR specific
        mir_tab = self._create_mir_tab()
        tabs.addTab(mir_tab, "MIR/FTIR ⭐")
        
        # Tab 4: NIR specific
        nir_tab = self._create_nir_tab()
        tabs.addTab(nir_tab, "NIR Specific ⭐")
        
        # Tab 5: Advanced processing
        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced Processing")
        
        # Tab 6: Custom (V1.3.1: Custom algorithm independent tab)
        custom_tab = self._create_custom_tab()
        tabs.addTab(custom_tab, "Custom")
        
        main_layout.addWidget(tabs)
        
        # Apply button
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("预览效果")
        self.preview_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        button_layout.addWidget(self.preview_button)
        
        self.apply_button = QPushButton("应用预处理")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        button_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #757575; }
        """)
        button_layout.addWidget(self.reset_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    
    def _create_universal_tab(self) -> QWidget:
        """Create universal preprocessing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        
        # 1. Despiking - Raman needs it most, but universal can also use
        despiking_group = self._create_despiking_group()
        content_layout.addWidget(despiking_group)
        
        # 2. Baseline Correction - Universal + new algorithms
        baseline_group = self._create_baseline_group()
        content_layout.addWidget(baseline_group)
        
        # 3. Smoothing
        smoothing_group = self._create_smoothing_group()
        content_layout.addWidget(smoothing_group)
        
        # 4. Scatter Correction
        scatter_group = self._create_scatter_group()
        content_layout.addWidget(scatter_group)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_raman_tab(self) -> QWidget:
        """Create Raman-specific tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        
        # Description
        info_label = QLabel("Raman-specific processing algorithms")
        info_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #E91E63; padding: 8px;")
        content_layout.addWidget(info_label)
        
        # 1. Fluorescence background removal ⭐⭐⭐⭐⭐ (Most important!)
        fluor_group = self._create_fluorescence_removal_group()
        content_layout.addWidget(fluor_group)
        
        # 2. Raman Shift calibration
        shift_group = self._create_raman_shift_calibration_group()
        content_layout.addWidget(shift_group)
        
        # 3. Despiking (Raman especially needs)
        despiking_group = self._create_despiking_group(raman_specific=True)
        content_layout.addWidget(despiking_group)
        
        # 4. Baseline correction (recommend ModPoly/SNIP)
        baseline_hint = QLabel("Hint: Raman spectra recommend ModPoly or SNIP baseline correction (in General Tab)")
        baseline_hint.setStyleSheet("font-size: 10px; color: #666; padding: 5px; background-color: #FFF9C4;")
        baseline_hint.setWordWrap(True)
        content_layout.addWidget(baseline_hint)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_mir_tab(self) -> QWidget:
        """Create MIR/FTIR-specific tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        
        # Description
        info_label = QLabel("MIR/FTIR-specific processing algorithms")
        info_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #9C27B0; padding: 8px;")
        content_layout.addWidget(info_label)
        
        # 1. Atmospheric compensation ⭐⭐⭐⭐⭐ (Reviewer requirement!)
        atmos_group = self._create_atmospheric_compensation_group()
        content_layout.addWidget(atmos_group)
        
        # 2. Other hints
        hint_label = QLabel("Hint: MIR spectra can combine with baseline correction, smoothing, normalization in General Tab")
        hint_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px; background-color: #E1BEE7;")
        hint_label.setWordWrap(True)
        content_layout.addWidget(hint_label)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_nir_tab(self) -> QWidget:
        """Create NIR-specific tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        
        # Description
        info_label = QLabel("NIR-specific processing algorithms")
        info_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #FF9800; padding: 8px;")
        content_layout.addWidget(info_label)
        
        # 1. Water peak removal ⭐⭐⭐⭐
        water_group = self._create_water_peak_removal_group()
        content_layout.addWidget(water_group)
        
        # 2. Other hints
        hint_label = QLabel("Hint: NIR spectra typically need MSC, SNV scatter correction (in General Tab)")
        hint_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px; background-color: #FFE0B2;")
        hint_label.setWordWrap(True)
        content_layout.addWidget(hint_label)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create advanced processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        
        # Normalization
        normalization_group = self._create_normalization_group()
        content_layout.addWidget(normalization_group)
        
        # Standardization
        standardization_group = self._create_standardization_group()
        content_layout.addWidget(standardization_group)
        
        # Derivative
        derivative_group = self._create_derivative_group()
        content_layout.addWidget(derivative_group)
        
        # Model transfer
        transfer_group = self._create_model_transfer_group()
        content_layout.addWidget(transfer_group)
        
        # Custom Algorithms Section
        custom_algo_group = self._create_custom_algorithms_group()
        content_layout.addWidget(custom_algo_group)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
    
    # ======================== Component creation methods ========================
    
    def _create_despiking_group(self, raman_specific=False) -> QGroupBox:
        """Create despiking component"""
        title = "Despiking (去峰) - Raman必备" if raman_specific else "Despiking (去峰)"
        group = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.despiking_check = QCheckBox("启用去峰")
        layout.addWidget(self.despiking_check)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("方法:"))
        self.despiking_method = QComboBox()
        self.despiking_method.addItems(["MAD", "Local Z-score"])
        params_layout.addWidget(self.despiking_method)
        
        params_layout.addWidget(QLabel("窗口:"))
        self.despiking_window = QSpinBox()
        self.despiking_window.setRange(3, 21)
        self.despiking_window.setValue(11)
        self.despiking_window.setSingleStep(2)
        self.despiking_window.setToolTip("窗口大小 (3-21, 奇数)")
        params_layout.addWidget(self.despiking_window)
        
        params_layout.addWidget(QLabel("阈值:"))
        self.despiking_threshold = QDoubleSpinBox()
        self.despiking_threshold.setRange(1.0, 10.0)
        self.despiking_threshold.setValue(5.0)
        self.despiking_threshold.setSingleStep(0.5)
        self.despiking_threshold.setToolTip("检测阈值 (1-10)")
        params_layout.addWidget(self.despiking_threshold)
        
        layout.addLayout(params_layout)
        group.setLayout(layout)
        
        return group
    
    def _create_baseline_group(self) -> QGroupBox:
        """Create baseline correction component (including new algorithms)"""
        group = QGroupBox("Baseline Correction")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.baseline_check = QCheckBox("启用基线校正")
        layout.addWidget(self.baseline_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.baseline_method = QComboBox()
        self.baseline_method.addItems([
            "Polynomial (多项式)",
            "ALS (渐近最小二乘)",
            "airPLS (自适应迭代)",
            "ModPoly (⭐ Raman专用)",
            "SNIP (⭐ 多峰光谱)"
        ])
        self.baseline_method.setToolTip(
            "ModPoly: Raman荧光背景去除\n"
            "SNIP: 适合复杂多峰光谱\n"
            "Polynomial: 通用简单基线\n"
            "ALS/airPLS: 自动拟合基线"
        )
        method_layout.addWidget(self.baseline_method)
        layout.addLayout(method_layout)
        
        # Parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("阶数/迭代:"))
        self.baseline_order = QSpinBox()
        self.baseline_order.setRange(1, 100)
        self.baseline_order.setValue(5)
        self.baseline_order.setToolTip("多项式阶数 / ModPoly迭代次数 / SNIP半宽度")
        params_layout.addWidget(self.baseline_order)
        layout.addLayout(params_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_fluorescence_removal_group(self) -> QGroupBox:
        """Create Raman fluorescence removal component"""
        group = QGroupBox("⭐ Fluorescence Background Removal")
        group.setStyleSheet("QGroupBox { font-weight: bold; color: #E91E63; }")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.fluor_check = QCheckBox("启用荧光背景去除")
        layout.addWidget(self.fluor_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.fluor_method = QComboBox()
        self.fluor_method.addItems([
            "ModPoly (改进多项式)",
            "VRA (可变比率)",
            "AFBS (自适应减除)"
        ])
        method_layout.addWidget(self.fluor_method)
        layout.addLayout(method_layout)
        
        # Parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("多项式阶数:"))
        self.fluor_poly_order = QSpinBox()
        self.fluor_poly_order.setRange(3, 10)
        self.fluor_poly_order.setValue(5)
        params_layout.addWidget(self.fluor_poly_order)
        
        params_layout.addWidget(QLabel("迭代次数:"))
        self.fluor_max_iter = QSpinBox()
        self.fluor_max_iter.setRange(10, 200)
        self.fluor_max_iter.setValue(100)
        params_layout.addWidget(self.fluor_max_iter)
        layout.addLayout(params_layout)
        
        info_label = QLabel("Fluorescence is the major interference in Raman, strongly recommended!")
        info_label.setStyleSheet("font-size: 9px; color: #C62828; font-weight: bold;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_raman_shift_calibration_group(self) -> QGroupBox:
        """Create Raman Shift calibration component"""
        group = QGroupBox("Raman Shift Calibration")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.raman_shift_check = QCheckBox("启用Raman Shift校准")
        layout.addWidget(self.raman_shift_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("标准物质:"))
        self.raman_calibration_standard = QComboBox()
        self.raman_calibration_standard.addItems([
            "Silicon (520.7 cm⁻¹)",
            "Cyclohexane (多峰)",
            "Polystyrene (多峰)",
            "Custom (自定义)"
        ])
        method_layout.addWidget(self.raman_calibration_standard)
        layout.addLayout(method_layout)
        
        info_label = QLabel("ℹ 需要提供标准物质的光谱数据")
        info_label.setStyleSheet("font-size: 9px; color: #666;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_atmospheric_compensation_group(self) -> QGroupBox:
        """Create MIR atmospheric compensation component"""
        group = QGroupBox("⭐ Atmospheric Compensation")
        group.setStyleSheet("QGroupBox { font-weight: bold; color: #9C27B0; }")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.atmos_check = QCheckBox("Enable atmospheric compensation")
        layout.addWidget(self.atmos_check)
        
        # CO2 compensation
        self.atmos_co2_check = QCheckBox("Compensate CO₂ (2280-2400 cm⁻¹)")
        self.atmos_co2_check.setChecked(True)
        layout.addWidget(self.atmos_co2_check)
        
        # H2O compensation
        self.atmos_h2o_check = QCheckBox("Compensate H₂O (1300-1900, 3200-3900 cm⁻¹)")
        self.atmos_h2o_check.setChecked(True)
        layout.addWidget(self.atmos_h2o_check)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.atmos_method = QComboBox()
        self.atmos_method.addItems([
            "Interpolation (插值替换)",
            "Background (背景减除)",
            "Reference (参考光谱)"
        ])
        method_layout.addWidget(self.atmos_method)
        layout.addLayout(method_layout)
        
        info_label = QLabel("Essential for MIR/FTIR! Corrects CO₂ and H₂O interference")
        info_label.setStyleSheet("font-size: 9px; color: #6A1B9A; font-weight: bold;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_water_peak_removal_group(self) -> QGroupBox:
        """Create NIR water peak removal component"""
        group = QGroupBox("⭐ Water Peak Removal")
        group.setStyleSheet("QGroupBox { font-weight: bold; color: #FF9800; }")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.water_check = QCheckBox("启用水峰去除")
        layout.addWidget(self.water_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.water_method = QComboBox()
        self.water_method.addItems([
            "EPO (外部参数正交化)",
            "DOSC (直接正交信号校正)",
            "Interpolation (插值)"
        ])
        method_layout.addWidget(self.water_method)
        layout.addLayout(method_layout)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("成分数:"))
        self.water_n_components = QSpinBox()
        self.water_n_components.setRange(1, 10)
        self.water_n_components.setValue(3)
        self.water_n_components.setToolTip("EPO/DOSC使用的正交成分数")
        params_layout.addWidget(self.water_n_components)
        layout.addLayout(params_layout)
        
        info_label = QLabel("NIR has strong water absorption at 1400-1900 nm, essential for food/agriculture!")
        info_label.setStyleSheet("font-size: 9px; color: #E65100; font-weight: bold;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_model_transfer_group(self) -> QGroupBox:
        """Create model transfer component"""
        group = QGroupBox("⭐ Model Transfer")
        group.setStyleSheet("QGroupBox { font-weight: bold; color: #00897B; }")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.transfer_check = QCheckBox("启用模型传递")
        layout.addWidget(self.transfer_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.transfer_method = QComboBox()
        self.transfer_method.addItems([
            "PDS (分段直接标准化)",
            "SBC (斜率偏移校正)"
        ])
        method_layout.addWidget(self.transfer_method)
        layout.addLayout(method_layout)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("窗口大小:"))
        self.transfer_window = QSpinBox()
        self.transfer_window.setRange(3, 21)
        self.transfer_window.setValue(5)
        self.transfer_window.setSingleStep(2)
        self.transfer_window.setToolTip("PDS窗口大小 (奇数)")
        params_layout.addWidget(self.transfer_window)
        layout.addLayout(params_layout)
        
        info_label = QLabel("ℹ 需要提供主仪器和从属仪器的配对样品数据")
        info_label.setStyleSheet("font-size: 9px; color: #666;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_smoothing_group(self) -> QGroupBox:
        """Create smoothing component"""
        group = QGroupBox("Smoothing")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.smoothing_check = QCheckBox("启用平滑")
        layout.addWidget(self.smoothing_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.smoothing_method = QComboBox()
        self.smoothing_method.addItems([
            "Savitzky-Golay",
            "Moving Average",
            "Gaussian",
            "Median"
        ])
        method_layout.addWidget(self.smoothing_method)
        layout.addLayout(method_layout)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("窗口:"))
        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(3, 51)
        self.smoothing_window.setValue(15)
        self.smoothing_window.setSingleStep(2)
        params_layout.addWidget(self.smoothing_window)
        
        params_layout.addWidget(QLabel("多项式阶数:"))
        self.smoothing_poly_order = QSpinBox()
        self.smoothing_poly_order.setRange(1, 5)
        self.smoothing_poly_order.setValue(2)
        params_layout.addWidget(self.smoothing_poly_order)
        layout.addLayout(params_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_scatter_group(self) -> QGroupBox:
        """Create scatter correction component"""
        group = QGroupBox("Scatter Correction")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.scatter_check = QCheckBox("启用散射校正")
        layout.addWidget(self.scatter_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.scatter_method = QComboBox()
        self.scatter_method.addItems(["MSC", "SNV", "EMSC", "RNV", "OSC"])
        method_layout.addWidget(self.scatter_method)
        layout.addLayout(method_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_normalization_group(self) -> QGroupBox:
        """Create normalization component"""
        group = QGroupBox("Normalization")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.normalization_check = QCheckBox("启用归一化")
        layout.addWidget(self.normalization_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.normalization_method = QComboBox()
        self.normalization_method.addItems(["Min-Max", "Vector", "Area", "Maximum"])
        method_layout.addWidget(self.normalization_method)
        layout.addLayout(method_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_standardization_group(self) -> QGroupBox:
        """Create standardization component"""
        group = QGroupBox("Standardization")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.standardization_check = QCheckBox("启用标准化")
        layout.addWidget(self.standardization_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.standardization_method = QComboBox()
        self.standardization_method.addItems(["Z-Score", "Robust"])
        method_layout.addWidget(self.standardization_method)
        layout.addLayout(method_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_derivative_group(self) -> QGroupBox:
        """Create derivative component"""
        group = QGroupBox("Derivative")
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        self.derivative_check = QCheckBox("启用导数")
        layout.addWidget(self.derivative_check)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.derivative_method = QComboBox()
        self.derivative_method.addItems([
            "First Derivative",
            "Second Derivative",
            "Savitzky-Golay",
            "Finite Difference",
            "Gap-Segment"
        ])
        method_layout.addWidget(self.derivative_method)
        layout.addLayout(method_layout)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("窗口/间隔:"))
        self.derivative_window = QSpinBox()
        self.derivative_window.setRange(3, 21)
        self.derivative_window.setValue(11)
        params_layout.addWidget(self.derivative_window)
        layout.addLayout(params_layout)
        
        group.setLayout(layout)
        return group
    
    def get_preprocessing_params(self) -> dict:
        """Get all preprocessing parameters"""
        params = {}
        
        # Despiking
        if hasattr(self, 'despiking_check') and self.despiking_check.isChecked():
            params['despiking'] = {
                'enabled': True,
                'method': self.despiking_method.currentText(),
                'window': self.despiking_window.value(),
                'threshold': self.despiking_threshold.value()
            }
        
        # Baseline correction
        if hasattr(self, 'baseline_check') and self.baseline_check.isChecked():
            params['baseline'] = {
                'enabled': True,
                'method': self.baseline_method.currentText(),
                'order': self.baseline_order.value()
            }
        
        # Raman fluorescence removal
        if hasattr(self, 'fluor_check') and self.fluor_check.isChecked():
            params['fluorescence_removal'] = {
                'enabled': True,
                'method': self.fluor_method.currentText(),
                'poly_order': self.fluor_poly_order.value(),
                'max_iter': self.fluor_max_iter.value()
            }
        
        # MIR atmospheric compensation
        if hasattr(self, 'atmos_check') and self.atmos_check.isChecked():
            params['atmospheric_compensation'] = {
                'enabled': True,
                'method': self.atmos_method.currentText(),
                'compensate_co2': self.atmos_co2_check.isChecked(),
                'compensate_h2o': self.atmos_h2o_check.isChecked()
            }
        
        # NIR water peak removal
        if hasattr(self, 'water_check') and self.water_check.isChecked():
            params['water_peak_removal'] = {
                'enabled': True,
                'method': self.water_method.currentText(),
                'n_components': self.water_n_components.value()
            }
        
        # Model transfer
        if hasattr(self, 'transfer_check') and self.transfer_check.isChecked():
            params['model_transfer'] = {
                'enabled': True,
                'method': self.transfer_method.currentText(),
                'window_size': self.transfer_window.value()
            }
        
        # Other universal processing...
        if hasattr(self, 'smoothing_check') and self.smoothing_check.isChecked():
            params['smoothing'] = {
                'enabled': True,
                'method': self.smoothing_method.currentText(),
                'window': self.smoothing_window.value(),
                'poly_order': self.smoothing_poly_order.value()
            }
        
        if hasattr(self, 'scatter_check') and self.scatter_check.isChecked():
            params['scatter'] = {
                'enabled': True,
                'method': self.scatter_method.currentText()
            }
        
        if hasattr(self, 'normalization_check') and self.normalization_check.isChecked():
            params['normalization'] = {
                'enabled': True,
                'method': self.normalization_method.currentText()
            }
        
        if hasattr(self, 'standardization_check') and self.standardization_check.isChecked():
            params['standardization'] = {
                'enabled': True,
                'method': self.standardization_method.currentText()
            }
        
        if hasattr(self, 'derivative_check') and self.derivative_check.isChecked():
            params['derivative'] = {
                'enabled': True,
                'method': self.derivative_method.currentText(),
                'window': self.derivative_window.value()
            }
        
        return params
    
    def get_parameters(self) -> dict:
        """Compatibility method: return preprocessing parameters (consistent with old UI interface)"""
        return self.get_preprocessing_params()
    
    def reset_parameters(self):
        """Reset all parameters to default values"""
        # Despiking
        if hasattr(self, 'despiking_check'):
            self.despiking_check.setChecked(False)
        
        # Baseline correction
        if hasattr(self, 'baseline_check'):
            self.baseline_check.setChecked(False)
        
        # Raman fluorescence removal
        if hasattr(self, 'fluor_check'):
            self.fluor_check.setChecked(False)
        
        # MIR atmospheric compensation
        if hasattr(self, 'atmos_check'):
            self.atmos_check.setChecked(False)
        
        # NIR water peak removal
        if hasattr(self, 'water_check'):
            self.water_check.setChecked(False)
        
        # Smoothing
        if hasattr(self, 'smoothing_check'):
            self.smoothing_check.setChecked(False)
        
        # Scatter correction
        if hasattr(self, 'scatter_check'):
            self.scatter_check.setChecked(False)
        
        # Normalization
        if hasattr(self, 'normalization_check'):
            self.normalization_check.setChecked(False)
        
        # Standardization
        if hasattr(self, 'standardization_check'):
            self.standardization_check.setChecked(False)
        
        # Derivative
        if hasattr(self, 'derivative_check'):
            self.derivative_check.setChecked(False)
        
        # Custom algorithms
        for checkbox in self.custom_algo_checkboxes.values():
            checkbox.setChecked(False)

    def _create_custom_tab(self) -> QWidget:
        """V1.3.1: Create Custom Tab - independently display custom algorithms"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title and description
        custom_title = QLabel("Custom Algorithms")
        custom_title.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        custom_title.setStyleSheet("color: #4CAF50; padding: 5px;")
        content_layout.addWidget(custom_title)
        
        custom_info = QLabel("Use 'Tools → Algorithm Conversion (LLM)' to add custom preprocessing algorithms.")
        custom_info.setStyleSheet("color: #666; font-style: italic; padding: 5px; font-size: 11px;")
        content_layout.addWidget(custom_info)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #ddd; margin: 10px 0px;")
        content_layout.addWidget(separator)
        
        # Custom algorithm list container
        self.custom_algos_layout = QVBoxLayout()
        self.custom_algos_layout.setSpacing(8)
        content_layout.addLayout(self.custom_algos_layout)
        
        # No algorithm prompt
        self.no_custom_algo_label = QLabel("No custom algorithms yet.\n\nClick 'Tools → Algorithm Conversion' to create one.")
        self.no_custom_algo_label.setAlignment(Qt.AlignCenter)
        self.no_custom_algo_label.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 12px;
                padding: 60px 40px;
                border: 3px dashed #ddd;
                border-radius: 10px;
                background-color: #fafafa;
            }
        """)
        self.custom_algos_layout.addWidget(self.no_custom_algo_label)
        
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_custom_algorithms_group(self) -> QGroupBox:
        """Create custom algorithms section"""
        group = QGroupBox("Custom Algorithms")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E91E63;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Container for custom algorithm checkboxes
        self.custom_algo_container = QVBoxLayout()
        self.custom_algo_container.setSpacing(3)
        layout.addLayout(self.custom_algo_container)
        
        # Info label when no custom algorithms
        self.no_custom_algo_label = QLabel("No custom algorithms. Use LLM to create custom preprocessing algorithms.")
        self.no_custom_algo_label.setStyleSheet("color: #666; font-style: italic; font-size: 9px;")
        self.no_custom_algo_label.setWordWrap(True)
        layout.addWidget(self.no_custom_algo_label)
        
        group.setLayout(layout)
        return group
    
    def add_custom_algorithm(self, code: str, auto_replace: bool = False):
        """
        Add custom preprocessing algorithm
        V1.3.1: 添加接口类支持
        
        Args:
            code: Python code defining the custom algorithm class
            auto_replace: If True, automatically replace existing algorithm without showing dialog
        """
        try:
            import types
            import numpy as np
            import pandas as pd
            from PyQt5.QtWidgets import QFrame
            from interfaces.preprocessing_algorithm import PreprocessingAlgorithm  # V1.3.1
            
            # Create a module to execute the code
            mod = types.ModuleType('custom_preprocessor')
            
            # V1.3.6: Provide comprehensive imports for scientific computing
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'PreprocessingAlgorithm': PreprocessingAlgorithm,
                'Dict': __import__('typing').Dict,
                'Any': __import__('typing').Any,
                'Tuple': __import__('typing').Tuple,
                'List': __import__('typing').List,
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
            
            # V1.3.7: Add numpy.polynomial support (for polyfit, polyval, etc.)
            try:
                from numpy.polynomial import polynomial
                from numpy.polynomial.polynomial import polyfit, polyval
                mod.__dict__.update({
                    'polynomial': polynomial,
                    'polyfit': polyfit,
                    'polyval': polyval,
                })
            except ImportError:
                pass
            
            # V1.3.6: Add sklearn support (if available)
            try:
                import sklearn
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                from sklearn.decomposition import PCA
                mod.__dict__.update({
                    'sklearn': sklearn,
                    'StandardScaler': StandardScaler,
                    'MinMaxScaler': MinMaxScaler,
                    'PCA': PCA,
                })
            except ImportError:
                pass
            
            # V1.3.7: BUGFIX - Record class list BEFORE execution (not after!)
            classes_before = set(item for item in mod.__dict__.values() if isinstance(item, type))
            
            exec(code, mod.__dict__)
            
            # V1.3.7: Record class list after execution, find newly added classes
            classes_after = set(item for item in mod.__dict__.values() if isinstance(item, type))
            new_classes = classes_after - classes_before
            
            # Find the algorithm class
            algorithm = None
            
            for item in new_classes:
                # V1.3.2: Must be subclass and not abstract class
                try:
                    import inspect
                    if (issubclass(item, PreprocessingAlgorithm) and 
                        not inspect.isabstract(item)):
                        algorithm = item()
                        break
                except (TypeError, AttributeError) as e:
                    print(f"Failed to instantiate algorithm: {e}")
                    continue
            
            if algorithm is None:
                raise Exception("No valid algorithm class found in code")
            
            # Get algorithm name
            algorithm_name = algorithm.get_name()
            
            # Check if algorithm already exists
            if algorithm_name in self.custom_algorithms:
                if not auto_replace:
                    # Show dialog only if not auto-replacing (e.g., during startup)
                    from PyQt5.QtWidgets import QMessageBox
                    reply = QMessageBox.question(
                        self,
                        'Algorithm Exists',
                        f"Custom algorithm '{algorithm_name}' already exists. Replace it?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                # Remove old algorithm UI
                self._remove_custom_algorithm_ui(algorithm_name)
            
            # Store algorithm
            self.custom_algorithms[algorithm_name] = algorithm
            
            # Also add to plugins dict so preprocessing service can find it
            if self.plugins is None:
                self.plugins = {}
            self.plugins[algorithm_name] = algorithm
            
            # Hide "no algorithms" label if this is the first one
            if len(self.custom_algorithms) == 1:
                self.no_custom_algo_label.hide()
            
            # Create UI for the algorithm
            self._create_custom_algorithm_ui(algorithm_name, algorithm)
            
            print(f"Successfully added custom preprocessing algorithm: {algorithm_name}")
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                'Error',
                f'Failed to add custom preprocessing algorithm:\n{str(e)}'
            )
            raise Exception(f"Error loading custom preprocessing algorithm: {str(e)}")
    
    def _create_custom_algorithm_ui(self, name: str, algorithm):
        """V1.3.1: Create UI elements for a custom algorithm in Custom Tab"""
        from PyQt5.QtWidgets import QFrame, QPushButton
        
        # V1.3.1: Create algorithm UI framework (green theme)
        algo_frame = QFrame()
        algo_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E8F5E9,
                    stop:1 white
                );
                border-left: 4px solid #4CAF50;
                border-radius: 5px;
                padding: 12px;
                margin: 5px 0px;
            }
            QFrame:hover {
                background: #E8F5E9;
                border-left: 4px solid #43A047;
            }
        """)
        algo_layout = QVBoxLayout(algo_frame)
        algo_layout.setSpacing(8)
        
        # Title row (name + enable checkbox)
        title_layout = QHBoxLayout()
        
        # V1.3.5: Use concise custom identifier
        checkbox = QCheckBox(f"[Custom Preproc] {name}")
        checkbox.setStyleSheet("font-size: 13px; font-weight: bold; color: #2E7D32; background-color: transparent;")
        checkbox.setToolTip(f"Custom Preprocessing: {name}\n✅ Created via LLM conversion")
        self.custom_algo_checkboxes[name] = checkbox
        title_layout.addWidget(checkbox)
        
        title_layout.addStretch()
        
        # Parameters button
        try:
            # V1.4.1: Fix method name - use get_params_info instead of get_parameter_info
            if hasattr(algorithm, 'get_params_info'):
                params = algorithm.get_params_info()
            elif hasattr(algorithm, 'get_parameter_info'):
                params = algorithm.get_parameter_info()  # Fallback for compatibility
            else:
                params = {}
            if params and len(params) > 0:
                params_btn = QPushButton("⚙️ Configure")
                params_btn.setFixedWidth(120)
                params_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 12px;
                        font-size: 11px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #43A047;
                    }
                    QPushButton:pressed {
                        background-color: #2E7D32;
                    }
                """)
                params_btn.clicked.connect(lambda: self._show_custom_algorithm_parameters(name, algorithm))
                title_layout.addWidget(params_btn)
        except Exception as e:
            print(f"Warning: Could not get parameter info for {name}: {e}")
        
        algo_layout.addLayout(title_layout)
        
        # V1.3.1: Insert into Custom Tab's layout (at the front)
        self.custom_algos_layout.insertWidget(0, algo_frame)
        
        print(f"✅ Custom algorithm '{name}' added to Custom Tab")
    
    def _remove_custom_algorithm_ui(self, name: str):
        """Remove UI elements for a custom algorithm"""
        # Remove checkbox
        if name in self.custom_algo_checkboxes:
            checkbox = self.custom_algo_checkboxes[name]
            # Find and remove the parent frame
            frame = checkbox.parent()
            if frame:
                self.custom_algo_container.removeWidget(frame)
                frame.deleteLater()
            del self.custom_algo_checkboxes[name]
    
    def _show_custom_algorithm_parameters(self, name: str, algorithm):
        """Show parameter dialog for custom algorithm"""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} - Parameters")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # Get parameter info
        try:
            # V1.4.1: Fix method name - use get_params_info instead of get_parameter_info
            if hasattr(algorithm, 'get_params_info'):
                params = algorithm.get_params_info()
            elif hasattr(algorithm, 'get_parameter_info'):
                params = algorithm.get_parameter_info()  # Fallback for compatibility
            else:
                params = {}
            param_widgets = {}
            
            for param_name, param_info in params.items():
                param_type = param_info.get('type', 'float')
                default_value = param_info.get('default', 0)
                description = param_info.get('description', param_name)
                
                label = QLabel(f"{param_name}:")
                label.setToolTip(description)
                
                if param_type == 'int':
                    widget = QSpinBox()
                    widget.setRange(-999999, 999999)
                    widget.setValue(int(default_value))
                elif param_type == 'float':
                    widget = QDoubleSpinBox()
                    widget.setRange(-999999.0, 999999.0)
                    widget.setValue(float(default_value))
                    widget.setDecimals(4)
                elif param_type == 'bool':
                    widget = QCheckBox()
                    widget.setChecked(bool(default_value))
                else:
                    widget = QLineEdit(str(default_value))
                
                param_widgets[param_name] = widget
                form_layout.addRow(label, widget)
            
            layout.addLayout(form_layout)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            # Store parameter widgets for later retrieval
            if not hasattr(self, 'custom_algo_params'):
                self.custom_algo_params = {}
            self.custom_algo_params[name] = param_widgets
            
            dialog.exec_()
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Error', f'Failed to show parameters: {str(e)}')
    
    def get_selected_custom_algorithms(self) -> dict:
        """Get selected custom algorithms with their parameters"""
        selected = {}
        
        print(f"🔍 DEBUG: Checking {len(self.custom_algo_checkboxes)} custom algorithm checkboxes")
        print(f"🔍 DEBUG: Available algorithms: {list(self.custom_algorithms.keys())}")
        print(f"🔍 DEBUG: Available checkboxes: {list(self.custom_algo_checkboxes.keys())}")
        
        for name, checkbox in self.custom_algo_checkboxes.items():
            if checkbox is None:
                print(f"   ⚠️  {name}: checkbox is None!")
                continue
            is_checked = checkbox.isChecked()
            print(f"   - {name}: checkbox.isChecked() = {is_checked}, checkbox object = {type(checkbox).__name__}")
            if is_checked:
                algorithm = self.custom_algorithms.get(name)
                if algorithm:
                    print(f"   ✅ Found algorithm for {name}")
                    # Get parameters if they were set
                    params = {}
                    if hasattr(self, 'custom_algo_params') and name in self.custom_algo_params:
                        param_widgets = self.custom_algo_params[name]
                        for param_name, widget in param_widgets.items():
                            if isinstance(widget, QSpinBox):
                                params[param_name] = widget.value()
                            elif isinstance(widget, QDoubleSpinBox):
                                params[param_name] = widget.value()
                            elif isinstance(widget, QCheckBox):
                                params[param_name] = widget.isChecked()
                            elif isinstance(widget, QLineEdit):
                                params[param_name] = widget.text()
                    else:
                        # Use default parameters (but don't override if param already exists)
                        try:
                            # V1.4.1: Fix method name - use get_params_info instead of get_parameter_info
                            if hasattr(algorithm, 'get_params_info'):
                                param_info = algorithm.get_params_info()
                            elif hasattr(algorithm, 'get_parameter_info'):
                                param_info = algorithm.get_parameter_info()  # Fallback for compatibility
                            else:
                                param_info = {}
                            # Only set default if param doesn't exist (to allow global params to override)
                            for k, v in param_info.items():
                                if k not in params:
                                    params[k] = v.get('default', 0)
                        except Exception as e:
                            print(f"⚠️  Error getting default params for {name}: {e}")
                            # Keep params as is (may have global params)
                    
                    selected[name] = {
                        'algorithm': algorithm,
                        'params': params
                    }
        
        return selected

