#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RNV (Robust Normal Variate) Parameter Dialog
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QCheckBox, QDoubleSpinBox, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np

class RNVParameterDialog(QDialog):
    """RNV Parameter Setting Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RNV Parameter Settings")
        self.setFixedSize(500, 620)
        self.init_ui()
        
    def init_ui(self):
        """Initialize interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Robust Normal Variate (RNV)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # Basic Parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout()
        
        # Centering method
        self.centering_combo = QComboBox()
        self.centering_combo.addItems(["Mean", "Median", "Trimmed Mean", "Huber"])
        self.centering_combo.setCurrentIndex(1)  # Default to Median
        basic_layout.addRow("Centering Method:", self.centering_combo)
        
        # Scaling method
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["Standard Deviation", "MAD", "IQR", "Robust Scale"])
        self.scaling_combo.setCurrentIndex(1)  # Default to MAD
        basic_layout.addRow("Scaling Method:", self.scaling_combo)
        
        # Outlier threshold
        self.outlier_threshold_spin = QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(1.0, 5.0)
        self.outlier_threshold_spin.setValue(2.5)
        self.outlier_threshold_spin.setDecimals(2)
        self.outlier_threshold_spin.setSingleStep(0.1)
        basic_layout.addRow("Outlier Threshold:", self.outlier_threshold_spin)
        
        # Robust estimation
        self.robust_check = QCheckBox()
        self.robust_check.setChecked(True)
        basic_layout.addRow("Enable Robust Estimation:", self.robust_check)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # Advanced Parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout()
        
        # Trimmed mean percentage (for trimmed mean centering)
        self.trim_percent_spin = QDoubleSpinBox()
        self.trim_percent_spin.setRange(0, 50)
        self.trim_percent_spin.setValue(10)
        self.trim_percent_spin.setSuffix(" %")
        advanced_layout.addRow("Trimmed Mean Percentage:", self.trim_percent_spin)
        
        # MAD constant (for MAD scaling)
        self.mad_constant_spin = QDoubleSpinBox()
        self.mad_constant_spin.setRange(0.1, 5.0)
        self.mad_constant_spin.setValue(1.4826)
        self.mad_constant_spin.setDecimals(4)
        self.mad_constant_spin.setSingleStep(0.1)
        advanced_layout.addRow("MAD Constant:", self.mad_constant_spin)
        
        # Minimum scale value
        self.min_scale_spin = QDoubleSpinBox()
        self.min_scale_spin.setRange(1e-10, 1e-2)
        self.min_scale_spin.setValue(1e-6)
        self.min_scale_spin.setDecimals(10)
        self.min_scale_spin.setSingleStep(1e-7)
        advanced_layout.addRow("Minimum Scale Value:", self.min_scale_spin)
        
        # Iterative refinement
        self.iterative_check = QCheckBox()
        self.iterative_check.setChecked(False)
        advanced_layout.addRow("Iterative Refinement:", self.iterative_check)
        
        # Maximum iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 100)
        self.max_iter_spin.setValue(10)
        self.max_iter_spin.setEnabled(False)
        advanced_layout.addRow("Maximum Iterations:", self.max_iter_spin)
        
        # Convergence tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-8, 1e-3)
        self.tolerance_spin.setValue(1e-5)
        self.tolerance_spin.setDecimals(8)
        self.tolerance_spin.setEnabled(False)
        advanced_layout.addRow("Convergence Tolerance:", self.tolerance_spin)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # Algorithm Description
        description_group = QGroupBox("Algorithm Description")
        description_layout = QVBoxLayout()
        
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(120)
        self.description_text.setText(
            "Robust Normal Variate (RNV) is a robust version of SNV:\n\n"
            "• Uses robust statistics (median, MAD) instead of mean and standard deviation\n"
            "• Less sensitive to outliers and extreme values\n"
            "• Maintains shape information while normalizing spectra\n"
            "• Particularly effective for noisy or contaminated spectra\n\n"
            "Formula: RNV(x) = (x - robust_center) / robust_scale"
        )
        description_layout.addWidget(self.description_text)
        
        description_group.setLayout(description_layout)
        main_layout.addWidget(description_group)
        
        # Preset Configurations
        preset_group = QGroupBox("Preset Configurations")
        preset_layout = QHBoxLayout()
        
        self.standard_preset_btn = QPushButton("Standard RNV")
        self.standard_preset_btn.clicked.connect(self.load_standard_preset)
        preset_layout.addWidget(self.standard_preset_btn)
        
        self.robust_preset_btn = QPushButton("Ultra Robust")
        self.robust_preset_btn.clicked.connect(self.load_robust_preset)
        preset_layout.addWidget(self.robust_preset_btn)
        
        self.adaptive_preset_btn = QPushButton("Adaptive RNV")
        self.adaptive_preset_btn.clicked.connect(self.load_adaptive_preset)
        preset_layout.addWidget(self.adaptive_preset_btn)
        
        preset_group.setLayout(preset_layout)
        main_layout.addWidget(preset_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test Parameters")
        self.test_btn.clicked.connect(self.test_parameters)
        button_layout.addWidget(self.test_btn)
        
        button_layout.addStretch()
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Connect signals
        self.centering_combo.currentIndexChanged.connect(self.update_description)
        self.scaling_combo.currentIndexChanged.connect(self.update_description)
        self.iterative_check.toggled.connect(self.on_iterative_changed)
        
    def on_iterative_changed(self, checked):
        """Handle iterative refinement checkbox"""
        self.max_iter_spin.setEnabled(checked)
        self.tolerance_spin.setEnabled(checked)
        
    def load_standard_preset(self):
        """Load standard RNV preset"""
        self.centering_combo.setCurrentIndex(1)  # Median
        self.scaling_combo.setCurrentIndex(1)    # MAD
        self.outlier_threshold_spin.setValue(2.5)
        self.robust_check.setChecked(True)
        self.trim_percent_spin.setValue(10)
        self.mad_constant_spin.setValue(1.4826)
        self.min_scale_spin.setValue(1e-6)
        self.iterative_check.setChecked(False)
        
    def load_robust_preset(self):
        """Load ultra robust preset"""
        self.centering_combo.setCurrentIndex(1)  # Median
        self.scaling_combo.setCurrentIndex(2)    # IQR
        self.outlier_threshold_spin.setValue(3.0)
        self.robust_check.setChecked(True)
        self.trim_percent_spin.setValue(20)
        self.min_scale_spin.setValue(1e-5)
        self.iterative_check.setChecked(True)
        self.max_iter_spin.setValue(15)
        self.tolerance_spin.setValue(1e-6)
        
    def load_adaptive_preset(self):
        """Load adaptive RNV preset"""
        self.centering_combo.setCurrentIndex(2)  # Trimmed Mean
        self.scaling_combo.setCurrentIndex(3)    # Robust Scale
        self.outlier_threshold_spin.setValue(2.0)
        self.robust_check.setChecked(True)
        self.trim_percent_spin.setValue(15)
        self.iterative_check.setChecked(True)
        self.max_iter_spin.setValue(20)
        self.tolerance_spin.setValue(1e-7)
        
    def update_description(self):
        """Update algorithm description"""
        centering = self.centering_combo.currentText()
        scaling = self.scaling_combo.currentText()
        
        desc = "Robust Normal Variate (RNV) configuration:\n\n"
        desc += f"• Centering: {centering}\n"
        desc += f"• Scaling: {scaling}\n"
        
        if centering == "Mean":
            desc += "• Centering mode: Standard (less robust)\n"
        elif centering == "Median":
            desc += "• Centering mode: Robust (recommended)\n"
        elif centering == "Trimmed Mean":
            desc += "• Centering mode: Semi-robust\n"
        elif centering == "Huber":
            desc += "• Centering mode: M-estimator\n"
            
        if scaling == "Standard Deviation":
            desc += "• Scaling mode: Standard (less robust)\n"
        elif scaling == "MAD":
            desc += "• Scaling mode: Very robust\n"
        elif scaling == "IQR":
            desc += "• Scaling mode: Robust\n"
        elif scaling == "Robust Scale":
            desc += "• Scaling mode: Adaptive robust\n"
            
        desc += "\nRNV provides robust normalization for contaminated spectra."
        
        self.description_text.setText(desc)
        
    def test_parameters(self):
        """Test parameters"""
        from PyQt5.QtWidgets import QMessageBox
        
        params = self.get_parameters()
        
        try:
            # Create test data with outliers
            np.random.seed(42)
            test_data = np.random.random((5, 10)) * 100 + 50
            # Add outliers
            test_data[0, 0] = 1000  # Outlier
            test_data[2, 5] = -500  # Outlier
            
            # Simulate RNV processing
            result_info = []
            for i in range(test_data.shape[0]):
                spectrum = test_data[i, :]
                
                if params['centering'] == "Mean":
                    center = np.mean(spectrum)
                elif params['centering'] == "Median":
                    center = np.median(spectrum)
                else:
                    center = np.mean(spectrum)  # Simplified
                    
                if params['scaling'] == "MAD":
                    scale = np.median(np.abs(spectrum - center)) * params['mad_constant']
                elif params['scaling'] == "Standard Deviation":
                    scale = np.std(spectrum)
                else:
                    scale = np.std(spectrum)  # Simplified
                    
                if scale < params['min_scale']:
                    scale = params['min_scale']
                    
                result_info.append(f"Sample {i+1}: Center={center:.2f}, Scale={scale:.4f}")
            
            msg = "RNV Parameter Test:\n\n" + "\n".join(result_info)
            msg += f"\n\nMethod: {params['centering']} + {params['scaling']}"
            msg += f"\nOutlier threshold: {params['outlier_threshold']}"
            
            QMessageBox.information(self, "Parameter Test", msg)
            
        except Exception as e:
            QMessageBox.warning(self, "Test Failed", f"Parameter test failed: {str(e)}")
            
    def get_parameters(self):
        """Get parameters"""
        return {
            'centering': self.centering_combo.currentText(),
            'scaling': self.scaling_combo.currentText(),
            'outlier_threshold': self.outlier_threshold_spin.value(),
            'robust_estimation': self.robust_check.isChecked(),
            'trim_percent': self.trim_percent_spin.value(),
            'mad_constant': self.mad_constant_spin.value(),
            'min_scale': self.min_scale_spin.value(),
            'iterative': self.iterative_check.isChecked(),
            'max_iter': self.max_iter_spin.value(),
            'tolerance': self.tolerance_spin.value()
        }
        
    def set_parameters(self, params):
        """Set parameters"""
        if 'centering' in params:
            index = self.centering_combo.findText(params['centering'])
            if index >= 0:
                self.centering_combo.setCurrentIndex(index)
                
        if 'scaling' in params:
            index = self.scaling_combo.findText(params['scaling'])
            if index >= 0:
                self.scaling_combo.setCurrentIndex(index)
        
        self.outlier_threshold_spin.setValue(params.get('outlier_threshold', 2.5))
        self.robust_check.setChecked(params.get('robust_estimation', True))
        self.trim_percent_spin.setValue(params.get('trim_percent', 10))
        self.mad_constant_spin.setValue(params.get('mad_constant', 1.4826))
        self.min_scale_spin.setValue(params.get('min_scale', 1e-6))
        self.iterative_check.setChecked(params.get('iterative', False))
        self.max_iter_spin.setValue(params.get('max_iter', 10))
        self.tolerance_spin.setValue(params.get('tolerance', 1e-5))
        
        # Update description
        self.update_description()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = RNVParameterDialog()
    dialog.show()
    sys.exit(app.exec_()) 