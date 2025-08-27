#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OSC (Orthogonal Signal Correction) Parameter Dialog
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QCheckBox, QDoubleSpinBox, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np

class OSCParameterDialog(QDialog):
    """OSC Parameter Setting Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OSC Parameter Settings")
        self.setFixedSize(520, 650)
        self.init_ui()
        
    def init_ui(self):
        """Initialize interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Orthogonal Signal Correction (OSC)")
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
        
        # Number of OSC components
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 20)
        self.n_components_spin.setValue(2)
        basic_layout.addRow("Number of OSC Components:", self.n_components_spin)
        
        # OSC algorithm
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Standard OSC", "Direct OSC", "Recursive OSC", "Consensus OSC"])
        self.algorithm_combo.setCurrentIndex(0)
        basic_layout.addRow("OSC Algorithm:", self.algorithm_combo)
        
        # Target tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-8, 1e-2)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setDecimals(8)
        self.tolerance_spin.setSingleStep(1e-7)
        basic_layout.addRow("Convergence Tolerance:", self.tolerance_spin)
        
        # Maximum iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(100)
        basic_layout.addRow("Maximum Iterations:", self.max_iter_spin)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # Advanced Parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout()
        
        # Deflation method
        self.deflation_combo = QComboBox()
        self.deflation_combo.addItems(["Classical", "Improved", "Symmetric", "NIPALS"])
        self.deflation_combo.setCurrentIndex(1)  # Improved
        advanced_layout.addRow("Deflation Method:", self.deflation_combo)
        
        # Variance threshold
        self.variance_threshold_spin = QDoubleSpinBox()
        self.variance_threshold_spin.setRange(0.01, 0.99)
        self.variance_threshold_spin.setValue(0.95)
        self.variance_threshold_spin.setDecimals(3)
        self.variance_threshold_spin.setSingleStep(0.01)
        advanced_layout.addRow("Variance Threshold:", self.variance_threshold_spin)
        
        # Center data
        self.center_check = QCheckBox()
        self.center_check.setChecked(True)
        advanced_layout.addRow("Center Data:", self.center_check)
        
        # Scale data
        self.scale_check = QCheckBox()
        self.scale_check.setChecked(False)
        advanced_layout.addRow("Scale Data:", self.scale_check)
        
        # Cross-validation
        self.cv_check = QCheckBox()
        self.cv_check.setChecked(True)
        advanced_layout.addRow("Cross-Validation:", self.cv_check)
        
        # CV folds
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 20)
        self.cv_folds_spin.setValue(5)
        advanced_layout.addRow("CV Folds:", self.cv_folds_spin)
        
        # Orthogonality constraint
        self.orthogonal_spin = QDoubleSpinBox()
        self.orthogonal_spin.setRange(0.0, 1.0)
        self.orthogonal_spin.setValue(1.0)
        self.orthogonal_spin.setDecimals(3)
        self.orthogonal_spin.setSingleStep(0.1)
        advanced_layout.addRow("Orthogonality Constraint:", self.orthogonal_spin)
        
        # Regularization
        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(0.0, 1.0)
        self.regularization_spin.setValue(0.0)
        self.regularization_spin.setDecimals(6)
        self.regularization_spin.setSingleStep(0.001)
        advanced_layout.addRow("Regularization Parameter:", self.regularization_spin)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # Algorithm Description
        description_group = QGroupBox("Algorithm Description")
        description_layout = QVBoxLayout()
        
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(120)
        self.description_text.setText(
            "Orthogonal Signal Correction (OSC) removes systematic variation orthogonal to Y:\n\n"
            "• Removes variation unrelated to the target variable\n"
            "• Improves prediction accuracy and model interpretability\n"
            "• Reduces model complexity and overfitting\n"
            "• Particularly effective for multivariate calibration\n\n"
            "Process: X_corrected = X - T_orth * P_orth^T (orthogonal to Y)"
        )
        description_layout.addWidget(self.description_text)
        
        description_group.setLayout(description_layout)
        main_layout.addWidget(description_group)
        
        # Preset Configurations
        preset_group = QGroupBox("Preset Configurations")
        preset_layout = QHBoxLayout()
        
        self.standard_preset_btn = QPushButton("Standard OSC")
        self.standard_preset_btn.clicked.connect(self.load_standard_preset)
        preset_layout.addWidget(self.standard_preset_btn)
        
        self.conservative_preset_btn = QPushButton("Conservative")
        self.conservative_preset_btn.clicked.connect(self.load_conservative_preset)
        preset_layout.addWidget(self.conservative_preset_btn)
        
        self.aggressive_preset_btn = QPushButton("Aggressive")
        self.aggressive_preset_btn.clicked.connect(self.load_aggressive_preset)
        preset_layout.addWidget(self.aggressive_preset_btn)
        
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
        self.algorithm_combo.currentIndexChanged.connect(self.update_description)
        self.deflation_combo.currentIndexChanged.connect(self.update_description)
        self.cv_check.toggled.connect(self.on_cv_changed)
        
    def on_cv_changed(self, checked):
        """Handle cross-validation checkbox"""
        self.cv_folds_spin.setEnabled(checked)
        
    def load_standard_preset(self):
        """Load standard OSC preset"""
        self.n_components_spin.setValue(2)
        self.algorithm_combo.setCurrentIndex(0)  # Standard OSC
        self.tolerance_spin.setValue(1e-6)
        self.max_iter_spin.setValue(100)
        self.deflation_combo.setCurrentIndex(1)  # Improved
        self.variance_threshold_spin.setValue(0.95)
        self.center_check.setChecked(True)
        self.scale_check.setChecked(False)
        self.cv_check.setChecked(True)
        self.cv_folds_spin.setValue(5)
        self.orthogonal_spin.setValue(1.0)
        self.regularization_spin.setValue(0.0)
        
    def load_conservative_preset(self):
        """Load conservative OSC preset"""
        self.n_components_spin.setValue(1)
        self.algorithm_combo.setCurrentIndex(0)  # Standard OSC
        self.tolerance_spin.setValue(1e-5)
        self.max_iter_spin.setValue(50)
        self.deflation_combo.setCurrentIndex(0)  # Classical
        self.variance_threshold_spin.setValue(0.99)
        self.center_check.setChecked(True)
        self.scale_check.setChecked(True)
        self.cv_check.setChecked(True)
        self.cv_folds_spin.setValue(10)
        self.orthogonal_spin.setValue(0.9)
        self.regularization_spin.setValue(0.01)
        
    def load_aggressive_preset(self):
        """Load aggressive OSC preset"""
        self.n_components_spin.setValue(5)
        self.algorithm_combo.setCurrentIndex(2)  # Recursive OSC
        self.tolerance_spin.setValue(1e-7)
        self.max_iter_spin.setValue(200)
        self.deflation_combo.setCurrentIndex(2)  # Symmetric
        self.variance_threshold_spin.setValue(0.90)
        self.center_check.setChecked(True)
        self.scale_check.setChecked(False)
        self.cv_check.setChecked(True)
        self.cv_folds_spin.setValue(3)
        self.orthogonal_spin.setValue(1.0)
        self.regularization_spin.setValue(0.0)
        
    def update_description(self):
        """Update algorithm description"""
        algorithm = self.algorithm_combo.currentText()
        deflation = self.deflation_combo.currentText()
        
        desc = "Orthogonal Signal Correction (OSC) configuration:\n\n"
        desc += f"• Algorithm: {algorithm}\n"
        desc += f"• Deflation: {deflation}\n"
        
        if algorithm == "Standard OSC":
            desc += "• Mode: Classical OSC implementation\n"
        elif algorithm == "Direct OSC":
            desc += "• Mode: Direct calculation (faster)\n"
        elif algorithm == "Recursive OSC":
            desc += "• Mode: Iterative refinement\n"
        elif algorithm == "Consensus OSC":
            desc += "• Mode: Ensemble approach\n"
            
        if deflation == "Classical":
            desc += "• Deflation: Traditional method\n"
        elif deflation == "Improved":
            desc += "• Deflation: Enhanced stability\n"
        elif deflation == "Symmetric":
            desc += "• Deflation: Symmetric approach\n"
        elif deflation == "NIPALS":
            desc += "• Deflation: NIPALS algorithm\n"
            
        desc += "\nOSC removes unwanted systematic variation from spectra."
        
        self.description_text.setText(desc)
        
    def test_parameters(self):
        """Test parameters"""
        from PyQt5.QtWidgets import QMessageBox
        
        params = self.get_parameters()
        
        try:
            # Validate parameters
            if params['n_components'] > 10 and params['variance_threshold'] > 0.95:
                QMessageBox.warning(self, "Parameter Warning", 
                                  "High number of components with high variance threshold may remove too much information.")
                return
                
            if params['regularization'] > 0.1:
                QMessageBox.warning(self, "Parameter Warning", 
                                  "High regularization may over-smooth the data.")
                return
                
            # Simulate OSC processing info
            msg = f"OSC Parameter Test:\n\n"
            msg += f"Algorithm: {params['algorithm']}\n"
            msg += f"Components: {params['n_components']}\n"
            msg += f"Deflation: {params['deflation']}\n"
            msg += f"Variance threshold: {params['variance_threshold']:.3f}\n"
            msg += f"Orthogonality: {params['orthogonal_constraint']:.3f}\n"
            msg += f"Regularization: {params['regularization']:.6f}\n\n"
            msg += "Parameters are valid for OSC processing."
            
            QMessageBox.information(self, "Parameter Test", msg)
            
        except Exception as e:
            QMessageBox.warning(self, "Test Failed", f"Parameter test failed: {str(e)}")
            
    def get_parameters(self):
        """Get parameters"""
        return {
            'n_components': self.n_components_spin.value(),
            'algorithm': self.algorithm_combo.currentText(),
            'tolerance': self.tolerance_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'deflation': self.deflation_combo.currentText(),
            'variance_threshold': self.variance_threshold_spin.value(),
            'center': self.center_check.isChecked(),
            'scale': self.scale_check.isChecked(),
            'cross_validation': self.cv_check.isChecked(),
            'cv_folds': self.cv_folds_spin.value(),
            'orthogonal_constraint': self.orthogonal_spin.value(),
            'regularization': self.regularization_spin.value()
        }
        
    def set_parameters(self, params):
        """Set parameters"""
        self.n_components_spin.setValue(params.get('n_components', 2))
        
        if 'algorithm' in params:
            index = self.algorithm_combo.findText(params['algorithm'])
            if index >= 0:
                self.algorithm_combo.setCurrentIndex(index)
                
        self.tolerance_spin.setValue(params.get('tolerance', 1e-6))
        self.max_iter_spin.setValue(params.get('max_iter', 100))
        
        if 'deflation' in params:
            index = self.deflation_combo.findText(params['deflation'])
            if index >= 0:
                self.deflation_combo.setCurrentIndex(index)
                
        self.variance_threshold_spin.setValue(params.get('variance_threshold', 0.95))
        self.center_check.setChecked(params.get('center', True))
        self.scale_check.setChecked(params.get('scale', False))
        self.cv_check.setChecked(params.get('cross_validation', True))
        self.cv_folds_spin.setValue(params.get('cv_folds', 5))
        self.orthogonal_spin.setValue(params.get('orthogonal_constraint', 1.0))
        self.regularization_spin.setValue(params.get('regularization', 0.0))
        
        # Update description
        self.update_description()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = OSCParameterDialog()
    dialog.show()
    sys.exit(app.exec_()) 