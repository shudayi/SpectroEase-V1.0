#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EMSC (Extended Multiplicative Scatter Correction) Parameter Dialog
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QCheckBox, QDoubleSpinBox, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
from app.views.responsive_dialog import ResponsiveDialog

class EMSCParameterDialog(ResponsiveDialog):
    """EMSC Parameter Setting Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent, base_width=520, base_height=650)
        self.setWindowTitle("EMSC Parameter Settings")
        self.init_ui()
        
    def init_ui(self):
        """Initialize interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Extended Multiplicative Scatter Correction (EMSC)")
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
        
        # Reference spectrum selection
        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["Mean Spectrum", "Median Spectrum", "First Sample", "Custom Reference"])
        self.reference_combo.setCurrentIndex(0)
        basic_layout.addRow("Reference Spectrum:", self.reference_combo)
        
        # Polynomial order
        self.poly_order_spin = QSpinBox()
        self.poly_order_spin.setRange(0, 5)
        self.poly_order_spin.setValue(2)
        basic_layout.addRow("Polynomial Order:", self.poly_order_spin)
        
        # Include wavelength terms
        self.wavelength_check = QCheckBox()
        self.wavelength_check.setChecked(True)
        basic_layout.addRow("Include Wavelength Terms:", self.wavelength_check)
        
        # Include pathlength variation
        self.pathlength_check = QCheckBox()
        self.pathlength_check.setChecked(True)
        basic_layout.addRow("Include Pathlength Variation:", self.pathlength_check)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
        # Advanced Parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout()
        
        # Wavelength range
        self.wavelength_start_spin = QDoubleSpinBox()
        self.wavelength_start_spin.setRange(0, 10000)
        self.wavelength_start_spin.setValue(400)
        self.wavelength_start_spin.setSuffix(" nm")
        advanced_layout.addRow("Wavelength Start:", self.wavelength_start_spin)
        
        self.wavelength_end_spin = QDoubleSpinBox()
        self.wavelength_end_spin.setRange(0, 10000)
        self.wavelength_end_spin.setValue(2500)
        self.wavelength_end_spin.setSuffix(" nm")
        advanced_layout.addRow("Wavelength End:", self.wavelength_end_spin)
        
        # Interference matrix
        self.interference_check = QCheckBox()
        self.interference_check.setChecked(False)
        advanced_layout.addRow("Use Interference Matrix:", self.interference_check)
        
        # Regularization parameter
        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(0, 1)
        self.regularization_spin.setValue(0.01)
        self.regularization_spin.setDecimals(6)
        self.regularization_spin.setSingleStep(0.001)
        advanced_layout.addRow("Regularization Parameter:", self.regularization_spin)
        
        # Maximum iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(100)
        advanced_layout.addRow("Maximum Iterations:", self.max_iter_spin)
        
        # Convergence tolerance
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-10, 1e-2)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.setSingleStep(1e-7)
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
            "Extended Multiplicative Scatter Correction (EMSC) is an advanced scatter correction method:\n\n"
            "• Extends MSC by including polynomial terms for wavelength dependence\n"
            "• Accounts for pathlength variations and chemical interferences\n"
            "• Uses orthogonal signal correction principles\n"
            "• Particularly effective for complex scatter patterns\n\n"
            "Model: X_corrected = (X - interference - polynomial) / scatter_factor"
        )
        description_layout.addWidget(self.description_text)
        
        description_group.setLayout(description_layout)
        main_layout.addWidget(description_group)
        
        # Preset Configurations
        preset_group = QGroupBox("Preset Configurations")
        preset_layout = QHBoxLayout()
        
        self.standard_preset_btn = QPushButton("Standard EMSC")
        self.standard_preset_btn.clicked.connect(self.load_standard_preset)
        preset_layout.addWidget(self.standard_preset_btn)
        
        self.simple_preset_btn = QPushButton("Simple EMSC")
        self.simple_preset_btn.clicked.connect(self.load_simple_preset)
        preset_layout.addWidget(self.simple_preset_btn)
        
        self.advanced_preset_btn = QPushButton("Advanced EMSC")
        self.advanced_preset_btn.clicked.connect(self.load_advanced_preset)
        preset_layout.addWidget(self.advanced_preset_btn)
        
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
        self.reference_combo.currentIndexChanged.connect(self.update_description)
        self.poly_order_spin.valueChanged.connect(self.update_description)
        self.wavelength_check.toggled.connect(self.update_description)
        
    def load_standard_preset(self):
        """Load standard EMSC preset"""
        self.reference_combo.setCurrentIndex(0)  # Mean Spectrum
        self.poly_order_spin.setValue(2)
        self.wavelength_check.setChecked(True)
        self.pathlength_check.setChecked(True)
        self.interference_check.setChecked(False)
        self.regularization_spin.setValue(0.01)
        self.max_iter_spin.setValue(100)
        self.tolerance_spin.setValue(1e-6)
        
    def load_simple_preset(self):
        """Load simple EMSC preset"""
        self.reference_combo.setCurrentIndex(0)  # Mean Spectrum
        self.poly_order_spin.setValue(1)
        self.wavelength_check.setChecked(False)
        self.pathlength_check.setChecked(True)
        self.interference_check.setChecked(False)
        self.regularization_spin.setValue(0.001)
        
    def load_advanced_preset(self):
        """Load advanced EMSC preset"""
        self.reference_combo.setCurrentIndex(0)  # Mean Spectrum
        self.poly_order_spin.setValue(3)
        self.wavelength_check.setChecked(True)
        self.pathlength_check.setChecked(True)
        self.interference_check.setChecked(True)
        self.regularization_spin.setValue(0.1)
        self.max_iter_spin.setValue(200)
        self.tolerance_spin.setValue(1e-8)
        
    def update_description(self):
        """Update algorithm description"""
        reference = self.reference_combo.currentText()
        poly_order = self.poly_order_spin.value()
        wavelength = self.wavelength_check.isChecked()
        
        desc = "Extended Multiplicative Scatter Correction (EMSC) configuration:\n\n"
        desc += f"• Reference: {reference}\n"
        desc += f"• Polynomial order: {poly_order}\n"
        desc += f"• Wavelength terms: {'Included' if wavelength else 'Excluded'}\n"
        
        if poly_order == 0:
            desc += "• Mode: Basic MSC (no polynomial terms)\n"
        elif poly_order == 1:
            desc += "• Mode: Linear correction\n"
        elif poly_order >= 2:
            desc += "• Mode: Non-linear correction\n"
            
        desc += "\nEMSC effectively removes both multiplicative and additive scatter effects."
        
        self.description_text.setText(desc)
        
    def test_parameters(self):
        """Test parameters"""
        from PyQt5.QtWidgets import QMessageBox
        
        params = self.get_parameters()
        
        try:
            # Validate parameters
            if params['poly_order'] > 3 and params['regularization'] < 0.001:
                QMessageBox.warning(self, "Parameter Warning", 
                                  "High polynomial order with low regularization may cause overfitting.")
                return
                
            if params['wavelength_start'] >= params['wavelength_end']:
                QMessageBox.warning(self, "Parameter Error", 
                                  "Wavelength start must be less than wavelength end.")
                return
                
            # Simulate EMSC processing info
            msg = f"EMSC Parameter Test:\n\n"
            msg += f"Reference: {params['reference']}\n"
            msg += f"Polynomial order: {params['poly_order']}\n"
            msg += f"Wavelength range: {params['wavelength_start']:.0f}-{params['wavelength_end']:.0f} nm\n"
            msg += f"Regularization: {params['regularization']:.6f}\n"
            msg += f"Max iterations: {params['max_iter']}\n\n"
            msg += "Parameters are valid for EMSC processing."
            
            QMessageBox.information(self, "Parameter Test", msg)
            
        except Exception as e:
            QMessageBox.warning(self, "Test Failed", f"Parameter test failed: {str(e)}")
            
    def get_parameters(self):
        """Get parameters"""
        return {
            'reference': self.reference_combo.currentText(),
            'poly_order': self.poly_order_spin.value(),
            'include_wavelength': self.wavelength_check.isChecked(),
            'include_pathlength': self.pathlength_check.isChecked(),
            'wavelength_start': self.wavelength_start_spin.value(),
            'wavelength_end': self.wavelength_end_spin.value(),
            'use_interference': self.interference_check.isChecked(),
            'regularization': self.regularization_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'tolerance': self.tolerance_spin.value()
        }
        
    def set_parameters(self, params):
        """Set parameters"""
        if 'reference' in params:
            index = self.reference_combo.findText(params['reference'])
            if index >= 0:
                self.reference_combo.setCurrentIndex(index)
        
        self.poly_order_spin.setValue(params.get('poly_order', 2))
        self.wavelength_check.setChecked(params.get('include_wavelength', True))
        self.pathlength_check.setChecked(params.get('include_pathlength', True))
        self.wavelength_start_spin.setValue(params.get('wavelength_start', 400))
        self.wavelength_end_spin.setValue(params.get('wavelength_end', 2500))
        self.interference_check.setChecked(params.get('use_interference', False))
        self.regularization_spin.setValue(params.get('regularization', 0.01))
        self.max_iter_spin.setValue(params.get('max_iter', 100))
        self.tolerance_spin.setValue(params.get('tolerance', 1e-6))
        
        # Update description
        self.update_description()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = EMSCParameterDialog()
    dialog.show()
    sys.exit(app.exec_()) 