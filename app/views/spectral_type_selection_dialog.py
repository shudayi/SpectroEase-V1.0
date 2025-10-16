# app/views/spectral_type_selection_dialog.py
"""
Spectral Type Selection Dialog
Advanced spectral type configuration dialog
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QGroupBox, QTextEdit,
                           QFormLayout, QFrame, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QTabWidget, QWidget, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import numpy as np

class SpectralTypeSelectionDialog(QDialog):
    """Advanced Spectral Type Selection Dialog"""
    
    def __init__(self, data, wavelengths, parent=None):
        super().__init__(parent)
        self.data = data
        self.wavelengths = wavelengths
        self.selected_config = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        self.setWindowTitle("Select Your Spectral Data Type")
        self.setMinimumSize(700, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Select Your Spectral Data Type")
        title_font = QFont("Arial", 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px; padding: 10px;")
        
        layout.addWidget(title_label)
        
        # Auto-detection results and explanation
        auto_detection_group = QGroupBox("Automatic Detection Results")
        auto_layout = QVBoxLayout()
        
        # Auto-detection explanation
        explanation_text = QLabel("""
        <b>Why Manual Selection After Auto-Detection?</b><br>
        • <b>Overlapping Ranges:</b> Some spectral types have overlapping wavelength ranges<br>
        • <b>Instrument Variations:</b> Different instruments may use slightly different ranges<br>
        • <b>Preprocessing Optimization:</b> Each spectral type requires specific preprocessing methods<br>
        • <b>Chemical Context:</b> The same wavelength range can be used for different techniques
        """)
        explanation_text.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        explanation_text.setWordWrap(True)
        auto_layout.addWidget(explanation_text)
        
        # Auto-detected type display
        self.auto_detected_label = QLabel()
        self.auto_detected_label.setStyleSheet("font-weight: bold; color: #27ae60; padding: 5px;")
        auto_layout.addWidget(self.auto_detected_label)
        
        auto_detection_group.setLayout(auto_layout)
        layout.addWidget(auto_detection_group)
        
        # Manual selection
        selection_group = QGroupBox("Manual Selection & Expert Recommendations")
        selection_layout = QVBoxLayout()
        
        # Spectral type combo
        type_layout = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItem("NIR (Near-Infrared) - 800-2500 nm", "nir")
        self.type_combo.addItem("Raman - 200-4000 cm⁻¹", "raman") 
        self.type_combo.addItem("MIR/FTIR (Mid-Infrared) - 2500-25000 nm", "mir")
        self.type_combo.addItem("VIS-NIR (Visible-Near-Infrared) - 400-1100 nm", "vis_nir")
        self.type_combo.addItem("UV-VIS (Ultraviolet-Visible) - 200-800 nm", "uv_vis")
        self.type_combo.addItem("Custom Range", "custom")
        
        self.type_combo.currentTextChanged.connect(self.update_preprocessing_recommendations)
        type_layout.addRow("Spectral Type:", self.type_combo)
        selection_layout.addLayout(type_layout)
        
        # Expert preprocessing recommendations
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(200)
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        selection_layout.addWidget(QLabel("Expert Preprocessing Recommendations:"))
        selection_layout.addWidget(self.recommendations_text)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("Apply Configuration")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Perform auto-detection and set initial recommendations
        self.perform_auto_detection()
        self.update_preprocessing_recommendations()
    
    def create_basic_tab(self):
        """Create basic configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Spectral type selection
        type_group = QGroupBox("Spectral Type")
        type_layout = QFormLayout()
        
        self.type_combo = QComboBox()
        self.type_combo.addItem("NIR (Near-Infrared)", "nir")
        self.type_combo.addItem("Raman Spectroscopy", "raman")
        self.type_combo.addItem("MIR (Mid-Infrared)", "mir")
        self.type_combo.addItem("VIS-NIR (Visible-Near-Infrared)", "vis_nir")
        self.type_combo.addItem("UV-VIS (Ultraviolet-Visible)", "uv_vis")
        self.type_combo.addItem("Custom", "custom")
        
        type_layout.addRow("Spectral Type:", self.type_combo)
        type_group.setLayout(type_layout)
        
        # Adaptation level
        adapt_group = QGroupBox("Adaptation Level")
        adapt_layout = QFormLayout()
        
        self.adapt_combo = QComboBox()
        self.adapt_combo.addItem("Basic - Standard parameters", "basic")
        self.adapt_combo.addItem("Advanced - Optimized parameters", "advanced")
        self.adapt_combo.addItem("Expert - Custom parameters", "expert")
        
        adapt_layout.addRow("Adaptation Level:", self.adapt_combo)
        adapt_group.setLayout(adapt_layout)
        
        layout.addWidget(type_group)
        layout.addWidget(adapt_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_advanced_tab(self):
        """Create advanced configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Preprocessing options
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        
        self.snv_check = QCheckBox("SNV (Standard Normal Variate)")
        self.msc_check = QCheckBox("MSC (Multiplicative Scatter Correction)")
        self.baseline_check = QCheckBox("Baseline Correction")
        self.smooth_check = QCheckBox("Smoothing")
        self.derivative_check = QCheckBox("Derivative")
        
        preproc_layout.addWidget(self.snv_check)
        preproc_layout.addWidget(self.msc_check)
        preproc_layout.addWidget(self.baseline_check)
        preproc_layout.addWidget(self.smooth_check)
        preproc_layout.addWidget(self.derivative_check)
        
        preproc_group.setLayout(preproc_layout)
        
        # Parameter settings
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 51)
        self.window_spin.setValue(15)
        self.window_spin.setSingleStep(2)
        
        self.poly_spin = QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)
        
        param_layout.addRow("Smoothing Window:", self.window_spin)
        param_layout.addRow("Polynomial Order:", self.poly_spin)
        
        param_group.setLayout(param_layout)
        
        layout.addWidget(preproc_group)
        layout.addWidget(param_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def get_selected_config(self):
        """Get the selected configuration"""
        if self.selected_config is None:
            self.selected_config = self.build_config()
        return self.selected_config
    
    def build_config(self):
        """Build configuration from UI selections"""
        spectral_type = self.type_combo.currentData()
        spectral_name = self.type_combo.currentText()
        
        config = {
            "spectral_type": spectral_type,
            "spectral_name": spectral_name,
            "wavelength_range": [np.min(self.wavelengths), np.max(self.wavelengths)] if self.wavelengths is not None else None,
            "auto_detected": hasattr(self, 'auto_detected_label') and '✅' in self.auto_detected_label.text(),
            "preprocessing_config": {
                "scatter_correction": {
                    "enabled": True,
                    "method": "snv"
                },
                "baseline_correction": {
                    "enabled": True,
                    "method": "als"
                },
                "smoothing": {
                    "enabled": True,
                    "method": "savgol",
                    "window_length": 11,
                    "polyorder": 2
                },
                "derivative": {
                    "enabled": False,
                    "order": 1
                }
            },
            "parameter_adaptations": {}
        }
        
        return config
    
    def accept(self):
        """Handle accept button"""
        self.selected_config = self.build_config()
        super().accept()
    
    def perform_auto_detection(self):
        """Perform automatic spectral type detection based on wavelength analysis"""
        if self.wavelengths is None or len(self.wavelengths) == 0:
            self.auto_detected_label.setText("❌ Auto-detection failed: No wavelength data available")
            return
        
        wl_min, wl_max = np.min(self.wavelengths), np.max(self.wavelengths)
        wl_range = wl_max - wl_min
        
        # Expert spectral type detection logic
        detected_type = "unknown"
        confidence = "Low"
        reasons = []
        
        # Raman Detection (200-4000 cm⁻¹ or similar ranges)
        # Raman can be in wavenumbers (200-4000 cm⁻¹) or wavelength shifts
        if (200 <= wl_min <= 400 and 2000 <= wl_max <= 4000) or (wl_min < 500 and wl_max > 2000):
            detected_type = "raman"
            confidence = "High"
            reasons.append(f"Range {wl_min:.0f}-{wl_max:.0f} matches Raman spectroscopy (wavenumber or shift range)")
            
        # NIR Detection (800-2500 nm) - but not if it looks like Raman
        elif 700 <= wl_min <= 1000 and 2000 <= wl_max <= 3000 and not (wl_min < 500):
            detected_type = "nir"
            confidence = "High"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm matches NIR spectroscopy")
            
        # MIR/FTIR Detection (2500-25000 nm or 4000-400 cm⁻¹)
        elif wl_min >= 2000 and wl_max >= 10000:
            detected_type = "mir"
            confidence = "High"
            reasons.append(f"Long wavelength range {wl_min:.0f}-{wl_max:.0f} nm indicates MIR/FTIR")
            
        # VIS-NIR Detection (400-1100 nm)
        elif 300 <= wl_min <= 500 and 900 <= wl_max <= 1200:
            detected_type = "vis_nir"
            confidence = "Medium"
            reasons.append(f"Range {wl_min:.0f}-{wl_max:.0f} nm spans visible to near-infrared")
            
        # UV-VIS Detection (200-800 nm)
        elif wl_min <= 300 and 600 <= wl_max <= 900:
            detected_type = "uv_vis"
            confidence = "Medium"
            reasons.append(f"Range {wl_min:.0f}-{wl_max:.0f} nm covers UV-visible spectrum")
            
        # Ambiguous cases
        else:
            confidence = "Low"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm does not clearly match standard spectral types")
            reasons.append("Manual selection recommended for optimal preprocessing")
        
        # Update display
        if detected_type != "unknown":
            # Set the combo box to the detected type
            for i in range(self.type_combo.count()):
                if self.type_combo.itemData(i) == detected_type:
                    self.type_combo.setCurrentIndex(i)
                    break
            
            self.auto_detected_label.setText(
                f"✅ Auto-detected: {self.type_combo.currentText()} (Confidence: {confidence})\n" +
                f"📋 Detection reasons: {'; '.join(reasons)}"
            )
        else:
            self.auto_detected_label.setText(
                f"⚠️ Auto-detection uncertain (Confidence: {confidence})\n" +
                f"📋 Analysis: {'; '.join(reasons)}"
            )
    
    def update_preprocessing_recommendations(self):
        """Update expert preprocessing recommendations based on selected spectral type"""
        current_type = self.type_combo.currentData()
        
        # Expert recommendations from chemometrics perspective - 基于系统实际功能
        recommendations = {
            "nir": """
<b>NIR Spectroscopy Expert Recommendations:</b><br><br>

<b>Primary Preprocessing (Essential):</b><br>
• <b>Baseline Correction:</b> ALS method to remove baseline drift<br>
• <b>SNV (Standard Normal Variate):</b> Remove multiplicative scattering effects<br>
• <b>Savitzky-Golay Smoothing:</b> Reduce noise while preserving features<br><br>

<b>Secondary Preprocessing (Optional):</b><br>
• <b>Vector Normalization:</b> Final standardization step<br><br>

<b>Chemical Rationale:</b><br>
NIR spectra are dominated by overtones and combinations. Scattering correction (SNV) is 
critical for NIR. Despiking is NOT needed for NIR spectra as they don't have cosmic ray spikes.
            """,
            
            "raman": """
<b>Raman Spectroscopy Expert Recommendations:</b><br><br>

<b>Primary Preprocessing (Essential):</b><br>
• <b>Despiking (MAD method):</b> Remove cosmic ray spikes - CRITICAL for Raman<br>
• <b>Peak Alignment (DTW method):</b> Align peak positions - IMPORTANT for Raman precision<br>
• <b>Baseline Correction:</b> ALS method to remove fluorescence background<br>
• <b>Savitzky-Golay Smoothing:</b> Gentle noise reduction (small window)<br><br>

<b>Secondary Preprocessing (Optional):</b><br>
• <b>Area Normalization:</b> Normalize to total spectral area<br><br>

<b>Chemical Rationale:</b><br>
Raman spectra require despiking due to cosmic ray artifacts. Peak alignment is crucial 
because Raman peaks are sharp and position-sensitive. Scatter correction is NOT 
needed as Raman measures vibrational modes directly. Fluorescence background removal is critical.
            """,
            
            "mir": """
<b>MIR/FTIR Spectroscopy Expert Recommendations:</b><br><br>

<b>Primary Preprocessing (Essential):</b><br>
• <b>MSC (Multiplicative Scatter Correction):</b> Correct path length variations<br>
• <b>Baseline Correction:</b> ALS method for baseline drift<br>
• <b>First Derivative:</b> Enhance spectral features and reduce baseline effects<br>
• <b>Savitzky-Golay Smoothing:</b> Noise reduction<br><br>

<b>Secondary Preprocessing (Optional):</b><br>
• <b>Vector Normalization:</b> Final standardization<br><br>

<b>Missing but Important (Not in System):</b><br>
• <b>Atmospheric Correction:</b> CO₂ (2349 cm⁻¹) and H₂O correction needed<br><br>

<b>Chemical Rationale:</b><br>
MIR contains fundamental vibrations with high chemical specificity. Path length variations 
require MSC correction. Derivatives are crucial for MIR to enhance overlapping bands. 
Atmospheric interference is a major issue in MIR but not addressed in this system.
            """,
            
            "vis_nir": """
<b>VIS-NIR Spectroscopy Expert Recommendations:</b><br><br>

<b>Primary Preprocessing (Essential):</b><br>
• <b>SNV (Standard Normal Variate):</b> Correct scattering effects<br>
• <b>Baseline Correction:</b> ALS method for baseline drift<br>
• <b>Savitzky-Golay Smoothing:</b> Reduce detector noise<br><br>

<b>Secondary Preprocessing (Optional):</b><br>
• <b>Min-Max Normalization:</b> Scale to 0-1 range<br><br>

<b>Chemical Rationale:</b><br>
VIS-NIR combines electronic transitions (VIS) and overtones/combinations (NIR). 
Scattering correction is important for the NIR portion. ALS baseline correction 
is more robust than polynomial methods for this broad spectral range.
            """,
            
            "uv_vis": """
<b>UV-VIS Spectroscopy Expert Recommendations:</b><br><br>

<b>Primary Preprocessing (Essential):</b><br>
• <b>Baseline Correction:</b> Linear polynomial for drift correction<br><br>

<b>Secondary Preprocessing (Optional):</b><br>
• <b>Min-Max Normalization:</b> Scale for comparative analysis<br><br>

<b>Chemical Rationale:</b><br>
UV-VIS measures electronic transitions with high sensitivity. Simple baseline correction 
is usually sufficient. Beer-Lambert law applies directly for quantitative analysis.
            """,
            
            "custom": """
<b>Custom Range Expert Recommendations:</b><br><br>

<b>Available Methods in This System:</b><br>
• <b>Despiking:</b> MAD method (for Raman-like data)<br>
• <b>Baseline Correction:</b> ALS or Polynomial methods<br>
• <b>Scatter Correction:</b> SNV, MSC, EMSC, RNV, OSC<br>
• <b>Smoothing:</b> Savitzky-Golay, Moving Average, Gaussian<br>
• <b>Normalization:</b> Vector, Area, Min-Max, Standard scaling<br>
• <b>Derivatives:</b> First and Second derivatives<br><br>

<b>General Guidelines:</b><br>
Choose methods based on your specific spectroscopy technique and data characteristics.
Contact a spectroscopy expert for optimal preprocessing strategy.
            """
        }
        
        recommendation_text = recommendations.get(current_type, recommendations["custom"])
        self.recommendations_text.setHtml(recommendation_text)
