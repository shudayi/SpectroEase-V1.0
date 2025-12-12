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
from app.views.responsive_dialog import ResponsiveDialog
import numpy as np

class SpectralTypeSelectionDialog(ResponsiveDialog):
    """Advanced Spectral Type Selection Dialog"""
    
    def __init__(self, data, wavelengths, parent=None):
        super().__init__(parent, base_width=700, base_height=600)
        self.data = data
        self.wavelengths = wavelengths
        self.selected_config = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        self.setWindowTitle("Select Your Spectral Data Type")
        
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
        <b>Manual confirmation required:</b><br>
        Wavelength ranges may overlap between techniques. Confirm spectral type for optimal preprocessing.
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
            "auto_detected": hasattr(self, 'auto_detected_label') and 'Auto-detected' in self.auto_detected_label.text(),
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
            self.auto_detected_label.setText("Auto-detection failed: No wavelength data available")
            return
        
        wl_min, wl_max = np.min(self.wavelengths), np.max(self.wavelengths)
        wl_range = wl_max - wl_min
        
        # Expert spectral type detection logic based on standard spectroscopy ranges
        detected_type = "unknown"
        confidence = "Low"
        reasons = []
        
        # Raman Detection (200-4000 cm⁻¹)
        # Raman spectroscopy uses wavenumber (cm⁻¹), typical range 200-4000 cm⁻¹
        # Most common range: 400-3200 cm⁻¹ for molecular fingerprinting
        if (200 <= wl_min <= 500 and 2500 <= wl_max <= 4200):
            detected_type = "raman"
            confidence = "High"
            reasons.append(f"Wavenumber range {wl_min:.0f}-{wl_max:.0f} cm⁻¹ matches standard Raman spectroscopy range")
            
        # NIR Detection (800-2500 nm)
        # Near-infrared spectroscopy: overtones and combinations of C-H, O-H, N-H bonds
        # Standard range: 780-2500 nm, most instruments use 1000-2500 nm
        elif 750 <= wl_min <= 950 and 2200 <= wl_max <= 2600:
            detected_type = "nir"
            confidence = "High"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm matches standard NIR range (800-2500 nm)")
            
        # MIR/FTIR Detection (2500-25000 nm or 4000-400 cm⁻¹)
        # Mid-infrared: fundamental vibrations, highly specific for functional groups
        # Standard wavelength: 2500-25000 nm, or wavenumber: 4000-400 cm⁻¹
        elif wl_min >= 2300 and wl_max >= 8000:
            detected_type = "mir"
            confidence = "High"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm matches standard MIR/FTIR range")
            
        # VIS-NIR Detection (400-1100 nm)
        # Visible-Near-Infrared: combines electronic transitions and some overtones
        # Typical for agricultural and food applications
        elif 350 <= wl_min <= 500 and 1000 <= wl_max <= 1200:
            detected_type = "vis_nir"
            confidence = "High"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm matches standard VIS-NIR range (400-1100 nm)")
            
        # UV-VIS Detection (200-800 nm)
        # Ultraviolet-Visible: electronic transitions, Beer-Lambert law applies
        # Standard range: 190-800 nm for most spectrophotometers
        elif wl_min <= 350 and 500 <= wl_max <= 900:
            detected_type = "uv_vis"
            confidence = "High"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} nm matches standard UV-VIS range (200-800 nm)")
            
        # Ambiguous cases - provide guidance
        else:
            confidence = "Low"
            reasons.append(f"Wavelength range {wl_min:.0f}-{wl_max:.0f} does not clearly match standard spectral types")
            reasons.append("Manual selection recommended for optimal preprocessing")
        
        # Update display (removed all icons)
        if detected_type != "unknown":
            # Set the combo box to the detected type
            for i in range(self.type_combo.count()):
                if self.type_combo.itemData(i) == detected_type:
                    self.type_combo.setCurrentIndex(i)
                    break
            
            self.auto_detected_label.setText(
                f"Auto-detected: {self.type_combo.currentText()} (Confidence: {confidence})\n" +
                f"Detection basis: {'; '.join(reasons)}"
            )
        else:
            self.auto_detected_label.setText(
                f"Auto-detection uncertain (Confidence: {confidence})\n" +
                f"Analysis: {'; '.join(reasons)}"
            )
    
    def update_preprocessing_recommendations(self):
        """Update expert preprocessing recommendations based on selected spectral type"""
        current_type = self.type_combo.currentData()
        
        # Preprocessing recommendations based on spectroscopy principles
        recommendations = {
            "nir": """
<b>NIR Preprocessing (800-2500 nm):</b><br><br>

<b>Required:</b><br>
- Baseline correction (ALS) - removes drift from instrument and sample<br>
- SNV or MSC - corrects multiplicative scatter effects<br>
- Savitzky-Golay smoothing (window 9-15, order 2-3) - reduces noise<br><br>

<b>Optional:</b><br>
- Vector normalization - final intensity scaling<br>
- 1st derivative - enhances spectral features, reduces baseline<br>
            """,
            
            "raman": """
<b>Raman Preprocessing (200-4000 cm⁻¹):</b><br><br>

<b>Required:</b><br>
- Despiking (MAD) - removes cosmic ray artifacts (essential)<br>
- Baseline correction (ModPoly or ALS) - removes fluorescence<br>
- Smoothing (window 5-9, order 2) - reduces CCD noise<br><br>

<b>Recommended:</b><br>
- Peak alignment (DTW) - corrects laser wavelength variations<br>
- Area normalization - compensates for sample concentration<br><br>

<b>Note:</b> Process in this order to avoid spreading cosmic ray artifacts.
            """,
            
            "mir": """
<b>MIR/FTIR Preprocessing (2500-25000 nm / 4000-400 cm⁻¹):</b><br><br>

<b>Required:</b><br>
- MSC or SNV - corrects pathlength and scatter variations<br>
- Baseline correction - removes interferometer artifacts<br>
- Smoothing (window 9-15) - reduces detector noise<br><br>

<b>Optional:</b><br>
- 1st derivative - resolves overlapping bands<br>
- 2nd derivative - sharper resolution (increases noise)<br>
- Vector normalization - final scaling<br><br>

Atmospheric interference: CO₂ at 2349 cm⁻¹, H₂O at 3500-4000 and 1300-2000 cm⁻¹. 
Background subtraction recommended.
            """,
            
            "vis_nir": """
<b>VIS-NIR Preprocessing (400-1100 nm):</b><br><br>

<b>Required:</b><br>
- SNV - scatter correction<br>
- Baseline correction (ALS) - removes drift<br>
- Smoothing (window 7-11) - reduces photodiode noise<br><br>

<b>Optional:</b><br>
- Min-max normalization - scales to 0-1 range<br>
- Detrending - removes long-wavelength tilt<br>
            """,
            
            "uv_vis": """
<b>UV-VIS Preprocessing (200-800 nm):</b><br><br>

<b>Typical:</b><br>
- Baseline correction (linear or 2nd order polynomial)<br>
- Min-max normalization (for comparative analysis)<br><br>

<b>Note:</b> UV-VIS spectra usually have high SNR. 
Minimal preprocessing recommended to avoid distortion.
            """,
            
            "custom": """
<b>Custom Range Preprocessing:</b><br><br>

<b>Available methods:</b><br>
- Baseline: ALS, Polynomial, ModPoly<br>
- Scatter: SNV, MSC, EMSC, RNV, OSC<br>
- Smoothing: Savitzky-Golay, Moving Average, Gaussian<br>
- Derivatives: 1st, 2nd order<br>
- Normalization: Vector, Area, Min-Max, Standard<br>
- Alignment: DTW (peak alignment)<br>
- Raman-specific: Despiking (MAD)<br><br>

Select methods based on your spectroscopy technique.
Consult instrument documentation for optimal workflow.
            """
        }
        
        recommendation_text = recommendations.get(current_type, recommendations["custom"])
        self.recommendations_text.setHtml(recommendation_text)
