# app/views/spectral_type_pre_selection_dialog.py
"""
Spectral Type Pre-selection Dialog
Select spectral type before importing data
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QGroupBox, QTextEdit,
                           QFormLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

class SpectralTypePreSelectionDialog(QDialog):
    """Spectral Type Pre-selection Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_type = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        self.setWindowTitle("Select Spectral Type")
        self.setMinimumSize(500, 400)
        self.setMaximumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("🔬 Select Your Spectral Data Type")
        title_font = QFont("Arial", 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px; padding: 10px;")
        
        layout.addWidget(title_label)
        
        # Description text
        info_label = QLabel("Please select the type of spectral data you want to import.\nThis will optimize the analysis parameters automatically.")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        
        layout.addWidget(info_label)
        
        # Spectral type selection group
        type_group = QGroupBox("Spectral Data Type")
        type_layout = QFormLayout()
        
        self.type_combo = QComboBox()
        self.type_combo.addItem("NIR (Near-Infrared) - 800-2500 nm", "nir")
        self.type_combo.addItem("Raman - 200-4000 cm⁻¹", "raman")
        self.type_combo.addItem("MIR/FTIR (Mid-Infrared) - 2500-25000 nm", "mir")
        self.type_combo.addItem("Vis-NIR (Visible-NIR) - 400-1100 nm", "vis_nir")
        self.type_combo.addItem("UV-Vis (Ultraviolet-Visible) - 200-800 nm", "uv_vis")
        self.type_combo.addItem("Auto-detect from data", "auto")
        
        self.type_combo.setCurrentIndex(0)  # Default select NIR
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        
        type_layout.addRow("Spectral Type:", self.type_combo)
        type_group.setLayout(type_layout)
        
        layout.addWidget(type_group)
        
        # Description area
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(120)
        self.description_text.setReadOnly(True)
        self.description_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        layout.addWidget(QLabel("Description:"))
        layout.addWidget(self.description_text)
        
        # Button area
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.continue_btn = QPushButton("Continue to Select Data")
        self.continue_btn.clicked.connect(self.accept)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.continue_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Initialize description
        self.on_type_changed()
    
    def on_type_changed(self):
        """Spectral type change event"""
        current_data = self.type_combo.currentData()
        descriptions = {
            "nir": """<b>Near-Infrared Spectroscopy (NIR)</b><br>
            <b>Wavelength:</b> 800-2500 nm<br>
            <b>Applications:</b> Food analysis, agriculture, pharmaceuticals, petrochemicals<br>
            <b>Characteristics:</b> Non-destructive, rapid analysis, suitable for moisture and composition analysis<br>
            <b>Preprocessing:</b> Strong scatter correction (SNV/MSC) + 2nd derivative recommended""",
            
            "raman": """<b>Raman Spectroscopy</b><br>
            <b>Range:</b> 200-4000 cm⁻¹ Raman shift<br>
            <b>Applications:</b> Molecular structure, crystal analysis, chemical identification, biomedical<br>
            <b>Characteristics:</b> Molecular fingerprinting, minimal sample preparation<br>
            <b>Preprocessing:</b> Cosmic ray removal + baseline correction for fluorescence background""",
            
            "mir": """<b>Mid-Infrared / FTIR Spectroscopy</b><br>
            <b>Wavelength:</b> 2500-25000 nm (4000-400 cm⁻¹)<br>
            <b>Applications:</b> Organic compounds, functional groups, proteins, polymers<br>
            <b>Characteristics:</b> High chemical specificity, quantitative analysis<br>
            <b>Preprocessing:</b> Atmospheric correction (CO₂, H₂O) + MSC + normalization""",
            
            "vis_nir": """<b>Visible-Near Infrared Spectroscopy</b><br>
            <b>Wavelength:</b> 400-1100 nm<br>
            <b>Applications:</b> Color analysis, electronic transitions, quality control<br>
            <b>Characteristics:</b> Combines visible and NIR information<br>
            <b>Preprocessing:</b> Standard normalization + smoothing""",
            
            "uv_vis": """<b>Ultraviolet-Visible Spectroscopy</b><br>
            <b>Wavelength:</b> 200-800 nm<br>
            <b>Applications:</b> Concentration analysis, electronic transitions, chromophores<br>
            <b>Characteristics:</b> High sensitivity, quantitative analysis<br>
            <b>Preprocessing:</b> Baseline correction + normalization""",
            
            "auto": """<b>Auto-detection Mode</b><br>
            The system will automatically detect the spectral type based on:<br>
            • Wavelength range analysis<br>
            • Data characteristics<br>
            • Column naming patterns<br>
            <b>Note:</b> Manual selection is recommended for better accuracy"""
        }
        
        description = descriptions.get(current_data, "")
        self.description_text.setHtml(description)
    
    def get_selected_type(self):
        """Get selected spectral type"""
        return self.type_combo.currentData()
    
    def get_type_name(self):
        """Get spectral type name"""
        return self.type_combo.currentText().split(" - ")[0]