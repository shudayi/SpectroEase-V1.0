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
from app.views.responsive_dialog import ResponsiveDialog

# Import design tokens for consistent UI
from app.config.ui_design_tokens import UIDesignTokens as DT
from app.utils.ui_helpers import create_primary_button, create_secondary_button
from app.utils.ui_scaling import ui_scaling_manager

class SpectralTypePreSelectionDialog(ResponsiveDialog):
    """Spectral Type Pre-selection Dialog"""
    
    def __init__(self, parent=None):
        # 优化对话框尺寸：减少不必要的高度
        super().__init__(parent, base_width=700, base_height=420)
        self.selected_type = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        self.setWindowTitle("Select Spectral Type")
        
        layout = QVBoxLayout()
        layout.setSpacing(DT.SPACING_STANDARD)  # 8px 标准间距
        layout.setContentsMargins(*DT.MARGIN_STANDARD)  # 8px 统一边距
        
        # 标题 - 使用响应式字体
        title_label = QLabel("🔬 Select Your Spectral Data Type")
        title_font = ui_scaling_manager.get_dynamic_font("Arial", 14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 8px;")  # 减少padding
        
        layout.addWidget(title_label)
        
        # 说明文字 - 使用响应式字体
        info_label = QLabel("Please select the type of spectral data you want to import.\nThis will optimize the analysis parameters automatically.")
        info_font = ui_scaling_manager.get_dynamic_font("Arial", 9)
        info_label.setFont(info_font)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #7f8c8d;")
        info_label.setWordWrap(True)  # 自动换行
        
        layout.addWidget(info_label)
        
        # 光谱类型选择组 - 使用设计令牌
        type_group = QGroupBox("Spectral Data Type")
        type_layout = QFormLayout()
        type_layout.setSpacing(DT.SPACING_TIGHT)  # 4px 紧凑间距
        type_layout.setContentsMargins(*DT.MARGIN_STANDARD)  # 8px 边距
        
        # 标签
        type_label = QLabel("Spectral Type:")
        type_label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)  # 110px
        
        # ComboBox使用响应式尺寸
        self.type_combo = QComboBox()
        self.type_combo.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)  # 24px
        self.type_combo.setMinimumWidth(DT.WIDTH_CONTROL_WIDE * 2)  # 440px，足够显示完整文本
        self.type_combo.addItem("NIR (Near-Infrared) - 800-2500 nm", "nir")
        self.type_combo.addItem("Raman - 200-4000 cm⁻¹", "raman")
        self.type_combo.addItem("MIR/FTIR (Mid-Infrared) - 2500-25000 nm", "mir")
        self.type_combo.addItem("Vis-NIR (Visible-NIR) - 400-1100 nm", "vis_nir")
        self.type_combo.addItem("UV-Vis (Ultraviolet-Visible) - 200-800 nm", "uv_vis")
        self.type_combo.addItem("Auto-detect from data", "auto")
        
        self.type_combo.setCurrentIndex(0)  # 默认选择NIR
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        
        type_layout.addRow(type_label, self.type_combo)
        type_group.setLayout(type_layout)
        
        layout.addWidget(type_group)
        
        # 描述区域 - 优化高度和样式
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold; color: #495057;")
        layout.addWidget(desc_label)
        
        self.description_text = QTextEdit()
        # 根据分辨率调整高度（从120降到100，更紧凑）
        screen_height = ui_scaling_manager.screen_info["height"]
        if screen_height >= 1440:  # 2K及以上
            desc_height = 120
        elif screen_height >= 1080:  # 1080p
            desc_height = 100
        else:  # 低分辨率
            desc_height = 90
        
        self.description_text.setFixedHeight(desc_height)
        self.description_text.setReadOnly(True)
        self.description_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 9pt;
            }
        """)
        
        layout.addWidget(self.description_text)
        
        # 添加弹性空间，确保按钮在底部
        layout.addStretch()
        
        # 按钮区域 - 使用辅助函数和设计令牌
        button_layout = QHBoxLayout()
        button_layout.setSpacing(DT.SPACING_STANDARD)  # 8px
        
        # 使用ui_helpers创建按钮
        self.cancel_btn = create_secondary_button("Cancel", callback=self.reject)
        self.continue_btn = create_primary_button("Continue to Select Data", callback=self.accept)
        
        # 设置按钮最小宽度
        self.cancel_btn.setMinimumWidth(DT.WIDTH_BUTTON_COMPACT)  # 90px
        self.continue_btn.setMinimumWidth(180)  # 主按钮稍宽
        
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