#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SNVparameters设置对话框
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QCheckBox, QDoubleSpinBox, QComboBox,
    QPushButton, QGroupBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np

class SNVParameterDialog(QDialog):
    """SNVparameters设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SNVparameters设置")
        self.setFixedSize(500, 600)
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
  
        title_label = QLabel("标准正态变量变换 (SNV) parameters设置")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        main_layout.addWidget(title_label)
        
  
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
  
        basic_group = QGroupBox("基本parameters")
        basic_layout = QFormLayout()
        
  
        self.center_check = QCheckBox()
        self.center_check.setChecked(True)
        basic_layout.addRow("中心化 (减去均值):", self.center_check)
        
  
        self.scale_check = QCheckBox()
        self.scale_check.setChecked(True)
        basic_layout.addRow("标准化 (除以标准差):", self.scale_check)
        
  
        self.min_std_spin = QDoubleSpinBox()
        self.min_std_spin.setRange(1e-10, 1e-3)
        self.min_std_spin.setValue(1e-6)
        self.min_std_spin.setDecimals(10)
        self.min_std_spin.setSingleStep(1e-7)
        basic_layout.addRow("最小标准差阈值:", self.min_std_spin)
        
  
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["按samples(推荐)", "按波长"])
        self.axis_combo.setCurrentIndex(0)
        basic_layout.addRow("processing方向:", self.axis_combo)
        
        basic_group.setLayout(basic_layout)
        main_layout.addWidget(basic_group)
        
  
        advanced_group = QGroupBox("高级parameters")
        advanced_layout = QFormLayout()
        
  
        self.copy_check = QCheckBox()
        self.copy_check.setChecked(True)
        advanced_layout.addRow("复制data (避免修改原始data):", self.copy_check)
        
  
        self.outlier_handling_combo = QComboBox()
        self.outlier_handling_combo.addItems(["忽略", "替换为阈值", "移除samples"])
        self.outlier_handling_combo.setCurrentIndex(1)
        advanced_layout.addRow("异常值processing:", self.outlier_handling_combo)
        
  
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(["浮点型", "双精度"])
        self.output_type_combo.setCurrentIndex(0)
        advanced_layout.addRow("输出data类型:", self.output_type_combo)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
  
        description_group = QGroupBox("算法说明")
        description_layout = QVBoxLayout()
        
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(120)
        self.description_text.setText(
            "SNV (标准正态变量变换) 是光谱预processing中重要的散射校正method：\n\n"
            "• 对每个光谱samples单独进行标准化processing\n"
            "• 消除乘性散射效应和基线漂移\n"
            "• 突出光谱的形状features，提高比较准确性\n"
            "• 特别适用于固体样品的漫反射光谱和近红外光谱\n\n"
            "变换公式: SNV(x) = (x - mean(x)) / std(x)"
        )
        description_layout.addWidget(self.description_text)
        
        description_group.setLayout(description_layout)
        main_layout.addWidget(description_group)
        
  
        preset_group = QGroupBox("预设配置")
        preset_layout = QHBoxLayout()
        
        self.standard_preset_btn = QPushButton("标准SNV")
        self.standard_preset_btn.clicked.connect(self.load_standard_preset)
        preset_layout.addWidget(self.standard_preset_btn)
        
        self.center_only_btn = QPushButton("仅中心化")
        self.center_only_btn.clicked.connect(self.load_center_only_preset)
        preset_layout.addWidget(self.center_only_btn)
        
        self.robust_btn = QPushButton("鲁棒模式")
        self.robust_btn.clicked.connect(self.load_robust_preset)
        preset_layout.addWidget(self.robust_btn)
        
        preset_group.setLayout(preset_layout)
        main_layout.addWidget(preset_group)
        
  
        button_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("测试parameters")
        self.test_btn.clicked.connect(self.test_parameters)
        button_layout.addWidget(self.test_btn)
        
        button_layout.addStretch()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
  
        self.center_check.toggled.connect(self.update_description)
        self.scale_check.toggled.connect(self.update_description)
        self.axis_combo.currentIndexChanged.connect(self.update_description)
        
    def load_standard_preset(self):
        """加载标准SNV预设"""
        self.center_check.setChecked(True)
        self.scale_check.setChecked(True)
        self.min_std_spin.setValue(1e-6)
        self.axis_combo.setCurrentIndex(0)
        self.copy_check.setChecked(True)
        self.outlier_handling_combo.setCurrentIndex(1)
        
    def load_center_only_preset(self):
        """加载仅中心化预设"""
        self.center_check.setChecked(True)
        self.scale_check.setChecked(False)
        self.min_std_spin.setValue(1e-6)
        self.axis_combo.setCurrentIndex(0)
        self.copy_check.setChecked(True)
        
    def load_robust_preset(self):
        """加载鲁棒模式预设"""
        self.center_check.setChecked(True)
        self.scale_check.setChecked(True)
        self.min_std_spin.setValue(1e-5)  # 更高的阈值
        self.axis_combo.setCurrentIndex(0)
        self.copy_check.setChecked(True)
        self.outlier_handling_combo.setCurrentIndex(2)  # 移除samples
        
    def update_description(self):
        """更新算法描述"""
        center = self.center_check.isChecked()
        scale = self.scale_check.isChecked()
        axis = self.axis_combo.currentIndex()
        
        desc = "SNV (标准正态变量变换) 是光谱预processing中重要的散射校正method：\n\n"
        
        if center and scale:
            desc += "• 完整SNV: 中心化 + 标准化\n"
            desc += "• 变换公式: SNV(x) = (x - mean(x)) / std(x)\n"
        elif center and not scale:
            desc += "• 仅中心化: 减去均值\n"
            desc += "• 变换公式: SNV(x) = x - mean(x)\n"
        elif not center and scale:
            desc += "• 仅标准化: 除以标准差 (不推荐)\n"
            desc += "• 变换公式: SNV(x) = x / std(x)\n"
        else:
            desc += "• 无变换: 返回原始data\n"
            
        if axis == 0:
            desc += "• 按samplesprocessing: 每个光谱单独标准化 (推荐)\n"
        else:
            desc += "• 按波长processing: 每个波长点跨samples标准化\n"
            
        desc += "\n效果: 消除散射效应、减少基线漂移、突出形状features"
        
        self.description_text.setText(desc)
        
    def test_parameters(self):
        """测试parameters"""
        from PyQt5.QtWidgets import QMessageBox
        
        params = self.get_parameters()
        
  
        np.random.seed(42)
        test_data = np.random.random((5, 10)) * 100 + 50
        
        try:
  
            if params['axis'] == 1:  # 按samples
                result_info = []
                for i in range(test_data.shape[0]):
                    spectrum = test_data[i, :]
                    mean_val = np.mean(spectrum) if params['center'] else 0
                    std_val = np.std(spectrum) if params['scale'] else 1
                    
                    if std_val < params['min_std']:
                        std_val = params['min_std']
                        
                    result_info.append(f"samples{i+1}: 均值={mean_val:.2f}, 标准差={std_val:.4f}")
                
                msg = "测试parameters预览:\n\n" + "\n".join(result_info)
            else:  # 按波长
                mean_vals = np.mean(test_data, axis=0) if params['center'] else np.zeros(test_data.shape[1])
                std_vals = np.std(test_data, axis=0) if params['scale'] else np.ones(test_data.shape[1])
                
                msg = f"测试parameters预览:\n\n波长均值范围: [{mean_vals.min():.2f}, {mean_vals.max():.2f}]\n"
                msg += f"波长标准差范围: [{std_vals.min():.4f}, {std_vals.max():.4f}]"
                
            QMessageBox.information(self, "parameters测试", msg)
            
        except Exception as e:
            QMessageBox.warning(self, "测试failed", f"parameters测试failed: {str(e)}")
            
    def get_parameters(self):
        """获取parameters"""
        return {
            'center': self.center_check.isChecked(),
            'scale': self.scale_check.isChecked(),
            'min_std': self.min_std_spin.value(),
            'axis': 1 if self.axis_combo.currentIndex() == 0 else 0,  # 0=按samples, 1=按波长
            'copy': self.copy_check.isChecked(),
            'outlier_handling': self.outlier_handling_combo.currentText(),
            'output_type': self.output_type_combo.currentText()
        }
        
    def set_parameters(self, params):
        """设置parameters"""
        self.center_check.setChecked(params.get('center', True))
        self.scale_check.setChecked(params.get('scale', True))
        self.min_std_spin.setValue(params.get('min_std', 1e-6))
        self.axis_combo.setCurrentIndex(0 if params.get('axis', 1) == 1 else 1)
        self.copy_check.setChecked(params.get('copy', True))
        
  
        self.update_description()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = SNVParameterDialog()
    dialog.show()
    sys.exit(app.exec_()) 