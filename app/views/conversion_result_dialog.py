from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, 
                            QPushButton, QHBoxLayout, QLabel)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

class ConversionResultDialog(QDialog):
    def __init__(self, parent, converted_code: str, algorithm_type: str):
        super().__init__(parent)
        self.converted_code = converted_code
        self.algorithm_type = algorithm_type
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f'Code Conversion Result - {self.algorithm_type}')
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # Result description
        result_label = QLabel('Converted Code:')
        result_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(result_label)
        
        # Code display area
        self.code_edit = QTextEdit()
        self.code_edit.setPlainText(self.converted_code)
        self.code_edit.setReadOnly(True)
        self.code_edit.setFont(QFont("Consolas", 10))
        self.code_edit.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 10px;")
        layout.addWidget(self.code_edit)
        
        # Tip information
        tip_label = QLabel('Tip: Please check if the code meets your requirements before clicking "Apply to System"')
        tip_label.setStyleSheet("color: #666; font-style: italic; margin-top: 5px;")
        layout.addWidget(tip_label)
        
        # Button area
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton('Apply to System')
        self.apply_button.setStyleSheet("padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px;")
        self.apply_button.clicked.connect(self.accept)
        
        self.copy_button = QPushButton('Copy Code')
        self.copy_button.setStyleSheet("padding: 8px 15px; background-color: #2196F3; color: white; border: none; border-radius: 4px;")
        self.copy_button.clicked.connect(self.copy_code)
        
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setStyleSheet("padding: 8px 15px; background-color: #f44336; color: white; border: none; border-radius: 4px;")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.setContentsMargins(0, 15, 0, 0)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def copy_code(self):
        """
        将conversion后的代码复制到剪贴板
        """
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self.converted_code)
  
        self.copy_button.setText("复制success!")
        self.copy_button.setStyleSheet("padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px;")
  
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self.reset_copy_button())
        
    def reset_copy_button(self):
        """恢复复制按钮的状态"""
        self.copy_button.setText("复制代码")
        self.copy_button.setStyleSheet("padding: 8px 15px; background-color: #2196F3; color: white; border: none; border-radius: 4px;") 