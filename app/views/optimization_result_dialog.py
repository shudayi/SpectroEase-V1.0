from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

class OptimizationResultDialog(QDialog):
    def __init__(self, parent, optimized_code):
        super().__init__(parent)
        self.optimized_code = optimized_code
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('AIoptimizationresults')
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
  
        self.code_edit = QTextEdit()
        self.code_edit.setPlainText(self.optimized_code)
        self.code_edit.setReadOnly(True)
        layout.addWidget(self.code_edit)
        
  
        button_layout = QHBoxLayout()
        
        apply_button = QPushButton('applying更改')
        apply_button.clicked.connect(self.apply_changes)
        
        cancel_button = QPushButton('取消')
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def apply_changes(self):
  
        self.accept() 