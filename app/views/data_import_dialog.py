from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QFileDialog
)
from app.views.responsive_dialog import ResponsiveDialog

class DataImportDialog(ResponsiveDialog):
    """Data Import Dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent, base_width=800, base_height=600)
        self.file_path = None
        self.format_type = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Import Data")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("File Format:"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "TXT", "XLSX"])
        format_layout.addWidget(self.format_combo)
        
        layout.addLayout(format_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def browse_file(self):
        """Browse file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(file_path)
            
            # Set format based on file extension
            ext = file_path.split('.')[-1].lower()
            if ext == 'csv':
                self.format_combo.setCurrentText("CSV")
            elif ext == 'txt':
                self.format_combo.setCurrentText("TXT")
            elif ext == 'xlsx':
                self.format_combo.setCurrentText("XLSX")
                
    def get_file_path(self):
        """获取文件路径"""
        return self.file_path
        
    def get_format_type(self):
        """获取文件格式"""
        return self.format_combo.currentText().lower() 