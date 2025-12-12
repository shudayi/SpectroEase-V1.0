from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel
from PyQt5.QtCore import Qt
from app.views.responsive_dialog import ResponsiveDialog

class ProgressDialog(ResponsiveDialog):
    """Progress Bar Dialog"""
    
    def __init__(self, title="Processing", parent=None):
        super().__init__(parent, base_width=400, base_height=200)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Processing...")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Set window properties
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        
    def update_progress(self, value, status=None):
        """Update progress"""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
            
    def closeEvent(self, event):
        """Override close event to prevent manual closing"""
        if self.progress_bar.value() < 100:
            event.ignore()
        else:
            super().closeEvent(event) 