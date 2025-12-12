# -*- coding: utf-8 -*-
"""
NaN/Inf Data Handling Dialog
Addresses Reviewer 1 Comment 1 about data loading errors with invalid values
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QRadioButton, QButtonGroup, QTextEdit, QGroupBox, QMessageBox,
    QProgressBar, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

class DataCleaningWorker(QThread):
    """Worker thread for data cleaning operations"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object, dict)  # cleaned_data, statistics
    
    def __init__(self, data, action, interpolation_method='linear'):
        super().__init__()
        self.data = data
        self.action = action
        self.interpolation_method = interpolation_method
    
    def run(self):
        try:
            self.status.emit("Analyzing data quality...")
            self.progress.emit(10)
            
            # Analyze data issues
            stats = self._analyze_data_issues(self.data)
            self.progress.emit(30)
            
            if self.action == "drop":
                self.status.emit("Removing rows with invalid values...")
                cleaned_data = self._drop_invalid_rows(self.data, stats)
            elif self.action == "interpolate":
                self.status.emit(f"Interpolating using {self.interpolation_method} method...")
                cleaned_data = self._interpolate_values(self.data, stats)
            else:  # keep
                self.status.emit("Keeping data unchanged...")
                cleaned_data = self.data.copy()
            
            self.progress.emit(90)
            
            # Final validation
            self.status.emit("Validating cleaned data...")
            final_stats = self._analyze_data_issues(cleaned_data)
            self.progress.emit(100)
            
            self.finished.emit(cleaned_data, {
                'original_stats': stats,
                'final_stats': final_stats,
                'action_taken': self.action
            })
            
        except Exception as e:
            self.status.emit(f"Error during data cleaning: {str(e)}")
            self.finished.emit(None, {'error': str(e)})
    
    def _analyze_data_issues(self, data):
        """Analyze data for NaN, Inf, and other issues"""
        stats = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'nan_count': 0,
            'inf_count': 0,
            'problematic_rows': set(),
            'problematic_columns': {},
            'zero_variance_columns': []
        }
        
        for col in data.columns:
            col_data = data[col]
            
            # Count NaN values
            nan_count = col_data.isna().sum()
            if nan_count > 0:
                stats['nan_count'] += nan_count
                stats['problematic_columns'][col] = stats['problematic_columns'].get(col, {})
                stats['problematic_columns'][col]['nan_count'] = nan_count
                stats['problematic_rows'].update(col_data[col_data.isna()].index.tolist())
            
            # Count Inf values (for numeric columns)
            if pd.api.types.is_numeric_dtype(col_data):
                inf_count = np.isinf(col_data).sum()
                if inf_count > 0:
                    stats['inf_count'] += inf_count
                    stats['problematic_columns'][col] = stats['problematic_columns'].get(col, {})
                    stats['problematic_columns'][col]['inf_count'] = inf_count
                    stats['problematic_rows'].update(col_data[np.isinf(col_data)].index.tolist())
                
                # Check for zero variance
                if col_data.var() == 0:
                    stats['zero_variance_columns'].append(col)
        
        stats['problematic_rows'] = list(stats['problematic_rows'])
        return stats
    
    def _drop_invalid_rows(self, data, stats):
        """Drop rows containing NaN or Inf values"""
        if not stats['problematic_rows']:
            return data.copy()
        
        cleaned_data = data.drop(index=stats['problematic_rows']).reset_index(drop=True)
        return cleaned_data
    
    def _interpolate_values(self, data, stats):
        """Interpolate NaN and Inf values"""
        cleaned_data = data.copy()
        
        for col in data.columns:
            if col in stats['problematic_columns']:
                col_data = cleaned_data[col]
                
                # Replace Inf with NaN first
                col_data = col_data.replace([np.inf, -np.inf], np.nan)
                
                # Interpolate NaN values
                if pd.api.types.is_numeric_dtype(col_data):
                    if self.interpolation_method == 'linear':
                        col_data = col_data.interpolate(method='linear')
                    elif self.interpolation_method == 'median':
                        col_data = col_data.fillna(col_data.median())
                    elif self.interpolation_method == 'mean':
                        col_data = col_data.fillna(col_data.mean())
                    
                    # Fill any remaining NaN with forward/backward fill
                    col_data = col_data.bfill().ffill()
                
                cleaned_data[col] = col_data
        
        return cleaned_data

class NaNInfHandlingDialog(QDialog):
    """Dialog for handling NaN and Inf values in spectral data"""
    
    def __init__(self, data, parent=None):
        from app.utils.ui_scaling import ui_scaling_manager
        # 注意：由于此类有特殊的__init__签名，我们手动设置响应式尺寸
        super().__init__(parent)
        self.data = data
        self.cleaned_data = None
        self.statistics = None
        
        # 手动应用响应式尺寸
        width, height = ui_scaling_manager.get_responsive_dialog_size(750, 600)
        self.resize(width, height)
        
        self.init_ui()
        self.analyze_data()
    
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Data Quality Issues Detected")
        self.setModal(True)
        
        # Set font
        font = QFont("Arial", 9)
        self.setFont(font)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Data Quality Issues Detected")
        header.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #d63031; }")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Problem description
        self.problem_text = QTextEdit()
        self.problem_text.setMaximumHeight(120)
        self.problem_text.setReadOnly(True)
        layout.addWidget(self.problem_text)
        
        # Action selection group
        action_group = QGroupBox("Choose Action")
        action_layout = QVBoxLayout()
        
        self.action_group = QButtonGroup()
        
        # Option 1: Drop affected rows
        self.drop_radio = QRadioButton("Drop affected rows (Recommended for small datasets)")
        self.drop_radio.setToolTip("Remove all rows containing NaN or Inf values")
        action_layout.addWidget(self.drop_radio)
        self.action_group.addButton(self.drop_radio, 1)
        
        # Option 2: Interpolate values
        self.interpolate_radio = QRadioButton("Interpolate values (Recommended for spectral data)")
        self.interpolate_radio.setToolTip("Replace NaN/Inf with interpolated values")
        self.interpolate_radio.setChecked(True)  # Default choice
        action_layout.addWidget(self.interpolate_radio)
        self.action_group.addButton(self.interpolate_radio, 2)
        
        # Option 3: Keep unchanged
        self.keep_radio = QRadioButton("Keep data unchanged (May cause plotting errors)")
        self.keep_radio.setToolTip("Proceed with data as-is - may cause visualization errors")
        action_layout.addWidget(self.keep_radio)
        self.action_group.addButton(self.keep_radio, 3)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Interpolation method selection (initially hidden)
        self.interpolation_group = QGroupBox("Interpolation Method")
        interp_layout = QVBoxLayout()
        
        self.interp_method_group = QButtonGroup()
        
        self.linear_radio = QRadioButton("Linear interpolation (Best for spectral data)")
        self.linear_radio.setChecked(True)
        interp_layout.addWidget(self.linear_radio)
        self.interp_method_group.addButton(self.linear_radio, 1)
        
        self.median_radio = QRadioButton("Median replacement")
        interp_layout.addWidget(self.median_radio)
        self.interp_method_group.addButton(self.median_radio, 2)
        
        self.mean_radio = QRadioButton("Mean replacement")
        interp_layout.addWidget(self.mean_radio)
        self.interp_method_group.addButton(self.mean_radio, 3)
        
        self.interpolation_group.setLayout(interp_layout)
        layout.addWidget(self.interpolation_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Preview Changes")
        self.preview_btn.clicked.connect(self.preview_changes)
        button_layout.addWidget(self.preview_btn)
        
        self.apply_btn = QPushButton("Apply & Continue")
        self.apply_btn.clicked.connect(self.apply_changes)
        self.apply_btn.setStyleSheet("QPushButton { background-color: #00b894; color: white; font-weight: bold; }")
        button_layout.addWidget(self.apply_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.interpolate_radio.toggled.connect(self.on_action_changed)
        self.on_action_changed()  # Initialize visibility
    
    def analyze_data(self):
        """Analyze the data for issues"""
        nan_count = self.data.isna().sum().sum()
        inf_count = 0
        
        # Count Inf values in numeric columns
        for col in self.data.select_dtypes(include=[np.number]).columns:
            inf_count += np.isinf(self.data[col]).sum()
        
        # Identify problematic rows
        problematic_rows = set()
        for col in self.data.columns:
            col_data = self.data[col]
            problematic_rows.update(col_data[col_data.isna()].index.tolist())
            if pd.api.types.is_numeric_dtype(col_data):
                problematic_rows.update(col_data[np.isinf(col_data)].index.tolist())
        
        # Update problem description
        problem_text = f"""
Data Quality Analysis:
• Total rows: {len(self.data)}
• Total columns: {len(self.data.columns)}
• NaN values detected: {nan_count}
• Infinite values detected: {inf_count}
• Affected rows: {len(problematic_rows)} ({len(problematic_rows)/len(self.data)*100:.1f}%)

These invalid values will cause plotting errors like "Axis limits cannot be NaN or Inf".
Please choose how to handle these issues:
        """
        
        self.problem_text.setPlainText(problem_text.strip())
    
    def on_action_changed(self):
        """Handle action selection changes"""
        interpolate_selected = self.interpolate_radio.isChecked()
        self.interpolation_group.setVisible(interpolate_selected)
    
    def preview_changes(self):
        """Preview the changes that would be made"""
        action = self.get_selected_action()
        interp_method = self.get_interpolation_method()
        
        # Create worker for preview
        self.worker = DataCleaningWorker(self.data, action, interp_method)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_preview_finished)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.preview_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        
        self.worker.start()
    
    def on_preview_finished(self, cleaned_data, stats):
        """Handle preview completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.preview_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        
        if cleaned_data is None:
            QMessageBox.critical(self, "Error", f"Preview failed: {stats.get('error', 'Unknown error')}")
            return
        
        # Show preview results
        original_stats = stats['original_stats']
        final_stats = stats['final_stats']
        action = stats['action_taken']
        
        preview_text = f"""
Preview Results for Action: {action.upper()}

BEFORE:
• Rows: {original_stats['total_rows']}
• NaN values: {original_stats['nan_count']}
• Inf values: {original_stats['inf_count']}
• Problematic rows: {len(original_stats['problematic_rows'])}

AFTER:
• Rows: {final_stats['total_rows']}
• NaN values: {final_stats['nan_count']}
• Inf values: {final_stats['inf_count']}
• Problematic rows: {len(final_stats['problematic_rows'])}

Data will be suitable for visualization: {"Yes" if final_stats['nan_count'] == 0 and final_stats['inf_count'] == 0 else "No"}
        """
        
        QMessageBox.information(self, "Preview Results", preview_text.strip())
    
    def apply_changes(self):
        """Apply the selected changes"""
        action = self.get_selected_action()
        interp_method = self.get_interpolation_method()
        
        # Create worker for actual processing
        self.worker = DataCleaningWorker(self.data, action, interp_method)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_apply_finished)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.preview_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        
        self.worker.start()
    
    def on_apply_finished(self, cleaned_data, stats):
        """Handle apply completion"""
        if cleaned_data is None:
            QMessageBox.critical(self, "Error", f"Data cleaning failed: {stats.get('error', 'Unknown error')}")
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            self.preview_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)
            return
        
        self.cleaned_data = cleaned_data
        self.statistics = stats
        
        # Show success message
        final_stats = stats['final_stats']
        success_msg = f"""
Data cleaning completed successfully!

Final data quality:
• Rows: {final_stats['total_rows']}
• Columns: {final_stats['total_columns']}
• NaN values: {final_stats['nan_count']}
• Inf values: {final_stats['inf_count']}

The data is now ready for visualization and analysis.
        """
        
        QMessageBox.information(self, "Success", success_msg.strip())
        self.accept()
    
    def get_selected_action(self):
        """Get the selected action"""
        if self.drop_radio.isChecked():
            return "drop"
        elif self.interpolate_radio.isChecked():
            return "interpolate"
        else:
            return "keep"
    
    def get_interpolation_method(self):
        """Get the selected interpolation method"""
        if self.linear_radio.isChecked():
            return "linear"
        elif self.median_radio.isChecked():
            return "median"
        else:
            return "mean"
    
    def get_cleaned_data(self):
        """Get the cleaned data"""
        return self.cleaned_data
    
    def get_statistics(self):
        """Get cleaning statistics"""
        return self.statistics

def detect_and_handle_data_issues(data, parent=None):
    """
    Detect and handle data quality issues
    
    Args:
        data: DataFrame to check
        parent: Parent widget for dialog
        
    Returns:
        Tuple[DataFrame, bool]: (cleaned_data, user_cancelled)
    """
    # Quick check for issues
    has_nan = data.isna().any().any()
    has_inf = False
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if np.isinf(data[col]).any():
            has_inf = True
            break
    
    if not has_nan and not has_inf:
        return data, False  # No issues found
    
    # Show dialog for handling issues
    dialog = NaNInfHandlingDialog(data, parent)
    if dialog.exec_() == QDialog.Accepted:
        cleaned_data = dialog.get_cleaned_data()
        return cleaned_data if cleaned_data is not None else data, False
    else:
        return data, True  # User cancelled

if __name__ == "__main__":
    # Test the dialog
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication([])
    
    # Create test data with issues
    test_data = pd.DataFrame({
        'wavelength_1': [1, 2, np.nan, 4, 5],
        'wavelength_2': [1.1, np.inf, 3.3, 4.4, 5.5],
        'wavelength_3': [2.1, 2.2, 2.3, np.nan, 2.5]
    })
    
    cleaned_data, cancelled = detect_and_handle_data_issues(test_data)
    
    if not cancelled:
        logging.info("Cleaned data:")
        logging.info(cleaned_data)
    else:
        logging.info("User cancelled")
    
    app.exec_()
