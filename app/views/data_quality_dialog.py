# -*- coding: utf-8 -*-
"""
Data Quality Dialog
Addresses Reviewer 1 Comment 1: Interactive NaN/Inf handling with three options
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QGroupBox, QRadioButton, QButtonGroup, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from typing import Dict, Any, Optional, Tuple, List
import warnings

class DataQualityAnalyzer:
    """Analyzes data quality issues (NaN, Inf, etc.)"""
    
    @staticmethod
    def analyze_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        analysis = {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'issues': {},
            'recommendations': [],
            'severity': 'none'  # none, low, medium, high
        }
        
        # Check for NaN values
        nan_info = DataQualityAnalyzer._analyze_nan_values(data)
        if nan_info['total_nan'] > 0:
            analysis['issues']['nan'] = nan_info
        
        # Check for infinite values
        inf_info = DataQualityAnalyzer._analyze_inf_values(data)
        if inf_info['total_inf'] > 0:
            analysis['issues']['inf'] = inf_info
        
        # Check for empty/zero columns
        empty_info = DataQualityAnalyzer._analyze_empty_columns(data)
        if empty_info['empty_columns']:
            analysis['issues']['empty'] = empty_info
        
        # Determine overall severity
        analysis['severity'] = DataQualityAnalyzer._determine_severity(analysis['issues'])
        
        # Generate recommendations
        analysis['recommendations'] = DataQualityAnalyzer._generate_recommendations(analysis)
        
        return analysis
    
    @staticmethod
    def _analyze_nan_values(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze NaN value distribution"""
        nan_counts = data.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        
        return {
            'total_nan': int(nan_counts.sum()),
            'affected_columns': len(nan_columns),
            'column_details': [
                {
                    'column': col,
                    'nan_count': int(count),
                    'nan_percentage': float(count / len(data) * 100)
                }
                for col, count in nan_columns.items()
            ],
            'max_nan_percentage': float(nan_columns.max() / len(data) * 100) if len(nan_columns) > 0 else 0
        }
    
    @staticmethod
    def _analyze_inf_values(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze infinite value distribution"""
        # Only check numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        inf_info = {
            'total_inf': 0,
            'affected_columns': 0,
            'column_details': []
        }
        
        for col in numeric_cols:
            inf_count = np.isinf(data[col]).sum()
            if inf_count > 0:
                inf_info['total_inf'] += inf_count
                inf_info['affected_columns'] += 1
                inf_info['column_details'].append({
                    'column': col,
                    'inf_count': int(inf_count),
                    'inf_percentage': float(inf_count / len(data) * 100)
                })
        
        return inf_info
    
    @staticmethod
    def _analyze_empty_columns(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze empty or all-zero columns"""
        empty_columns = []
        
        for col in data.columns:
            if data[col].isnull().all():
                empty_columns.append({'column': col, 'type': 'all_null'})
            elif data[col].dtype in [np.number] and (data[col] == 0).all():
                empty_columns.append({'column': col, 'type': 'all_zero'})
        
        return {
            'empty_columns': empty_columns,
            'count': len(empty_columns)
        }
    
    @staticmethod
    def _determine_severity(issues: Dict[str, Any]) -> str:
        """Determine overall severity level"""
        if not issues:
            return 'none'
        
        severity = 'low'
        
        # Check NaN severity
        if 'nan' in issues:
            max_nan_pct = issues['nan']['max_nan_percentage']
            if max_nan_pct > 50:
                severity = 'high'
            elif max_nan_pct > 20:
                severity = 'medium'
        
        # Check Inf severity
        if 'inf' in issues:
            if issues['inf']['total_inf'] > 0:
                severity = max(severity, 'medium', key=['low', 'medium', 'high'].index)
        
        # Check empty columns
        if 'empty' in issues:
            if issues['empty']['count'] > 0:
                severity = max(severity, 'medium', key=['low', 'medium', 'high'].index)
        
        return severity
    
    @staticmethod
    def _generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if 'nan' in analysis['issues']:
            nan_info = analysis['issues']['nan']
            if nan_info['max_nan_percentage'] > 50:
                recommendations.append("High NaN percentage detected - consider removing affected columns")
            elif nan_info['max_nan_percentage'] > 20:
                recommendations.append("Moderate NaN percentage - interpolation or row removal recommended")
            else:
                recommendations.append("Low NaN percentage - interpolation should work well")
        
        if 'inf' in analysis['issues']:
            recommendations.append("Infinite values detected - removal or replacement required")
        
        if 'empty' in analysis['issues']:
            recommendations.append("Empty columns detected - consider removing them")
        
        return recommendations

class DataQualityDialog(QDialog):
    """Interactive dialog for handling data quality issues"""
    
    def __init__(self, data: pd.DataFrame, file_path: str = "", parent=None):
        from app.views.responsive_dialog import ResponsiveDialog
        from app.utils.ui_scaling import ui_scaling_manager
        # 注意：由于此类有特殊的__init__签名，我们手动设置响应式尺寸
        super().__init__(parent)
        self.data = data.copy()
        self.original_data = data.copy()
        self.file_path = file_path
        self.analysis = DataQualityAnalyzer.analyze_data_quality(data)
        self.selected_action = 'cancel'  # Default action
        
        # 手动应用响应式尺寸
        width, height = ui_scaling_manager.get_responsive_dialog_size(800, 650)
        self.resize(width, height)
        
        self.init_ui()
        self.update_analysis_display()
    
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Data Quality Issues Detected")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Title and file info
        title_label = QLabel("Data Quality Analysis")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)
        
        if self.file_path:
            file_label = QLabel(f"File: {self.file_path}")
            file_label.setFont(QFont("Arial", 9))
            layout.addWidget(file_label)
        
        # Analysis results
        self.analysis_group = QGroupBox("Issues Detected")
        analysis_layout = QVBoxLayout()
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setMaximumHeight(200)
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        self.analysis_group.setLayout(analysis_layout)
        layout.addWidget(self.analysis_group)
        
        # Action selection
        action_group = QGroupBox("Choose Action")
        action_layout = QVBoxLayout()
        
        self.action_group = QButtonGroup()
        
        # Option 1: Remove affected rows/columns
        self.remove_radio = QRadioButton("Remove affected rows/columns")
        self.remove_radio.setToolTip("Remove rows or columns with NaN/Inf values")
        self.action_group.addButton(self.remove_radio, 0)
        action_layout.addWidget(self.remove_radio)
        
        # Remove options
        remove_options_layout = QHBoxLayout()
        remove_options_layout.addSpacing(20)
        
        self.remove_rows_check = QCheckBox("Remove rows")
        self.remove_cols_check = QCheckBox("Remove columns")
        self.remove_threshold_label = QLabel("Threshold (%):")
        self.remove_threshold = QSpinBox()
        self.remove_threshold.setRange(1, 100)
        self.remove_threshold.setValue(50)
        
        remove_options_layout.addWidget(self.remove_rows_check)
        remove_options_layout.addWidget(self.remove_cols_check)
        remove_options_layout.addWidget(self.remove_threshold_label)
        remove_options_layout.addWidget(self.remove_threshold)
        remove_options_layout.addStretch()
        
        action_layout.addLayout(remove_options_layout)
        
        # Option 2: Interpolate values
        self.interpolate_radio = QRadioButton("Interpolate missing values")
        self.interpolate_radio.setToolTip("Fill NaN values using interpolation")
        self.action_group.addButton(self.interpolate_radio, 1)
        action_layout.addWidget(self.interpolate_radio)
        
        # Interpolation options
        interp_options_layout = QHBoxLayout()
        interp_options_layout.addSpacing(20)
        
        self.interp_method_label = QLabel("Method:")
        self.interp_method = QComboBox()
        self.interp_method.addItems(["linear", "polynomial", "spline", "nearest"])
        
        interp_options_layout.addWidget(self.interp_method_label)
        interp_options_layout.addWidget(self.interp_method)
        interp_options_layout.addStretch()
        
        action_layout.addLayout(interp_options_layout)
        
        # Option 3: Cancel
        self.cancel_radio = QRadioButton("Cancel - do not load this data")
        self.cancel_radio.setToolTip("Cancel data loading and return to file selection")
        self.cancel_radio.setChecked(True)  # Default selection
        self.action_group.addButton(self.cancel_radio, 2)
        action_layout.addWidget(self.cancel_radio)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Preview area
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_button = QPushButton("Preview Changes")
        self.preview_button.clicked.connect(self.preview_changes)
        preview_layout.addWidget(self.preview_button)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        self.apply_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect signals
        self.action_group.buttonClicked.connect(self.on_action_changed)
        
        # Set initial state
        self.on_action_changed()
    
    def update_analysis_display(self):
        """Update the analysis display"""
        text = f"Data Shape: {self.analysis['total_samples']} samples × {self.analysis['total_features']} features\n"
        text += f"Severity: {self.analysis['severity'].upper()}\n\n"
        
        if 'nan' in self.analysis['issues']:
            nan_info = self.analysis['issues']['nan']
            text += f"NaN Values:\n"
            text += f"  Total: {nan_info['total_nan']}\n"
            text += f"  Affected columns: {nan_info['affected_columns']}\n"
            for col_info in nan_info['column_details'][:5]:  # Show first 5
                text += f"    {col_info['column']}: {col_info['nan_count']} ({col_info['nan_percentage']:.1f}%)\n"
            if len(nan_info['column_details']) > 5:
                text += f"    ... and {len(nan_info['column_details']) - 5} more columns\n"
            text += "\n"
        
        if 'inf' in self.analysis['issues']:
            inf_info = self.analysis['issues']['inf']
            text += f"Infinite Values:\n"
            text += f"  Total: {inf_info['total_inf']}\n"
            text += f"  Affected columns: {inf_info['affected_columns']}\n"
            text += "\n"
        
        if 'empty' in self.analysis['issues']:
            empty_info = self.analysis['issues']['empty']
            text += f"Empty Columns: {empty_info['count']}\n\n"
        
        if self.analysis['recommendations']:
            text += "Recommendations:\n"
            for rec in self.analysis['recommendations']:
                text += f"  • {rec}\n"
        
        self.analysis_text.setPlainText(text)
    
    def on_action_changed(self):
        """Handle action selection change"""
        selected_id = self.action_group.checkedId()
        
        # Enable/disable options based on selection
        self.remove_rows_check.setEnabled(selected_id == 0)
        self.remove_cols_check.setEnabled(selected_id == 0)
        self.remove_threshold.setEnabled(selected_id == 0)
        self.remove_threshold_label.setEnabled(selected_id == 0)
        
        self.interp_method.setEnabled(selected_id == 1)
        self.interp_method_label.setEnabled(selected_id == 1)
        
        # Set default sub-options
        if selected_id == 0:  # Remove
            self.remove_rows_check.setChecked(True)
            if 'nan' in self.analysis['issues']:
                threshold = min(50, self.analysis['issues']['nan']['max_nan_percentage'])
                self.remove_threshold.setValue(int(threshold))
        
        self.preview_text.clear()
    
    def preview_changes(self):
        """Preview the changes that would be made"""
        selected_id = self.action_group.checkedId()
        
        try:
            if selected_id == 0:  # Remove
                preview_data = self._apply_removal(self.data.copy(), preview=True)
            elif selected_id == 1:  # Interpolate
                preview_data = self._apply_interpolation(self.data.copy(), preview=True)
            else:  # Cancel
                self.preview_text.setPlainText("No changes - data loading will be cancelled.")
                return
            
            # Show preview information
            original_shape = self.data.shape
            new_shape = preview_data.shape
            
            preview_text = f"Original shape: {original_shape[0]} × {original_shape[1]}\n"
            preview_text += f"New shape: {new_shape[0]} × {new_shape[1]}\n"
            preview_text += f"Rows removed: {original_shape[0] - new_shape[0]}\n"
            preview_text += f"Columns removed: {original_shape[1] - new_shape[1]}\n\n"
            
            # Check remaining issues
            remaining_analysis = DataQualityAnalyzer.analyze_data_quality(preview_data)
            if remaining_analysis['severity'] == 'none':
                preview_text += "All data quality issues resolved!"
            else:
                preview_text += f"Remaining issues (severity: {remaining_analysis['severity']})\n"
                for issue_type, issue_info in remaining_analysis['issues'].items():
                    if issue_type == 'nan':
                        preview_text += f"  NaN values: {issue_info['total_nan']}\n"
                    elif issue_type == 'inf':
                        preview_text += f"  Inf values: {issue_info['total_inf']}\n"
            
            self.preview_text.setPlainText(preview_text)
            
        except Exception as e:
            self.preview_text.setPlainText(f"Preview failed: {str(e)}")
    
    def _apply_removal(self, data: pd.DataFrame, preview: bool = False) -> pd.DataFrame:
        """Apply removal strategy"""
        threshold = self.remove_threshold.value() / 100.0
        
        # Remove columns with high NaN percentage
        if self.remove_cols_check.isChecked():
            nan_percentages = data.isnull().sum() / len(data)
            cols_to_remove = nan_percentages[nan_percentages > threshold].index
            data = data.drop(columns=cols_to_remove)
            if not preview:
                print(f"Removed {len(cols_to_remove)} columns with >{threshold:.0%} NaN values")
        
        # Remove rows with NaN values
        if self.remove_rows_check.isChecked():
            initial_rows = len(data)
            data = data.dropna()
            removed_rows = initial_rows - len(data)
            if not preview:
                print(f"Removed {removed_rows} rows with NaN values")
        
        # Remove infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = data[~np.isinf(data[col])]
        
        return data
    
    def _apply_interpolation(self, data: pd.DataFrame, preview: bool = False) -> pd.DataFrame:
        """Apply interpolation strategy"""
        method = self.interp_method.currentText()
        
        # Interpolate NaN values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().any():
                if method == "linear":
                    data[col] = data[col].interpolate(method='linear')
                elif method == "polynomial":
                    data[col] = data[col].interpolate(method='polynomial', order=2)
                elif method == "spline":
                    data[col] = data[col].interpolate(method='spline', order=3)
                elif method == "nearest":
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Replace infinite values with NaN, then interpolate
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].interpolate(method='linear')
        
        # Fill any remaining NaN with column mean
        data.fillna(data.mean(), inplace=True)
        
        if not preview:
            print(f"Applied {method} interpolation to missing values")
        
        return data
    
    def apply_action(self):
        """Apply the selected action"""
        selected_id = self.action_group.checkedId()
        
        if selected_id == 2:  # Cancel
            self.selected_action = 'cancel'
            self.reject()
            return
        
        try:
            if selected_id == 0:  # Remove
                self.data = self._apply_removal(self.data)
                self.selected_action = 'remove'
            elif selected_id == 1:  # Interpolate
                self.data = self._apply_interpolation(self.data)
                self.selected_action = 'interpolate'
            
            # Verify the result
            final_analysis = DataQualityAnalyzer.analyze_data_quality(self.data)
            if final_analysis['severity'] != 'none':
                reply = QMessageBox.question(
                    self, "Remaining Issues",
                    f"Some data quality issues remain (severity: {final_analysis['severity']}).\n"
                    "Do you want to proceed anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply action: {str(e)}")
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, str]:
        """Get the processed data and action taken"""
        return self.data, self.selected_action
