# app/views/feature_selection_view.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QMessageBox, 
    QAbstractItemView, QDialog, QFormLayout, QSpinBox, QLineEdit, QHBoxLayout,
    QFrame, QCheckBox, QDoubleSpinBox, QScrollArea, QGridLayout, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pandas as pd

class FeatureSelectionView(QWidget):
    def __init__(self, plugins: dict):
        super(FeatureSelectionView, self).__init__()
        self.plugins = plugins  # Dynamically loaded plugins
        self.layout = QVBoxLayout()
        self.layout.setSpacing(8)  # Reduced spacing between main layout elements
        self.layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        
        # Title
        title_label = QLabel("特征选择方法 (可选)")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        self.layout.addWidget(title_label)
        
        # Description
        description = QLabel("选择特征选择方法 (可选步骤，如不选择将使用全部特征):")
        description.setStyleSheet("font-size: 11px; color: #555;")
        self.layout.addWidget(description)
        
        # Built-in methods
        built_in_methods = [
            "SelectKBest",
            "Recursive Feature Elimination (RFE)",
            "Feature Importance",
            "Mutual Information",
            "LASSO",
            "Principal Component Analysis (PCA)"
        ]
        
        # Plugin methods
        plugin_methods = list(self.plugins.keys())
        all_methods = built_in_methods + plugin_methods
        
        # Create methods container
        methods_container = QFrame()
        methods_container.setFrameShape(QFrame.StyledPanel)
        methods_container.setStyleSheet("background-color: #f5f5f5; border-radius: 4px; padding: 6px;")
        methods_layout = QVBoxLayout(methods_container)
        methods_layout.setSpacing(2)  # Reduced spacing between methods
        methods_layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        
        # Create checkboxes
        self.method_checkboxes = {}
        
        for method in all_methods:
            method_frame = QFrame()
            method_frame.setFrameShape(QFrame.StyledPanel)
            method_frame.setStyleSheet("background-color: #f8f8f8; border-radius: 3px; padding: 4px;")
            
            method_layout = QHBoxLayout(method_frame)
            method_layout.setContentsMargins(3, 3, 3, 3)
            method_layout.setSpacing(4)
            
            checkbox = QCheckBox(method)
            checkbox.setStyleSheet("font-size: 11px;")
            # Ensure only one method can be selected
            checkbox.clicked.connect(lambda checked, m=method: self.handle_checkbox_clicked(m, checked))
            self.method_checkboxes[method] = checkbox
            
            method_layout.addWidget(checkbox)
            
            # Add parameters button
            params_button = QPushButton("Parameters")
            params_button.setStyleSheet("background-color: #e0e0e0; border: none; border-radius: 2px; padding: 2px 6px; font-size: 10px;")
            params_button.setFixedWidth(70)
            params_button.clicked.connect(lambda checked, m=method: self.show_parameters(m))
            method_layout.addWidget(params_button)
            
            methods_layout.addWidget(method_frame)
        
        # Add methods list to main layout
        self.layout.addWidget(methods_container)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0;")
        self.layout.addWidget(line)
        
        # Apply button
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)
        
        # Skip button
        self.skip_button = QPushButton("跳过特征选择")
        self.skip_button.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
            QPushButton:pressed {
                background-color: #616161;
            }
        """)
        
        self.apply_button = QPushButton("应用特征选择")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        button_layout.addStretch()
        button_layout.addWidget(self.skip_button)
        button_layout.addWidget(self.apply_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)
    
    def handle_checkbox_clicked(self, method, checked):
        """Ensure only one checkbox is selected"""
        if checked:
            for m, checkbox in self.method_checkboxes.items():
                if m != method:
                    checkbox.setChecked(False)

    def get_selected_method(self):
        selected = None
        for method, checkbox in self.method_checkboxes.items():
            if checkbox.isChecked():
                selected = method
                break
        
  
        if selected is None and self.method_checkboxes:
  
            first_method = list(self.method_checkboxes.keys())[0]
            self.method_checkboxes[first_method].setChecked(True)
            selected = first_method
            
        return selected
    
    def show_parameters(self, method):
        """Show parameter setting dialog"""
        dialog = FeatureSelectionParameterDialog(method, self.plugins)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            # You can save parameter settings here
            pass

    def get_parameters(self, method=None):
        """获取parameters设置，如果提供methodparameters则打开对话框"""
        if method:
            dialog = FeatureSelectionParameterDialog(method, self.plugins)
            result = dialog.exec_()
            if result == QDialog.Accepted:
                return dialog.get_parameters()
            else:
                return {}
        
  
        params = {}
        
  
        if hasattr(self, 'variance_check'):
            params['variance_threshold'] = self.variance_check.isChecked()
            params['variance_threshold_value'] = self.variance_threshold.value()
            
        if hasattr(self, 'correlation_check'):
            params['correlation_threshold'] = self.correlation_check.isChecked()
            params['correlation_threshold_value'] = self.correlation_threshold.value()
            
        if hasattr(self, 'mi_check'):
            params['mutual_information'] = self.mi_check.isChecked()
            params['mi_n_features'] = self.mi_k.value()
            
        if hasattr(self, 'lasso_check'):
            params['lasso'] = self.lasso_check.isChecked()
            params['lasso_alpha'] = self.lasso_alpha.value()
            
        if hasattr(self, 'rf_check'):
            params['random_forest'] = self.rf_check.isChecked()
            params['rf_n_features'] = self.rf_n_features.value()
            
        if hasattr(self, 'rfe_check'):
            params['recursive_feature_elimination'] = self.rfe_check.isChecked()
            params['rfe_n_features'] = self.rfe_n_features.value()
            
  
        if hasattr(self, 'param_widgets') and self.param_widgets:
            for name, widget in self.param_widgets.items():
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[name] = widget.value()
                elif isinstance(widget, QLineEdit):
                    params[name] = widget.text()
                elif isinstance(widget, QCheckBox):
                    params[name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[name] = widget.currentText()
        
        return params

    def update_feature_selection_results(self, selected_features):
        # You can implement the logic to update feature selection results display here
        pass

    def display_error(self, message):
        QMessageBox.critical(self, "Error", message)
        
    def display_message(self, message, title="Information"):
        """Display a message with the given title"""
        if title == "Error":
            QMessageBox.critical(self, title, message)
        elif title == "Warning":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def add_custom_algorithm(self, code):
        """Add custom feature selection algorithm"""
        try:
            import types
            mod = types.ModuleType('custom_feature_selector')
            exec(code, mod.__dict__)
            
            for item in mod.__dict__.values():
                if isinstance(item, type):
                    algorithm = item()
                    self.plugins[algorithm.get_name()] = algorithm
                    
                    # Add to UI
                    method = algorithm.get_name()
                    method_frame = QFrame()
                    method_frame.setFrameShape(QFrame.StyledPanel)
                    method_frame.setStyleSheet("background-color: #f8f8f8; border-radius: 3px; padding: 4px;")
                    
                    method_layout = QHBoxLayout(method_frame)
                    method_layout.setContentsMargins(3, 3, 3, 3)
                    method_layout.setSpacing(4)
                    
                    checkbox = QCheckBox(method)
                    checkbox.setStyleSheet("font-size: 11px;")
                    checkbox.clicked.connect(lambda checked, m=method: self.handle_checkbox_clicked(m, checked))
                    self.method_checkboxes[method] = checkbox
                    
                    method_layout.addWidget(checkbox)
                    
                    # Add parameters button
                    params_button = QPushButton("Parameters")
                    params_button.setStyleSheet("background-color: #e0e0e0; border: none; border-radius: 2px; padding: 2px 6px; font-size: 10px;")
                    params_button.setFixedWidth(70)
                    params_button.clicked.connect(lambda checked, m=method: self.show_parameters(m))
                    method_layout.addWidget(params_button)
                    
                    # Find parent container and add new method
                    for i in range(self.layout.count()):
                        widget = self.layout.itemAt(i).widget()
                        if isinstance(widget, QFrame):
                            widget.layout().addWidget(method_frame)
                            break
                    
                    break
                    
        except Exception as e:
            raise Exception(f"Error loading custom feature selector: {str(e)}")

class FeatureSelectionParameterDialog(QDialog):
    def __init__(self, method, plugins, parent=None):
        super(FeatureSelectionParameterDialog, self).__init__(parent)
        self.setWindowTitle(f"Parameters for {method}")
        self.setMinimumWidth(400)
        self.layout = QVBoxLayout()
        
  
        title_label = QLabel(f"Configure parameters for {method}")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 15px;")
        self.layout.addWidget(title_label)
        
  
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 0, 10, 0)
        
        self.parameters = {}
        self.method = method
        self.plugins = plugins
        self.param_widgets = {}

        if method in self.plugins:
  
            param_info = self.plugins[method].get_parameter_info()
            for param_name, info in param_info.items():
                widget = self.create_widget(info)
                form_layout.addRow(f"{param_name} ({info['description']}):", widget)
                self.param_widgets[param_name] = widget
        else:
  
            if method == "SelectKBest":
                k_spin = QSpinBox()
                k_spin.setRange(1, 100)
                k_spin.setValue(10)
                form_layout.addRow("Number of features (k):", k_spin)
                self.param_widgets['k'] = k_spin
                
                score_func = QComboBox()
                score_func.addItems(["f_classif", "chi2", "mutual_info_classif"])
                form_layout.addRow("Score function:", score_func)
                self.param_widgets['score_func'] = score_func
            elif method == "Recursive Feature Elimination (RFE)":
                n_features_to_select = QSpinBox()
                n_features_to_select.setRange(1, 100)
                n_features_to_select.setValue(10)
                form_layout.addRow("Number of features to select:", n_features_to_select)
                self.param_widgets['n_features_to_select'] = n_features_to_select
                
                step = QSpinBox()
                step.setRange(1, 10)
                step.setValue(1)
                form_layout.addRow("Features to remove each step:", step)
                self.param_widgets['step'] = step
            elif method == "Feature Importance":
                threshold = QDoubleSpinBox()
                threshold.setRange(0.0, 1.0)
                threshold.setValue(0.05)
                threshold.setSingleStep(0.01)
                form_layout.addRow("Importance threshold:", threshold)
                self.param_widgets['threshold'] = threshold
            elif method == "LASSO":
                alpha = QDoubleSpinBox()
                alpha.setRange(0.0001, 10.0)
                alpha.setValue(1.0)
                alpha.setSingleStep(0.1)
                form_layout.addRow("Regularization strength (alpha):", alpha)
                self.param_widgets['alpha'] = alpha
            elif method == "Principal Component Analysis (PCA)":
                n_components = QSpinBox()
                n_components.setRange(1, 100)
                n_components.setValue(2)
                form_layout.addRow("Number of components:", n_components)
                self.param_widgets['n_components'] = n_components
  
        
        self.layout.addLayout(form_layout)
        
  
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def get_parameters(self):
        """Get parameters based on widgets"""
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
        return params

    def create_widget(self, info):
        if info['type'] == 'int':
            widget = QSpinBox()
            widget.setRange(info.get('min', 0), info.get('max', 1000))
            widget.setValue(info.get('default', 0))
        elif info['type'] == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(info.get('min', 0.0), info.get('max', 1000.0))
            widget.setValue(info.get('default', 0.0))
            widget.setSingleStep(0.1)
        elif info['type'] == 'str':
            widget = QLineEdit()
            widget.setText(info.get('default', ''))
        elif info['type'] == 'bool':
            widget = QCheckBox()
            widget.setChecked(info.get('default', False))
        else:
            widget = QLineEdit()  # Default use text input box
        return widget

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Feature selection methods
        methods_group = QGroupBox("Feature Selection Methods")
        methods_layout = QVBoxLayout()
        
        # Variance threshold
        variance_layout = QHBoxLayout()
        self.variance_check = QCheckBox("Variance Threshold")
        self.variance_threshold = QDoubleSpinBox()
        self.variance_threshold.setRange(0.0, 1.0)
        self.variance_threshold.setValue(0.01)
        self.variance_threshold.setSingleStep(0.001)
        variance_layout.addWidget(self.variance_check)
        variance_layout.addWidget(QLabel("Threshold:"))
        variance_layout.addWidget(self.variance_threshold)
        methods_layout.addLayout(variance_layout)
        
        # Correlation threshold
        correlation_layout = QHBoxLayout()
        self.correlation_check = QCheckBox("Correlation Threshold")
        self.correlation_threshold = QDoubleSpinBox()
        self.correlation_threshold.setRange(0.0, 1.0)
        self.correlation_threshold.setValue(0.95)
        self.correlation_threshold.setSingleStep(0.01)
        correlation_layout.addWidget(self.correlation_check)
        correlation_layout.addWidget(QLabel("Threshold:"))
        correlation_layout.addWidget(self.correlation_threshold)
        methods_layout.addLayout(correlation_layout)
        
        # Mutual information
        mi_layout = QHBoxLayout()
        self.mi_check = QCheckBox("Mutual Information")
        self.mi_k = QSpinBox()
        self.mi_k.setRange(1, 100)
        self.mi_k.setValue(10)
        mi_layout.addWidget(self.mi_check)
        mi_layout.addWidget(QLabel("Number of Features:"))
        mi_layout.addWidget(self.mi_k)
        methods_layout.addLayout(mi_layout)
        
        # LASSO
        lasso_layout = QHBoxLayout()
        self.lasso_check = QCheckBox("LASSO")
        self.lasso_alpha = QDoubleSpinBox()
        self.lasso_alpha.setRange(0.0001, 1.0)
        self.lasso_alpha.setValue(0.01)
        self.lasso_alpha.setSingleStep(0.0001)
        lasso_layout.addWidget(self.lasso_check)
        lasso_layout.addWidget(QLabel("Alpha:"))
        lasso_layout.addWidget(self.lasso_alpha)
        methods_layout.addLayout(lasso_layout)
        
        # Random Forest
        rf_layout = QHBoxLayout()
        self.rf_check = QCheckBox("Random Forest")
        self.rf_n_features = QSpinBox()
        self.rf_n_features.setRange(1, 100)
        self.rf_n_features.setValue(10)
        rf_layout.addWidget(self.rf_check)
        rf_layout.addWidget(QLabel("Number of Features:"))
        rf_layout.addWidget(self.rf_n_features)
        methods_layout.addLayout(rf_layout)
        
        # Recursive feature elimination
        rfe_layout = QHBoxLayout()
        self.rfe_check = QCheckBox("Recursive Feature Elimination")
        self.rfe_n_features = QSpinBox()
        self.rfe_n_features.setRange(1, 100)
        self.rfe_n_features.setValue(10)
        rfe_layout.addWidget(self.rfe_check)
        rfe_layout.addWidget(QLabel("Number of Features:"))
        rfe_layout.addWidget(self.rfe_n_features)
        methods_layout.addLayout(rfe_layout)
        
        methods_group.setLayout(methods_layout)
        layout.addWidget(methods_group)
        
        # Feature importance plot
        plot_group = QGroupBox("Feature Importance")
        plot_layout = QVBoxLayout()
        
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def reset_parameters(self):
        """Reset all parameters to default values"""
        self.variance_check.setChecked(False)
        self.variance_threshold.setValue(0.01)
        self.correlation_check.setChecked(False)
        self.correlation_threshold.setValue(0.95)
        self.mi_check.setChecked(False)
        self.mi_k.setValue(10)
        self.lasso_check.setChecked(False)
        self.lasso_alpha.setValue(0.01)
        self.rf_check.setChecked(False)
        self.rf_n_features.setValue(10)
        self.rfe_check.setChecked(False)
        self.rfe_n_features.setValue(10)
