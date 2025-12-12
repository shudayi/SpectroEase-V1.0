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
        title_label = QLabel("Feature Selection Method (Optional)")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        self.layout.addWidget(title_label)
        
        # Description
        description = QLabel("Select feature selection method (Optional step, all features will be used if not selected):")
        description.setStyleSheet("font-size: 11px; color: #555;")
        self.layout.addWidget(description)
        
        # Built-in methods - designed from spectral analysis expert perspective
        built_in_methods = [
            "CARS",
            "SPA",
            "PLSR",
            "SelectKBest",
            "Mutual Information",
            "RFE",
            "PCA"
        ]
        
        # Plugin methods
        plugin_methods = list(self.plugins.keys())
        
        # V1.3.0: Handle built-in and custom algorithms separately
        # Create methods container
        methods_container = QFrame()
        methods_container.setFrameShape(QFrame.StyledPanel)
        methods_container.setStyleSheet("background-color: #f5f5f5; border-radius: 4px; padding: 6px;")
        methods_layout = QVBoxLayout(methods_container)
        methods_layout.setSpacing(2)  # Reduced spacing between methods
        methods_layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        
        # Store methods layout for later use
        self.methods_layout = methods_layout
        
        # Create checkboxes
        self.method_checkboxes = {}
        
        # V1.3.0: Add built-in algorithms first
        for method in built_in_methods:
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
        
        # V1.3.0: Add separator line and custom algorithms section
        if plugin_methods:  # If there are custom algorithms
            self._add_custom_algorithms_section(plugin_methods)
        
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
        self.skip_button = QPushButton("Skip Feature Selection")
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
        
        self.apply_button = QPushButton("Apply Feature Selection")
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
        """Get parameters settings, if method parameters are provided, open dialog"""
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

    def _add_custom_algorithms_section(self, plugin_methods):
        """V1.3.0: Add custom algorithms section (with separator line and title)"""
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #d0d0d0; margin: 8px 0px;")
        self.methods_layout.addWidget(separator)
        
        # Add title label
        custom_label = QLabel("Custom Algorithms")
        custom_label.setAlignment(Qt.AlignCenter)
        custom_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-weight: bold;
                font-size: 11px;
                padding: 4px;
                background-color: transparent;
            }
        """)
        self.methods_layout.addWidget(custom_label)
        
        # Add custom algorithms
        for method in plugin_methods:
            self._add_custom_algorithm_ui(method)
    
    def _add_custom_algorithm_ui(self, method_name):
        """V1.3.0: Add single custom algorithm UI (with special style)"""
        method_frame = QFrame()
        method_frame.setFrameShape(QFrame.StyledPanel)
        # Custom algorithm special style
        method_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E3F2FD,
                    stop:1 #f8f8f8
                );
                border-left: 3px solid #2196F3;
                border-radius: 3px;
                padding: 4px;
            }
            QFrame:hover {
                background: #E3F2FD;
                border-left: 3px solid #1976D2;
            }
        """)
        
        method_layout = QHBoxLayout(method_frame)
        method_layout.setContentsMargins(3, 3, 3, 3)
        method_layout.setSpacing(4)
        
        # V1.3.5: Use concise custom identifier
        checkbox = QCheckBox(f"[Custom Feature] {method_name}")
        checkbox.setStyleSheet("font-size: 11px; background-color: transparent;")
        checkbox.setToolTip(f"Custom Feature Selection: {method_name}\n✅ Converted by LLM or user-defined\n⚙️ Click 'Parameters' to configure")
        checkbox.clicked.connect(lambda checked, m=method_name: self.handle_checkbox_clicked(m, checked))
        self.method_checkboxes[method_name] = checkbox
        
        method_layout.addWidget(checkbox)
        
        # Add parameters button
        params_button = QPushButton("Parameters")
        params_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: none;
                border-radius: 2px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        params_button.setFixedWidth(70)
        params_button.clicked.connect(lambda: self.show_parameters(method_name))
        method_layout.addWidget(params_button)
        
        self.methods_layout.addWidget(method_frame)
    
    def add_custom_algorithm(self, code):
        """V1.3.6: Add custom feature selection algorithm - Full scientific computing support"""
        try:
            import types
            import inspect
            from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
            import pandas as pd
            import numpy as np
            
            mod = types.ModuleType('custom_feature_selector')
            
            # V1.3.6: Provide comprehensive imports for scientific computing
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'FeatureSelectionAlgorithm': FeatureSelectionAlgorithm,
                'Dict': __import__('typing').Dict,
                'List': __import__('typing').List,
                'Any': __import__('typing').Any,
                'Tuple': __import__('typing').Tuple,
            })
            
            # V1.3.6: Add scipy support (if available)
            try:
                import scipy
                import scipy.signal
                import scipy.stats
                from scipy.signal import savgol_filter
                mod.__dict__.update({
                    'scipy': scipy,
                    'savgol_filter': savgol_filter,
                })
            except ImportError:
                pass  # scipy not available
            
            # V1.3.7: Add numpy.polynomial support
            try:
                from numpy.polynomial import polynomial
                from numpy.polynomial.polynomial import polyfit, polyval
                mod.__dict__.update({
                    'polynomial': polynomial,
                    'polyfit': polyfit,
                    'polyval': polyval,
                })
            except ImportError:
                pass
            
            # V1.3.6: Add sklearn support (if available)
            try:
                import sklearn
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
                from sklearn.decomposition import PCA
                mod.__dict__.update({
                    'sklearn': sklearn,
                    'LinearRegression': LinearRegression,
                    'StandardScaler': StandardScaler,
                    'SelectKBest': SelectKBest,
                    'mutual_info_classif': mutual_info_classif,
                    'RFE': RFE,
                    'PCA': PCA,
                })
            except ImportError:
                pass  # sklearn components not available
            
            # Record class list before execution
            classes_before = set(item for item in mod.__dict__.values() if isinstance(item, type))
            
            exec(code, mod.__dict__)
            
            # Record class list after execution, find newly added classes
            classes_after = set(item for item in mod.__dict__.values() if isinstance(item, type))
            new_classes = classes_after - classes_before
            
            algorithm_found = False
            for item in new_classes:
                # V1.3.2: Must be subclass and not abstract class
                if (issubclass(item, FeatureSelectionAlgorithm) and 
                    not inspect.isabstract(item)):
                    try:
                        algorithm = item()
                        method_name = algorithm.get_name()
                        self.plugins[method_name] = algorithm
                        
                        # V1.3.0: Check if separator line already exists, if not add it
                        if not hasattr(self, '_custom_section_added'):
                            print("📌 Adding custom algorithms section separator...")
                            self._add_custom_algorithms_section([])
                            self._custom_section_added = True
                        
                        # Add to UI with special styling
                        print(f"✅ Adding custom feature selection algorithm '{method_name}' to UI...")
                        self._add_custom_algorithm_ui(method_name)
                        print(f"✅ Custom feature selection algorithm '{method_name}' added to UI successfully")
                        print(f"📊 Total feature selection methods: {len(self.method_checkboxes)}")
                        algorithm_found = True
                        break
                    except Exception as e:
                        print(f"Failed to instantiate {item.__name__}: {e}")
                        continue
            
            if not algorithm_found:
                raise Exception("No valid algorithm class found in code")
                    
        except Exception as e:
            raise Exception(f"Error loading custom feature selector: {str(e)}")

class FeatureSelectionParameterDialog(QDialog):
    def __init__(self, method, plugins, parent=None):
        from app.views.responsive_dialog import ResponsiveDialog
        # Note: Due to special __init__ signature of this class, manually set responsive size
        super(FeatureSelectionParameterDialog, self).__init__(parent)
        self.setWindowTitle(f"Parameters for {method}")
        
        # Use responsive size instead of fixed width
        from app.utils.ui_scaling import ui_scaling_manager
        width, height = ui_scaling_manager.get_responsive_dialog_size(600, 500)
        self.resize(width, height)
        
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
  
            # V1.4.1: Fix method name - use get_params_info instead of get_parameter_info
            plugin_instance = self.plugins[method]
            if hasattr(plugin_instance, 'get_params_info'):
                param_info = plugin_instance.get_params_info()
            elif hasattr(plugin_instance, 'get_parameter_info'):
                param_info = plugin_instance.get_parameter_info()  # Fallback for compatibility
            else:
                raise AttributeError(f"Algorithm '{method}' has neither 'get_params_info' nor 'get_parameter_info' method")
            for param_name, info in param_info.items():
                widget = self.create_widget(info)
                form_layout.addRow(f"{param_name} ({info['description']}):", widget)
                self.param_widgets[param_name] = widget
        else:
  
            if "CARS" in method:
                # CARS (Competitive Adaptive Reweighted Sampling) Parameters
                n_iterations = QSpinBox()
                n_iterations.setRange(10, 200)
                n_iterations.setValue(50)
                n_iterations.setToolTip("Monte Carlo采样迭代次数 (推荐30-100)")
                form_layout.addRow("迭代次数 (Iterations):", n_iterations)
                self.param_widgets['n_iterations'] = n_iterations
                
                pls_components = QSpinBox()
                pls_components.setRange(2, 20)
                pls_components.setValue(5)
                pls_components.setToolTip("PLS成分数 (通常3-10)")
                form_layout.addRow("PLS成分数 (Components):", pls_components)
                self.param_widgets['pls_components'] = pls_components
                
                cv_folds = QSpinBox()
                cv_folds.setRange(3, 10)
                cv_folds.setValue(5)
                cv_folds.setToolTip("交叉验证折数")
                form_layout.addRow("交叉验证折数 (CV Folds):", cv_folds)
                self.param_widgets['n_folds'] = cv_folds
                
                sampling_ratio = QDoubleSpinBox()
                sampling_ratio.setRange(0.5, 1.0)
                sampling_ratio.setValue(0.9)
                sampling_ratio.setSingleStep(0.05)
                sampling_ratio.setToolTip("每次采样的样本比例")
                form_layout.addRow("采样比例 (Sampling Ratio):", sampling_ratio)
                self.param_widgets['sampling_ratio'] = sampling_ratio
                
            elif "SPA" in method:
                # SPA (Successive Projections Algorithm) Parameters
                n_wavelengths = QSpinBox()
                n_wavelengths.setRange(5, 100)
                n_wavelengths.setValue(20)
                n_wavelengths.setToolTip("目标波长数量 (推荐10-30)")
                form_layout.addRow("目标波长数 (Target Wavelengths):", n_wavelengths)
                self.param_widgets['n_wavelengths'] = n_wavelengths
                
                min_wavelengths = QSpinBox()
                min_wavelengths.setRange(3, 20)
                min_wavelengths.setValue(5)
                min_wavelengths.setToolTip("最小波长数")
                form_layout.addRow("最小波长数 (Minimum):", min_wavelengths)
                self.param_widgets['min_wavelengths'] = min_wavelengths
                
                max_wavelengths = QSpinBox()
                max_wavelengths.setRange(20, 200)
                max_wavelengths.setValue(50)
                max_wavelengths.setToolTip("最大波长数")
                form_layout.addRow("最大波长数 (Maximum):", max_wavelengths)
                self.param_widgets['max_wavelengths'] = max_wavelengths
                
            elif "PLSR" in method or "PLS" in method:
                # PLSR (Partial Least Squares Regression) Parameters
                n_components = QSpinBox()
                n_components.setRange(2, 20)
                n_components.setValue(5)
                n_components.setToolTip("PLS成分数 (推荐3-10)")
                form_layout.addRow("PLS成分数 (Components):", n_components)
                self.param_widgets['n_components'] = n_components
                
                k_features = QSpinBox()
                k_features.setRange(5, 200)
                k_features.setValue(30)
                k_features.setToolTip("选择的特征数量")
                form_layout.addRow("特征数量 (Features):", k_features)
                self.param_widgets['k'] = k_features
                
            elif method == "SelectKBest":
                k_spin = QSpinBox()
                k_spin.setRange(1, 100)
                k_spin.setValue(10)
                form_layout.addRow("Number of features (k):", k_spin)
                self.param_widgets['k'] = k_spin
                
                score_func = QComboBox()
                score_func.addItems(["f_classif", "chi2", "mutual_info_classif"])
                form_layout.addRow("Score function:", score_func)
                self.param_widgets['score_func'] = score_func
            elif method == "RFE" or "Recursive Feature Elimination" in method:
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
