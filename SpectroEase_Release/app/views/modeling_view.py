# app/views/modeling_view.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QMessageBox, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit, QHBoxLayout,
    QFrame, QCheckBox, QScrollArea, QGridLayout, QListWidget, QGroupBox, QTableWidget, QTableWidgetItem, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from interfaces.modeling_algorithm import ModelingAlgorithm
import pandas as pd
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas

class ModelingView(QWidget):
    def __init__(self, plugins: dict):
        super(ModelingView, self).__init__()
        self.plugins = plugins  # Dynamically loaded plugins
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)  # Reduced spacing between main layout elements
        self.layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        
        # Title
        title_label = QLabel("Algorithm Selection")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        self.layout.addWidget(title_label)
        
        # Description
        description = QLabel("Select modeling algorithm to use:")
        description.setStyleSheet("font-size: 11px; color: #555;")
        self.layout.addWidget(description)
        
        # Built-in methods
        built_in_methods = [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Decision Tree",
            "Gradient Boosting",
            "XGBoost",
            "Neural Network"
        ]
        
        # Plugin methods
        plugin_methods = list(self.plugins.keys())
        all_methods = built_in_methods + plugin_methods
        
        # Create algorithm list container
        methods_container = QFrame()
        methods_container.setFrameShape(QFrame.StyledPanel)
        methods_container.setStyleSheet("background-color: #f5f5f5; border-radius: 4px; padding: 6px;")
        methods_layout = QVBoxLayout(methods_container)
        methods_layout.setSpacing(2)  # Reduced spacing between algorithms
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
        
        # Add algorithm list to main layout
        self.layout.addWidget(methods_container)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0;")
        self.layout.addWidget(line)
        
        # Parameters area
        self.params_container = QFrame()
        self.params_container.setFrameShape(QFrame.StyledPanel)
        self.params_container.setStyleSheet("background-color: #f5f5f5; border-radius: 4px; padding: 6px;")
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(4, 4, 4, 4)
        self.params_layout.setSpacing(4)
        
        params_title = QLabel("Algorithm Parameters")
        params_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #2196F3;")
        self.params_layout.addWidget(params_title)
        
        self.params_form = QFormLayout()
        self.params_form.setSpacing(4)
        self.params_form.setContentsMargins(2, 2, 2, 2)
        self.params_layout.addLayout(self.params_form)
        
        self.layout.addWidget(self.params_container)
        self.params_container.setVisible(False)  # Initially hide parameters area
        
        # Apply button
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)
        self.train_button = QPushButton("Train Model")
        self.train_button.setStyleSheet("""
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
        button_layout.addWidget(self.train_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)
        
        # Initialize parameter widgets
        self.param_widgets = {}
        self.init_param_widgets()
        
        # Initialize evaluation results property
        self.evaluation_results = {}
        
        # Initialize UI - Only call this method after setting the main layout
        self.init_ui()
        
        # Set default value to "Random Forest" - directly select the checkbox
        if "Random Forest" in self.method_checkboxes:
            self.method_checkboxes["Random Forest"].setChecked(True)
            self.handle_checkbox_clicked("Random Forest", True)
    
    def handle_checkbox_clicked(self, method, checked):
        """Ensure only one checkbox is selected and display corresponding parameters"""
        if checked:
            for m, checkbox in self.method_checkboxes.items():
                if m != method:
                    checkbox.setChecked(False)
            self.on_model_change(method)
        else:
            # If unselected, hide parameters area
            self.params_container.setVisible(False)

    def init_param_widgets(self):
        # Initialize parameter widgets for built-in models
        # Logistic Regression parameters
        self.param_widgets['Logistic Regression'] = {
            'C': self.create_double_spin(0.01, 100.0, 1.0, 0.1, "Regularization Strength"),
            'max_iter': self.create_spin(100, 10000, 1000, 100, "Maximum Iterations")
        }

        # Random Forest parameters
        self.param_widgets['Random Forest'] = {
            'n_estimators': self.create_spin(10, 1000, 100, 10, "Number of Trees"),
            'max_depth': self.create_spin(1, 100, 10, 1, "Maximum Depth"),
            'min_samples_split': self.create_spin(2, 20, 2, 1, "Minimum Samples Required to Split an Internal Node")
        }
        
        # SVM parameters
        self.param_widgets['Support Vector Machine (SVM)'] = {
            'C': self.create_double_spin(0.01, 100.0, 1.0, 0.1, "Regularization Parameter"),
            'kernel': self.create_combo(['linear', 'poly', 'rbf', 'sigmoid'], "Kernel Function")
        }
        
        # KNN parameters
        self.param_widgets['K-Nearest Neighbors (KNN)'] = {
            'n_neighbors': self.create_spin(1, 50, 5, 1, "Number of Neighbors"),
            'weights': self.create_combo(['uniform', 'distance'], "Weight Function")
        }
        
        # Other built-in model parameters initialization
        self.param_widgets['Decision Tree'] = {
            'max_depth': self.create_spin(1, 100, 10, 1, "Maximum Depth"),
            'min_samples_split': self.create_spin(2, 20, 2, 1, "Minimum Samples Required to Split an Internal Node")
        }
        
        self.param_widgets['Gradient Boosting'] = {
            'n_estimators': self.create_spin(10, 1000, 100, 10, "Number of Weak Learners"),
            'learning_rate': self.create_double_spin(0.001, 1.0, 0.1, 0.01, "Learning Rate"),
            'max_depth': self.create_spin(1, 10, 3, 1, "Maximum Depth")
        }
        
        self.param_widgets['XGBoost'] = {
            'n_estimators': self.create_spin(10, 1000, 100, 10, "Number of Weak Learners"),
            'learning_rate': self.create_double_spin(0.001, 1.0, 0.1, 0.01, "Learning Rate"),
            'max_depth': self.create_spin(1, 10, 3, 1, "Maximum Depth")
        }
        
        self.param_widgets['Neural Network'] = {
            'hidden_layer_sizes': self.create_line_edit("100,100", "Hidden Layer Sizes (Comma Separated)"),
            'activation': self.create_combo(['relu', 'tanh', 'logistic'], "Activation Function"),
            'learning_rate_init': self.create_double_spin(0.0001, 0.1, 0.001, 0.0001, "Initial Learning Rate")
        }

    def create_spin(self, min_val, max_val, default, step, tooltip=""):
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(default)
        widget.setSingleStep(step)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget
    
    def create_double_spin(self, min_val, max_val, default, step, tooltip=""):
        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(default)
        widget.setSingleStep(step)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget
    
    def create_combo(self, items, tooltip=""):
        widget = QComboBox()
        widget.addItems(items)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget
    
    def create_line_edit(self, default, tooltip=""):
        widget = QLineEdit()
        widget.setText(default)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget
    
    def show_parameters(self, method):
        """Display parameter setting dialog"""
        self.on_model_change(method)
        # Ensure corresponding checkbox is selected
        if method in self.method_checkboxes:
            self.method_checkboxes[method].setChecked(True)
            self.handle_checkbox_clicked(method, True)

    def on_model_change(self, model):
        # Clear current parameter form
        for i in reversed(range(self.params_form.count())):
            self.params_form.removeRow(i)

        if model in self.param_widgets:
            # Built-in model
            for param_name, widget in self.param_widgets[model].items():
                self.params_form.addRow(f"{param_name}:", widget)
            self.params_container.setVisible(True)
        elif model in self.plugins:
            # Plugin model
            param_info = self.plugins[model].get_parameter_info()
            for param_name, info in param_info.items():
                widget = self.create_widget(info)
                self.params_form.addRow(f"{param_name} ({info['description']}):", widget)
                self.param_widgets[param_name] = widget
            self.params_container.setVisible(True)
        else:
            self.params_container.setVisible(False)

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

    def get_selected_model(self):
        """获取选择的模型method"""
        for method, checkbox in self.method_checkboxes.items():
            if checkbox.isChecked():
                return method
  
        return "Random Forest"
        
    def get_parameters(self):
        """获取所有parameters，兼容性method"""
        return self.get_model_parameters()

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
        """Add custom modeling algorithm"""
        try:
            import types
            mod = types.ModuleType('custom_model')
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
                        if isinstance(widget, QFrame) and widget != self.params_container:
                            widget.layout().addWidget(method_frame)
                            break
                    
                    break
                    
        except Exception as e:
            raise Exception(f"Error loading custom model: {str(e)}")

    def init_ui(self):
        """Initialize UI interface"""
  
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
  
        app_font = QFont("Microsoft YaHei UI", 9)
        self.setFont(app_font)
        
  
        min_label_width = 70
        min_combobox_width = 110
        
  
        tabs = QTabWidget()
        tabs.setDocumentMode(True)  # 更现代的外观
        
  
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(5)
        basic_layout.setContentsMargins(8, 8, 8, 8)
        
  
        model_selection_group = QGroupBox("Model Selection")
        model_selection_layout = QVBoxLayout()
        model_selection_layout.setSpacing(5)
        model_selection_layout.setContentsMargins(8, 10, 8, 8)
        
  
        analysis_type_row = QHBoxLayout()
        analysis_type_label = QLabel("Analysis Type:")
        analysis_type_label.setMinimumWidth(min_label_width)
        analysis_type_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        analysis_type_row.addWidget(analysis_type_label)
        
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Quantitative", "Qualitative"])
        self.analysis_type.setMinimumWidth(min_combobox_width)
        analysis_type_row.addWidget(self.analysis_type)
        model_selection_layout.addLayout(analysis_type_row)
        
  
        model_method_row = QHBoxLayout()
        model_method_label = QLabel("Model Method:")
        model_method_label.setMinimumWidth(min_label_width)
        model_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_method_row.addWidget(model_method_label)
        
        self.model_method = QComboBox()
  
        self.model_method.addItems(["PLSR", "SVR", "RF", "XGBoost", "NN"])
        self.model_method.setMinimumWidth(min_combobox_width)
        model_method_row.addWidget(self.model_method)
        model_selection_layout.addLayout(model_method_row)
        
        model_selection_group.setLayout(model_selection_layout)
        basic_layout.addWidget(model_selection_group)
        
  
        hyperparameter_group = QGroupBox("Hyperparameter Optimization")
        hyperparameter_layout = QVBoxLayout()
        hyperparameter_layout.setSpacing(5)
        hyperparameter_layout.setContentsMargins(8, 10, 8, 8)
        
  
        opt_method_row = QHBoxLayout()
        opt_method_label = QLabel("Optimization Method:")
        opt_method_label.setMinimumWidth(min_label_width + 30)
        opt_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        opt_method_row.addWidget(opt_method_label)
        
        self.optimization_method = QComboBox()
        self.optimization_method.addItems(["Grid Search", "Random Search", "Bayesian Optimization"])
        self.optimization_method.setMinimumWidth(min_combobox_width)
        opt_method_row.addWidget(self.optimization_method)
        hyperparameter_layout.addLayout(opt_method_row)
        
  
        metric_row = QHBoxLayout()
        metric_label = QLabel("Evaluation Metric:")
        metric_label.setMinimumWidth(min_label_width + 30)
        metric_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        metric_row.addWidget(metric_label)
        
        self.evaluation_metric = QComboBox()
        self.evaluation_metric.addItems(["RMSE", "R²", "MAE", "Accuracy", "F1 Score"])
        self.evaluation_metric.setMinimumWidth(min_combobox_width)
        metric_row.addWidget(self.evaluation_metric)
        hyperparameter_layout.addLayout(metric_row)
        
  
        cv_row = QHBoxLayout()
        cv_label = QLabel("Cross-Validation Folds:")
        cv_label.setMinimumWidth(min_label_width + 30)
        cv_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cv_row.addWidget(cv_label)
        
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        self.cv_folds.setMinimumWidth(60)
        cv_row.addWidget(self.cv_folds)
        hyperparameter_layout.addLayout(cv_row)
        
  
        iter_row = QHBoxLayout()
        iter_label = QLabel("Maximum Iterations:")
        iter_label.setMinimumWidth(min_label_width + 30)
        iter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        iter_row.addWidget(iter_label)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 1000)
        self.max_iterations.setValue(100)
        self.max_iterations.setSingleStep(10)
        self.max_iterations.setMinimumWidth(60)
        iter_row.addWidget(self.max_iterations)
        hyperparameter_layout.addLayout(iter_row)
        
        hyperparameter_group.setLayout(hyperparameter_layout)
        basic_layout.addWidget(hyperparameter_group)
        
  
        training_control_group = QGroupBox("Training Control")
        training_control_layout = QVBoxLayout()
        training_control_layout.setSpacing(5)
        training_control_layout.setContentsMargins(8, 10, 8, 8)
        
  
        split_row = QHBoxLayout()
        split_label = QLabel("Test Split Ratio:")
        split_label.setMinimumWidth(min_label_width + 30)
        split_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        split_row.addWidget(split_label)
        
        self.test_split = QDoubleSpinBox()
        self.test_split.setRange(0.1, 0.5)
        self.test_split.setValue(0.2)
        self.test_split.setSingleStep(0.05)
        self.test_split.setMinimumWidth(60)
        split_row.addWidget(self.test_split)
        training_control_layout.addLayout(split_row)
        
  
        seed_row = QHBoxLayout()
        seed_label = QLabel("Random Seed:")
        seed_label.setMinimumWidth(min_label_width + 30)
        seed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seed_row.addWidget(seed_label)
        
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 1000)
        self.random_seed.setValue(42)
        self.random_seed.setMinimumWidth(60)
        seed_row.addWidget(self.random_seed)
        training_control_layout.addLayout(seed_row)
        
  
        standardize_row = QHBoxLayout()
        self.standardize_check = QCheckBox("Standardize Features")
        self.standardize_check.setChecked(True)
        standardize_row.addWidget(self.standardize_check)
        training_control_layout.addLayout(standardize_row)
        
        training_control_group.setLayout(training_control_layout)
        basic_layout.addWidget(training_control_group)
        
        basic_tab.setLayout(basic_layout)
        tabs.addTab(basic_tab, "Model Setup")
        
  
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(5)
        advanced_layout.setContentsMargins(8, 8, 8, 8)
        
  
        model_params_group = QGroupBox("Model Specific Parameters")
        model_params_layout = QVBoxLayout()
        model_params_layout.setSpacing(5)
        model_params_layout.setContentsMargins(8, 10, 8, 8)
        
  
        self.model_params_widget = QWidget()
        self.model_params_widget.setLayout(QVBoxLayout())
        model_params_layout.addWidget(self.model_params_widget)
        
        model_params_group.setLayout(model_params_layout)
        advanced_layout.addWidget(model_params_group)
        
  
        feature_selection_group = QGroupBox("Feature Selection")
        feature_selection_layout = QVBoxLayout()
        feature_selection_layout.setSpacing(5)
        feature_selection_layout.setContentsMargins(8, 10, 8, 8)
        
  
        enable_fs_row = QHBoxLayout()
        self.enable_feature_selection = QCheckBox("Enable Feature Selection")
        enable_fs_row.addWidget(self.enable_feature_selection)
        feature_selection_layout.addLayout(enable_fs_row)
        
  
        fs_method_row = QHBoxLayout()
        fs_method_label = QLabel("Method:")
        fs_method_label.setMinimumWidth(min_label_width)
        fs_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fs_method_row.addWidget(fs_method_label)
        
        self.fs_method = QComboBox()
        self.fs_method.addItems(["Recursive Feature Elimination", "LASSO", "Random Forest", "Mutual Information"])
        self.fs_method.setMinimumWidth(min_combobox_width + 40)
        fs_method_row.addWidget(self.fs_method)
        feature_selection_layout.addLayout(fs_method_row)
        
  
        feature_num_row = QHBoxLayout()
        feature_num_label = QLabel("Number of Features:")
        feature_num_label.setMinimumWidth(min_label_width + 30)
        feature_num_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        feature_num_row.addWidget(feature_num_label)
        
        self.feature_num = QSpinBox()
        self.feature_num.setRange(1, 100)
        self.feature_num.setValue(10)
        self.feature_num.setMinimumWidth(60)
        feature_num_row.addWidget(self.feature_num)
        feature_selection_layout.addLayout(feature_num_row)
        
        feature_selection_group.setLayout(feature_selection_layout)
        advanced_layout.addWidget(feature_selection_group)
        
        advanced_tab.setLayout(advanced_layout)
        tabs.addTab(advanced_tab, "Advanced")
        
  
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_layout.setSpacing(5)
        results_layout.setContentsMargins(8, 8, 8, 8)
        
  
        evaluation_group = QGroupBox("Model Evaluation")
        evaluation_layout = QVBoxLayout()
        evaluation_layout.setSpacing(5)
        evaluation_layout.setContentsMargins(8, 10, 8, 8)
        
  
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        evaluation_layout.addWidget(self.results_table)
        
        evaluation_group.setLayout(evaluation_layout)
        results_layout.addWidget(evaluation_group)
        
  
        curve_group = QGroupBox("Learning Curve")
        curve_layout = QVBoxLayout()
        curve_layout.setSpacing(5)
        curve_layout.setContentsMargins(8, 10, 8, 8)
        
  
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.setBackground('w')
        self.curve_plot.setLabel('left', 'Performance')
        self.curve_plot.setLabel('bottom', 'Training Examples')
        self.curve_plot.showGrid(x=True, y=True)
        curve_layout.addWidget(self.curve_plot)
        
        curve_group.setLayout(curve_layout)
        results_layout.addWidget(curve_group)
        
        results_tab.setLayout(results_layout)
        tabs.addTab(results_tab, "Results")
        
        main_layout.addWidget(tabs)
        
  
        buttons_layout = QHBoxLayout()
        
  
        self.train_button = QPushButton("Train Model")
        buttons_layout.addWidget(self.train_button)
        
  
        self.predict_button = QPushButton("Make Predictions")
        buttons_layout.addWidget(self.predict_button)
        
  
        self.save_model_button = QPushButton("Save Model")
        buttons_layout.addWidget(self.save_model_button)
        
  
        self.load_model_button = QPushButton("Load Model")
        buttons_layout.addWidget(self.load_model_button)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
        
    def update_model_methods(self, analysis_type):
        """根据analysis类型更新可用的模型method"""
        self.model_method.clear()
        if analysis_type == "Quantitative":
            self.model_method.addItems(["PLSR", "SVR", "Random Forest", "XGBoost", "Neural Network"])
            self.evaluation_metric.clear()
            self.evaluation_metric.addItems(["RMSE", "R²", "MAE"])
        else:  # Qualitative
            self.model_method.addItems(["SVM", "Random Forest", "XGBoost", "Neural Network", "LDA"])
            self.evaluation_metric.clear()
            self.evaluation_metric.addItems(["Accuracy", "F1 Score", "Precision", "Recall"])
    
    def update_model_params(self, model_method):
        """根据所选模型更新模型特定parametersUI"""
  
        layout = self.model_params_widget.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
  
        if model_method == "PLSR":
            # PLSRparameters
            components_row = QHBoxLayout()
            components_label = QLabel("Number of Components:")
            components_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            components_row.addWidget(components_label)
            
            self.plsr_components = QSpinBox()
            self.plsr_components.setRange(1, 20)
            self.plsr_components.setValue(5)
            components_row.addWidget(self.plsr_components)
            layout.addLayout(components_row)
        
        elif model_method == "SVR" or model_method == "SVM":
            # SVM/SVRparameters
            kernel_row = QHBoxLayout()
            kernel_label = QLabel("Kernel:")
            kernel_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            kernel_row.addWidget(kernel_label)
            
            self.svm_kernel = QComboBox()
            self.svm_kernel.addItems(["linear", "poly", "rbf", "sigmoid"])
            kernel_row.addWidget(self.svm_kernel)
            layout.addLayout(kernel_row)
            
            c_row = QHBoxLayout()
            c_label = QLabel("C Parameter:")
            c_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            c_row.addWidget(c_label)
            
            self.svm_c = QDoubleSpinBox()
            self.svm_c.setRange(0.1, 100.0)
            self.svm_c.setValue(1.0)
            self.svm_c.setSingleStep(0.1)
            c_row.addWidget(self.svm_c)
            layout.addLayout(c_row)
        
        elif model_method == "Random Forest":
  
            trees_row = QHBoxLayout()
            trees_label = QLabel("Number of Trees:")
            trees_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            trees_row.addWidget(trees_label)
            
            self.rf_trees = QSpinBox()
            self.rf_trees.setRange(10, 500)
            self.rf_trees.setValue(100)
            self.rf_trees.setSingleStep(10)
            trees_row.addWidget(self.rf_trees)
            layout.addLayout(trees_row)
            
            depth_row = QHBoxLayout()
            depth_label = QLabel("Max Depth:")
            depth_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            depth_row.addWidget(depth_label)
            
            self.rf_depth = QSpinBox()
            self.rf_depth.setRange(1, 50)
            self.rf_depth.setValue(10)
            depth_row.addWidget(self.rf_depth)
            layout.addLayout(depth_row)
        
        elif model_method == "XGBoost":
            # XGBoostparameters
            lr_row = QHBoxLayout()
            lr_label = QLabel("Learning Rate:")
            lr_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lr_row.addWidget(lr_label)
            
            self.xgb_lr = QDoubleSpinBox()
            self.xgb_lr.setRange(0.01, 1.0)
            self.xgb_lr.setValue(0.1)
            self.xgb_lr.setSingleStep(0.01)
            lr_row.addWidget(self.xgb_lr)
            layout.addLayout(lr_row)
            
            depth_row = QHBoxLayout()
            depth_label = QLabel("Max Depth:")
            depth_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            depth_row.addWidget(depth_label)
            
            self.xgb_depth = QSpinBox()
            self.xgb_depth.setRange(1, 20)
            self.xgb_depth.setValue(6)
            depth_row.addWidget(self.xgb_depth)
            layout.addLayout(depth_row)
            
            iterations_row = QHBoxLayout()
            iterations_label = QLabel("Number of Boosting Rounds:")
            iterations_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            iterations_row.addWidget(iterations_label)
            
            self.xgb_rounds = QSpinBox()
            self.xgb_rounds.setRange(10, 500)
            self.xgb_rounds.setValue(100)
            self.xgb_rounds.setSingleStep(10)
            iterations_row.addWidget(self.xgb_rounds)
            layout.addLayout(iterations_row)
        
        elif model_method == "Neural Network":
  
            layers_row = QHBoxLayout()
            layers_label = QLabel("Hidden Layers:")
            layers_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layers_row.addWidget(layers_label)
            
            self.nn_layers = QSpinBox()
            self.nn_layers.setRange(1, 5)
            self.nn_layers.setValue(2)
            layers_row.addWidget(self.nn_layers)
            layout.addLayout(layers_row)
            
            neurons_row = QHBoxLayout()
            neurons_label = QLabel("Neurons per Layer:")
            neurons_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            neurons_row.addWidget(neurons_label)
            
            self.nn_neurons = QSpinBox()
            self.nn_neurons.setRange(5, 100)
            self.nn_neurons.setValue(20)
            self.nn_neurons.setSingleStep(5)
            neurons_row.addWidget(self.nn_neurons)
            layout.addLayout(neurons_row)
            
            lr_row = QHBoxLayout()
            lr_label = QLabel("Learning Rate:")
            lr_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lr_row.addWidget(lr_label)
            
            self.nn_lr = QDoubleSpinBox()
            self.nn_lr.setRange(0.0001, 0.1)
            self.nn_lr.setValue(0.001)
            self.nn_lr.setSingleStep(0.0001)
            self.nn_lr.setDecimals(4)
            lr_row.addWidget(self.nn_lr)
            layout.addLayout(lr_row)
            
            epochs_row = QHBoxLayout()
            epochs_label = QLabel("Epochs:")
            epochs_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            epochs_row.addWidget(epochs_label)
            
            self.nn_epochs = QSpinBox()
            self.nn_epochs.setRange(10, 1000)
            self.nn_epochs.setValue(100)
            self.nn_epochs.setSingleStep(10)
            epochs_row.addWidget(self.nn_epochs)
            layout.addLayout(epochs_row)
    
    def get_model_parameters(self):
        """获取模型parameters"""
        try:
  
            model = self.get_selected_model()
  
            params = {}
            
            if model in self.param_widgets:
  
                for param_name, widget in self.param_widgets[model].items():
                    if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                        params[param_name] = widget.value()
                    elif isinstance(widget, QLineEdit):
                        text = widget.text()
  
                        if ',' in text:
                            try:
  
                                params[param_name] = tuple(int(x.strip()) for x in text.split(','))
                            except:
                                params[param_name] = text
                        else:
                            params[param_name] = text
                    elif isinstance(widget, QComboBox):
                        params[param_name] = widget.currentText()
                    elif isinstance(widget, QCheckBox):
                        params[param_name] = widget.isChecked()
            elif model in self.plugins:
  
                param_info = self.plugins[model].get_parameter_info()
                for param_name, info in param_info.items():
                    widget = self.param_widgets.get(param_name)
                    if widget:
                        if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                            params[param_name] = widget.value()
                        elif isinstance(widget, QLineEdit):
                            params[param_name] = widget.text()
                        elif isinstance(widget, QComboBox):
                            params[param_name] = widget.currentText()
                        elif isinstance(widget, QCheckBox):
                            params[param_name] = widget.isChecked()
            
            return params
        except Exception as e:
            self.display_error(f"Error getting model parameters: {str(e)}")
            return {}
            
    def update_evaluation_results(self, results):
        """Update evaluation results display"""
        try:
            # Save evaluation results
            self.evaluation_results = results
            
            # Clear existing evaluation results
            self.results_table.clearContents()
            self.results_table.setRowCount(0)
            
            # Check if results are empty
            if not results or not isinstance(results, dict):
                print("No evaluation results or incorrect format")
                return
            
            # Detect task type
            task_type = results.get('task_type', None)
            
            # Print task type and result keys
            print(f"Evaluation results - Task type: {task_type}")
            print(f"Evaluation results contain the following metrics: {list(results.keys())}")
            
            # Add evaluation metrics to table
            row = 0
            for key, value in results.items():
                # Skip keys that should not be displayed
                if key in ['task_type', 'Confusion Matrix', 'Classification Report', 'y_true_encoded', 'y_pred_encoded']:
                    continue
                
                # Add evaluation metric to table
                self.results_table.insertRow(row)
                
                # Set metric name
                metric_item = QTableWidgetItem(key)
                self.results_table.setItem(row, 0, metric_item)
                
                # Set metric value (based on value type)
                if isinstance(value, (int, float)):
                    # Numeric types formatted with 4 decimal places
                    value_item = QTableWidgetItem(f"{value:.4f}")
                else:
                    # Non-numeric types converted to string
                    value_item = QTableWidgetItem(str(value))
                
                self.results_table.setItem(row, 1, value_item)
                row += 1
            
            # Adjust table size
            self.results_table.resizeColumnsToContents()
            
            # If classification task and confusion matrix exists, display it
            if task_type == 'classification' and 'Confusion Matrix' in results:
                self.display_confusion_matrix(results['Confusion Matrix'])
            
            # If classification report exists, display it
            if 'Classification Report' in results:
                self.display_classification_report(results['Classification Report'])
        except Exception as e:
            import traceback
            error_msg = f"Error updating evaluation results: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Display error message
            self.display_error(f"Cannot display evaluation results: {str(e)}")
        
    def display_confusion_matrix(self, confusion_matrix):
        """Display confusion matrix"""
        try:
            # Ensure confusion matrix is valid
            if not isinstance(confusion_matrix, (list, np.ndarray)):
                return
            
            # Convert to numpy array
            if isinstance(confusion_matrix, list):
                cm = np.array(confusion_matrix)
            else:
                cm = confusion_matrix
            
            # Clear existing chart
            for i in reversed(range(self.curve_plot.layout().count())): 
                self.curve_plot.layout().itemAt(i).widget().setParent(None)
            
            # Create confusion matrix chart
            figure = plt.figure(figsize=(5, 5))
            ax = figure.add_subplot(111)
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            figure.colorbar(cax)
            
            # Set labels
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            # Display values in each cell
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), va='center', ha='center')
            
            # Create Qt widget to display the chart
            canvas = FigureCanvas(figure)
            self.curve_plot.layout().addWidget(canvas)
        except Exception as e:
            print(f"Error displaying confusion matrix: {e}")
        
    def display_classification_report(self, report):
        """Display classification report"""
        try:
            # Clear existing report
            self.results_table.clearContents()
            self.results_table.setRowCount(0)
            
            # Set monospace font to maintain formatting
            font = QFont("Courier New")
            self.results_table.horizontalHeader().setFont(font)
            self.results_table.verticalHeader().setFont(font)
            
            # Display report
            for row, (metric, value) in enumerate(report.items()):
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(metric))
                self.results_table.setItem(row, 1, QTableWidgetItem(value))
            
            # Adjust table size
            self.results_table.resizeColumnsToContents()
        except Exception as e:
            print(f"Error displaying classification report: {e}")
