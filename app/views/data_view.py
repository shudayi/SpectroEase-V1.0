# app/views/data_view.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableView, QComboBox, QPushButton, QFormLayout,
    QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox, QCheckBox, QGroupBox, QGridLayout,
    QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
import pandas as pd


class DataView(QWidget):
    def __init__(self):
        super(DataView, self).__init__()
        self.layout = QVBoxLayout()

        self.label = QLabel("Data Preview:")
        self.layout.addWidget(self.label)

        self.table_view = QTableView()
        self.layout.addWidget(self.table_view)

        self.setLayout(self.layout)

    def update_data(self, data: pd.DataFrame):
        if data.empty:
            self.label.setText("No data available.")
            self.table_view.setModel(None)
            return

        self.label.setText("Data Preview:")
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(data.columns.tolist())
        for row in data.itertuples(index=False):
            items = [QStandardItem(str(field)) for field in row]
            model.appendRow(items)
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()


class DataPartitioningView(QWidget):
    def __init__(self, data_partitioning_plugins: dict):
        super(DataPartitioningView, self).__init__()
        self.plugins = data_partitioning_plugins
        self.layout = QVBoxLayout()
        self.layout.setSpacing(8)  # Standard spacing
        self.layout.setContentsMargins(4, 4, 4, 4)  # Standard margins

        # Method selection with label
        method_layout = QHBoxLayout()
        method_layout.setSpacing(8)  # Standard spacing
        
        # Add label for better visual balance
        method_label = QLabel("Method:")
        method_label.setFixedWidth(60)
        method_layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(180)  # Wider for better visibility
        self.method_combo.setFixedHeight(24)  # Standard input control height
  
        built_in_methods = ["Train-Test Split", "K-Fold", "LOGO", "Random", "Stratified"]
        plugin_methods = list(self.plugins.keys())
        self.method_combo.addItems(built_in_methods + plugin_methods)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()  # Push to left for better alignment
  
        self.layout.addLayout(method_layout)

  
        self.params_form = QFormLayout()
        self.params_form.setSpacing(8)  # Standard spacing
        self.params_form.setContentsMargins(0, 4, 0, 0)  # Top margin for separation
        self.param_widgets = {}
        self.init_param_widgets("Train-Test Split")
        self.layout.addLayout(self.params_form)

        self.setLayout(self.layout)

  
        self.method_combo.currentTextChanged.connect(self.on_method_change)

    def init_param_widgets(self, method):
  
        for i in reversed(range(self.params_form.count())):
            self.params_form.removeRow(i)
        self.param_widgets = {}

        if method in self.plugins:
            # Fix: Instantiate plugin class (plugin loader returns class)
            plugin_class = self.plugins[method]
            plugin_instance = plugin_class()
            param_info = plugin_instance.get_params_info()
  
            horizontal_layout = QHBoxLayout()
            horizontal_layout.setSpacing(10)
            
            for param_name, info in param_info.items():
  
                param_container = QHBoxLayout()
                param_container.setSpacing(10)
                
  
                label = QLabel(f"{param_name}:")
                label.setMaximumWidth(60)  # Adjust to required width
                param_container.addWidget(label)
                
  
                widget = self.create_widget(info)
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox) or isinstance(widget, QComboBox):
                    widget.setMaximumWidth(60)  # Reduce width
                param_container.addWidget(widget)
                self.param_widgets[param_name] = widget
                
  
                horizontal_layout.addLayout(param_container)
            
  
            self.params_form.addRow(horizontal_layout)
        else:
  
            if method == "Train-Test Split":
  
                params_container = QHBoxLayout()
                params_container.setSpacing(6)
                
  
                test_size_container = QHBoxLayout()
                test_size_container.setSpacing(5)
                test_size_label = QLabel("Test:")
                test_size_label.setMaximumWidth(40)  # Adjust Test label width
                test_size_container.addWidget(test_size_label)
                
                self.test_size_spin = QDoubleSpinBox()
                self.test_size_spin.setRange(0.01, 0.99)
                self.test_size_spin.setValue(0.2)
                self.test_size_spin.setSingleStep(0.05)
                self.test_size_spin.setMaximumWidth(60)  # Increase width to display decimals
                test_size_container.addWidget(self.test_size_spin)
                params_container.addLayout(test_size_container)
                
                params_container.addStretch()  # Add stretch for even distribution

  
                random_container = QHBoxLayout()
                random_container.setSpacing(5)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(40)  # Adjust Seed label width
                random_container.addWidget(random_label)
                
                self.random_state_spin = QSpinBox()
                self.random_state_spin.setRange(0, 10000)
                self.random_state_spin.setValue(42)
                self.random_state_spin.setMaximumWidth(70)
                random_container.addWidget(self.random_state_spin)
                params_container.addLayout(random_container)
                
                params_container.addStretch()

  
                min_samples_container = QHBoxLayout()
                min_samples_container.setSpacing(5)
                min_samples_label = QLabel("Min:")
                min_samples_label.setMaximumWidth(40)
                min_samples_container.addWidget(min_samples_label)
                
                self.min_samples_spin = QSpinBox()
                self.min_samples_spin.setRange(1, 100)
                self.min_samples_spin.setValue(5)
                self.min_samples_spin.setMaximumWidth(60)  # Increase width to display numbers
                min_samples_container.addWidget(self.min_samples_spin)
                params_container.addLayout(min_samples_container)
                
                params_container.addStretch()  # Add stretch for even distribution

  
                shuffle_container = QHBoxLayout()
                shuffle_container.setSpacing(5)
                shuffle_label = QLabel("Shuffle:")
                shuffle_label.setMaximumWidth(55)  # Adjust Shuffle label width
                shuffle_container.addWidget(shuffle_label)
                
                self.shuffle_checkbox = QCheckBox()
                self.shuffle_checkbox.setChecked(True)
                self.shuffle_checkbox.setMaximumWidth(15)
                shuffle_container.addWidget(self.shuffle_checkbox)
                params_container.addLayout(shuffle_container)
                
                params_container.addStretch()

  
                stratify_container = QHBoxLayout()
                stratify_container.setSpacing(5)
                stratify_label = QLabel("Stratify:")
                stratify_label.setMaximumWidth(60)  # Increase Stratify label width to display full text
                stratify_container.addWidget(stratify_label)
                
                self.stratify_combo = QComboBox()
                self.stratify_combo.addItem("None")
                self.stratify_combo.addItem("label")  # If there is a labels column
                self.stratify_combo.setMaximumWidth(90)
                stratify_container.addWidget(self.stratify_combo)
                params_container.addLayout(stratify_container)
                
  
                self.params_form.addRow(params_container)
            
            elif method == "K-Fold":
  
                params_container = QHBoxLayout()
                params_container.setSpacing(2)
                
  
                k_container = QHBoxLayout()
                k_container.setSpacing(1)
                k_label = QLabel("K:")
                k_label.setMaximumWidth(15)  # Reduce width to 15
                k_container.addWidget(k_label)
                
                self.k_fold_spin = QSpinBox()
                self.k_fold_spin.setRange(2, 10)
                self.k_fold_spin.setValue(5)
                self.k_fold_spin.setMaximumWidth(35)  # Reduce width to 35
                k_container.addWidget(self.k_fold_spin)
                params_container.addLayout(k_container)
                
  
                random_container = QHBoxLayout()
                random_container.setSpacing(1)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(25)  # Reduce width to 25
                random_container.addWidget(random_label)
                
                self.kf_random_state_spin = QSpinBox()
                self.kf_random_state_spin.setRange(0, 10000)
                self.kf_random_state_spin.setValue(42)
                self.kf_random_state_spin.setMaximumWidth(40)  # Reduce width to 40
                random_container.addWidget(self.kf_random_state_spin)
                params_container.addLayout(random_container)
                
  
                shuffle_container = QHBoxLayout()
                shuffle_container.setSpacing(1)
                shuffle_label = QLabel("Shuffle:")
                shuffle_label.setMaximumWidth(35)  # Reduce width to 35
                shuffle_container.addWidget(shuffle_label)
                
                self.kf_shuffle_checkbox = QCheckBox()
                self.kf_shuffle_checkbox.setChecked(True)
                self.kf_shuffle_checkbox.setMaximumWidth(15)  # Reduce width to 15
                shuffle_container.addWidget(self.kf_shuffle_checkbox)
                params_container.addLayout(shuffle_container)
                
  
                self.params_form.addRow(params_container)
            
            elif method == "LOGO":
                # Leave-One-Group-Out
                params_container = QHBoxLayout()
                params_container.setSpacing(2)
                
  
                group_container = QHBoxLayout()
                group_container.setSpacing(1)
                group_label = QLabel("Group:")
                group_label.setMaximumWidth(30)  # Reduce width to 30
                group_container.addWidget(group_label)
                
                self.logo_group_combo = QComboBox()
                self.logo_group_combo.addItem("label")
                self.logo_group_combo.addItem("batch")
                self.logo_group_combo.setMaximumWidth(45)  # Reduce width to 45
                group_container.addWidget(self.logo_group_combo)
                params_container.addLayout(group_container)
                
  
                self.params_form.addRow(params_container)
            
            elif method == "Random" or method == "Stratified":
  
                params_container = QHBoxLayout()
                params_container.setSpacing(2)
                
  
                n_container = QHBoxLayout()
                n_container.setSpacing(1)
                n_label = QLabel("N:")
                n_label.setMaximumWidth(15)  # Reduce width to 15
                n_container.addWidget(n_label)
                
                self.n_splits_spin = QSpinBox()
                self.n_splits_spin.setRange(2, 10)
                self.n_splits_spin.setValue(5)
                self.n_splits_spin.setMaximumWidth(35)  # Reduce width to 35
                n_container.addWidget(self.n_splits_spin)
                params_container.addLayout(n_container)
                
  
                test_size_container = QHBoxLayout()
                test_size_container.setSpacing(1)
                test_size_label = QLabel("Test:")
                test_size_label.setMaximumWidth(25)  # Reduce width to 25
                test_size_container.addWidget(test_size_label)
                
                self.random_test_size_spin = QDoubleSpinBox()
                self.random_test_size_spin.setRange(0.01, 0.99)
                self.random_test_size_spin.setValue(0.2)
                self.random_test_size_spin.setSingleStep(0.05)
                self.random_test_size_spin.setMaximumWidth(40)  # Reduce width to 40
                test_size_container.addWidget(self.random_test_size_spin)
                params_container.addLayout(test_size_container)
                
  
                random_container = QHBoxLayout()
                random_container.setSpacing(1)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(25)  # Reduce width to 25
                random_container.addWidget(random_label)
                
                self.random_state_seed_spin = QSpinBox()
                self.random_state_seed_spin.setRange(0, 10000)
                self.random_state_seed_spin.setValue(42)
                self.random_state_seed_spin.setMaximumWidth(40)  # Reduce width to 40
                random_container.addWidget(self.random_state_seed_spin)
                params_container.addLayout(random_container)
                
  
                self.params_form.addRow(params_container)

    def create_widget(self, info):
        if info['type'] == 'int':
            widget = QSpinBox()
            widget.setRange(info.get('min', 0), info.get('max', 1000))
            widget.setValue(info.get('default', 0))
        elif info['type'] == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(info.get('min', 0.0), info.get('max', 1000.0))
            widget.setValue(info.get('default', 0.0))
        elif info['type'] == 'str':
            widget = QLineEdit()
            widget.setText(info.get('default', ''))
        elif info['type'] == 'bool':
            widget = QCheckBox()
            widget.setChecked(info.get('default', False))
        else:
            widget = QLineEdit()  # Default to text input box
        return widget

    def on_method_change(self, method):
        self.init_param_widgets(method)
    
    def _is_widget_valid(self, widget):
        """V1.4.0: Check if a Qt widget is still valid (not deleted)"""
        try:
            # Try to access a property - will raise RuntimeError if deleted
            widget.isVisible()
            return True
        except (RuntimeError, AttributeError):
            return False

    def get_selected_method(self):
        # V1.3.1: Prefer userData (real name), if not available use display name
        method = self.method_combo.currentData()
        if method is None:  # Built-in algorithms don't have userData
            method = self.method_combo.currentText()
        return method

    def get_parameters(self):
        """
        Get all parameters.
        
        Returns:
            dict: A dictionary of parameters.
        """
  
        # V1.3.1: Use get_selected_method() to get real algorithm name
        method = self.get_selected_method()
        
  
        params = {}
  
        if method == "Train-Test Split":
            # V1.4.0: Safe widget access with validation
            params.update({
                'test_size': self.test_size_spin.value() if hasattr(self, 'test_size_spin') and self._is_widget_valid(self.test_size_spin) else 0.2,
                'random_state': self.random_state_spin.value() if hasattr(self, 'random_state_spin') and self._is_widget_valid(self.random_state_spin) else 42,
                'shuffle': self.shuffle_checkbox.isChecked() if hasattr(self, 'shuffle_checkbox') and self._is_widget_valid(self.shuffle_checkbox) else True,
                'min_samples': self.min_samples_spin.value() if hasattr(self, 'min_samples_spin') and self._is_widget_valid(self.min_samples_spin) else 1,
                'target_column': self.stratify_combo.currentIndex() if hasattr(self, 'stratify_combo') and self._is_widget_valid(self.stratify_combo) else 0
            })
        
  
        elif method == "K-Fold":
            # V1.4.0: Safe widget access with validation
            params.update({
                'n_splits': self.k_fold_spin.value() if hasattr(self, 'k_fold_spin') and self._is_widget_valid(self.k_fold_spin) else 5,
                'random_state': self.kf_random_state_spin.value() if hasattr(self, 'kf_random_state_spin') and self._is_widget_valid(self.kf_random_state_spin) else 42,
                'shuffle': self.kf_shuffle_checkbox.isChecked() if hasattr(self, 'kf_shuffle_checkbox') and self._is_widget_valid(self.kf_shuffle_checkbox) else True,
                'target_column': 0
            })
        
  
        elif method == "LOGO":
            # V1.4.0: Safe widget access with validation
            params.update({
                'groups_column': self.logo_group_combo.currentText() if hasattr(self, 'logo_group_combo') and self._is_widget_valid(self.logo_group_combo) else '',
                'target_column': 0
            })
        
  
        elif method == "Time Series":
            # V1.4.0: Safe widget access with validation
            params.update({
                'n_splits': self.n_splits_spin.value() if hasattr(self, 'n_splits_spin') and self._is_widget_valid(self.n_splits_spin) else 5,
                'test_size': self.random_test_size_spin.value() if hasattr(self, 'random_test_size_spin') and self._is_widget_valid(self.random_test_size_spin) else 0.2,
                'gap': 0,
                'target_column': 0
            })
        
        return params

    def update_data_info(self, data=None):
        """Update data information display"""
        if data is None:
            self.info_label.setText("No data loaded")
            return
            
        info_text = f"Number of samples: {data.shape[0]}\n"
        info_text += f"Number of features: {data.shape[1]}\n"
        info_text += f"Data type: {data.dtype}\n"
        info_text += f"Memory usage: {data.nbytes / 1024 / 1024:.2f} MB"
        
        self.info_label.setText(info_text)
        
    def save_plot(self):
        """Save current plot"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if file_path:
            self.plot_widget.save_figure(file_path)
    
    def add_custom_algorithm(self, code: str):
        """
        Add custom data partitioning algorithm
        V1.3.2: Improved class recognition logic to avoid instantiating abstract base classes
        
        Args:
            code: Python code defining the custom algorithm class
        """
        try:
            import types
            import inspect
            from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
            import pandas as pd
            import numpy as np
            
            mod = types.ModuleType('custom_splitter')
            
            # V1.3.6: Provide comprehensive imports for scientific computing
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'DataPartitioningAlgorithm': DataPartitioningAlgorithm,
                'Dict': __import__('typing').Dict,
                'Tuple': __import__('typing').Tuple,
                'Any': __import__('typing').Any,
                'List': __import__('typing').List,
            })
            
            # V1.3.6: Add sklearn support for train_test_split (commonly used)
            try:
                from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
                mod.__dict__.update({
                    'train_test_split': train_test_split,
                    'StratifiedShuffleSplit': StratifiedShuffleSplit,
                    'KFold': KFold,
                })
            except ImportError:
                pass
            
            # Record class list before execution
            classes_before = set(item for item in mod.__dict__.values() if isinstance(item, type))
            
            exec(code, mod.__dict__)
            
            # Record class list after execution
            classes_after = set(item for item in mod.__dict__.values() if isinstance(item, type))
            
            # Find newly added classes (user-defined classes)
            new_classes = classes_after - classes_before
            
            print(f"🔍 Found {len(new_classes)} new classes from user code")
            
            # Find algorithm class and instantiate
            algorithm_found = False
            for item in new_classes:
                print(f"🔍 Checking class: {item.__name__}")
                print(f"  📋 Base classes: {[base.__name__ for base in item.__bases__]}")
                
                # V1.3.2: Must be a subclass of DataPartitioningAlgorithm
                try:
                    is_subclass = issubclass(item, DataPartitioningAlgorithm)
                except TypeError:
                    print(f"  ⏭️  Skipping {item.__name__}: Not a class")
                    continue
                
                if not is_subclass:
                    print(f"  ⚠️  ERROR: {item.__name__} does NOT inherit from DataPartitioningAlgorithm!")
                    print(f"  💡 Your class definition should be:")
                    print(f"     class {item.__name__}(DataPartitioningAlgorithm):")
                    print(f"  📝 Current definition appears to be:")
                    print(f"     class {item.__name__}({', '.join([base.__name__ for base in item.__bases__]) if item.__bases__ else ''}):")
                    continue
                
                # V1.3.2: Cannot be abstract class
                if inspect.isabstract(item):
                    print(f"  ⏭️  Skipping {item.__name__}: Abstract class (missing method implementations)")
                    # Show missing methods
                    abstract_methods = [method for method in dir(item) 
                                      if getattr(getattr(item, method, None), '__isabstractmethod__', False)]
                    if abstract_methods:
                        print(f"     Missing methods: {', '.join(abstract_methods)}")
                    continue
                
                try:
                    algorithm = item()
                    algorithm_name = algorithm.get_name()
                    
                    # V1.4.1: Store instance, not class (for consistency with other views)
                    self.plugins[algorithm_name] = algorithm  # Store instance, not class
                    
                    # V1.3.5: Use concise custom identifier
                    display_name = f"[Custom Split] {algorithm_name}"
                    self.method_combo.addItem(display_name, algorithm_name)  # Second parameter is userData
                    
                    print(f"✅ Custom data partitioning algorithm '{algorithm_name}' added to UI")
                    algorithm_found = True
                    break
                except Exception as e:
                    print(f"  ❌ Failed to instantiate {item.__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not algorithm_found:
                error_msg = "❌ No valid algorithm class found in code.\n\n"
                error_msg += "Please ensure your class:\n"
                error_msg += "  1. ✅ Inherits from DataPartitioningAlgorithm:\n"
                error_msg += "     class YourClassName(DataPartitioningAlgorithm):\n\n"
                error_msg += "  2. ✅ Implements all abstract methods:\n"
                error_msg += "     - get_name(self) -> str\n"
                error_msg += "     - get_params_info(self) -> Dict\n"
                error_msg += "     - partition(self, data, params) -> Tuple\n\n"
                error_msg += "  3. ✅ Has a no-argument __init__ or default parameters\n\n"
                if new_classes:
                    error_msg += f"Found {len(new_classes)} class(es) but none were valid:\n"
                    for cls in new_classes:
                        error_msg += f"  - {cls.__name__} (bases: {[b.__name__ for b in cls.__bases__]})\n"
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"❌ Error loading custom splitter: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error loading custom splitter: {str(e)}")
