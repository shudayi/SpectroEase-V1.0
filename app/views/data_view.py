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
        self.layout.setSpacing(10)  # 减少间距
        self.layout.setContentsMargins(1, 1, 1, 1)  # 减少边距

  
        method_layout = QHBoxLayout()
        method_layout.setSpacing(1)  # 减少间距
        
  
        self.method_combo = QComboBox()
        self.method_combo.setMaximumWidth(140)  # 限制宽度，减少为120
  
        built_in_methods = ["Train-Test Split", "K-Fold", "LOGO", "Random", "Stratified"]
        plugin_methods = list(self.plugins.keys())
        self.method_combo.addItems(built_in_methods + plugin_methods)
        method_layout.addWidget(self.method_combo)
        
  
        self.force_classification_checkbox = QCheckBox("Force Classification Task")
        self.force_classification_checkbox.setToolTip("勾选后，无论data类型如何，都会将任务视为classification任务")
        method_layout.addWidget(self.force_classification_checkbox)
        
  
        self.load_data_btn = QPushButton("Load Data")
        self.load_data_btn.setFixedHeight(25)
        method_layout.addWidget(self.load_data_btn)
        
  
        self.layout.addLayout(method_layout)

  
        self.params_form = QFormLayout()
        self.params_form.setSpacing(10)  # 减少间距
        self.params_form.setContentsMargins(1, 1, 1, 1)  # 减少边距
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
  
            param_info = self.plugins[method].get_parameter_info()
  
            horizontal_layout = QHBoxLayout()
            horizontal_layout.setSpacing(10)
            
            for param_name, info in param_info.items():
  
                param_container = QHBoxLayout()
                param_container.setSpacing(10)
                
  
                label = QLabel(f"{param_name}:")
                label.setMaximumWidth(60)  # 调整为需要的宽度
                param_container.addWidget(label)
                
  
                widget = self.create_widget(info)
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox) or isinstance(widget, QComboBox):
                    widget.setMaximumWidth(60)  # 减少宽度为40
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
                test_size_label.setMaximumWidth(40)  # 调整Testlabels的宽度，可改为更小或更大的值
                test_size_container.addWidget(test_size_label)
                
                self.test_size_spin = QDoubleSpinBox()
                self.test_size_spin.setRange(0.01, 0.99)
                self.test_size_spin.setValue(0.2)
                self.test_size_spin.setSingleStep(0.05)
                self.test_size_spin.setMaximumWidth(60) # 增加宽度以显示小数
                test_size_container.addWidget(self.test_size_spin)
                params_container.addLayout(test_size_container)
                
                params_container.addStretch() # 添加弹簧实现均匀分布

  
                random_container = QHBoxLayout()
                random_container.setSpacing(5)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(40)  # 调整Seedlabels的宽度
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
                self.min_samples_spin.setMaximumWidth(60) # 增加宽度以显示数字
                min_samples_container.addWidget(self.min_samples_spin)
                params_container.addLayout(min_samples_container)
                
                params_container.addStretch() # 添加弹簧实现均匀分布

  
                shuffle_container = QHBoxLayout()
                shuffle_container.setSpacing(5)
                shuffle_label = QLabel("Shuffle:")
                shuffle_label.setMaximumWidth(55)  # 调整Shufflelabels的宽度
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
                stratify_label.setMaximumWidth(60)  # 增加Stratifylabels的宽度显示完整文本
                stratify_container.addWidget(stratify_label)
                
                self.stratify_combo = QComboBox()
                self.stratify_combo.addItem("None")
                self.stratify_combo.addItem("label")  # 如果有labels列
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
                k_label.setMaximumWidth(15)  # 减少宽度为15
                k_container.addWidget(k_label)
                
                self.k_fold_spin = QSpinBox()
                self.k_fold_spin.setRange(2, 10)
                self.k_fold_spin.setValue(5)
                self.k_fold_spin.setMaximumWidth(35)  # 减少宽度为35
                k_container.addWidget(self.k_fold_spin)
                params_container.addLayout(k_container)
                
  
                random_container = QHBoxLayout()
                random_container.setSpacing(1)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(25)  # 减少宽度为25
                random_container.addWidget(random_label)
                
                self.kf_random_state_spin = QSpinBox()
                self.kf_random_state_spin.setRange(0, 10000)
                self.kf_random_state_spin.setValue(42)
                self.kf_random_state_spin.setMaximumWidth(40)  # 减少宽度为40
                random_container.addWidget(self.kf_random_state_spin)
                params_container.addLayout(random_container)
                
  
                shuffle_container = QHBoxLayout()
                shuffle_container.setSpacing(1)
                shuffle_label = QLabel("Shuffle:")
                shuffle_label.setMaximumWidth(35)  # 减少宽度为35
                shuffle_container.addWidget(shuffle_label)
                
                self.kf_shuffle_checkbox = QCheckBox()
                self.kf_shuffle_checkbox.setChecked(True)
                self.kf_shuffle_checkbox.setMaximumWidth(15)  # 减少宽度为15
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
                group_label.setMaximumWidth(30)  # 减少宽度为30
                group_container.addWidget(group_label)
                
                self.logo_group_combo = QComboBox()
                self.logo_group_combo.addItem("label")
                self.logo_group_combo.addItem("batch")
                self.logo_group_combo.setMaximumWidth(45)  # 减少宽度为45
                group_container.addWidget(self.logo_group_combo)
                params_container.addLayout(group_container)
                
  
                self.params_form.addRow(params_container)
            
            elif method == "Random" or method == "Stratified":
  
                params_container = QHBoxLayout()
                params_container.setSpacing(2)
                
  
                n_container = QHBoxLayout()
                n_container.setSpacing(1)
                n_label = QLabel("N:")
                n_label.setMaximumWidth(15)  # 减少宽度为15
                n_container.addWidget(n_label)
                
                self.n_splits_spin = QSpinBox()
                self.n_splits_spin.setRange(2, 10)
                self.n_splits_spin.setValue(5)
                self.n_splits_spin.setMaximumWidth(35)  # 减少宽度为35
                n_container.addWidget(self.n_splits_spin)
                params_container.addLayout(n_container)
                
  
                test_size_container = QHBoxLayout()
                test_size_container.setSpacing(1)
                test_size_label = QLabel("Test:")
                test_size_label.setMaximumWidth(25)  # 减少宽度为25
                test_size_container.addWidget(test_size_label)
                
                self.random_test_size_spin = QDoubleSpinBox()
                self.random_test_size_spin.setRange(0.01, 0.99)
                self.random_test_size_spin.setValue(0.2)
                self.random_test_size_spin.setSingleStep(0.05)
                self.random_test_size_spin.setMaximumWidth(40)  # 减少宽度为40
                test_size_container.addWidget(self.random_test_size_spin)
                params_container.addLayout(test_size_container)
                
  
                random_container = QHBoxLayout()
                random_container.setSpacing(1)
                random_label = QLabel("Seed:")
                random_label.setMaximumWidth(25)  # 减少宽度为25
                random_container.addWidget(random_label)
                
                self.random_state_seed_spin = QSpinBox()
                self.random_state_seed_spin.setRange(0, 10000)
                self.random_state_seed_spin.setValue(42)
                self.random_state_seed_spin.setMaximumWidth(40)  # 减少宽度为40
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
            widget = QLineEdit()  # 默认使用文本输入框
        return widget

    def on_method_change(self, method):
        self.init_param_widgets(method)

    def get_selected_method(self):
        return self.method_combo.currentText()

    def get_parameters(self):
        """
        获取所有parameters。
        
        Returns:
            dict: A dictionary of parameters.
        """
  
        method = self.method_combo.currentText()
        
  
        params = {}
        
  
        params['force_classification'] = self.force_classification_checkbox.isChecked()
        
  
        if method == "Train-Test Split":
            params.update({
                'test_size': self.test_size_spin.value(),
                'random_state': self.random_state_spin.value(),
                'shuffle': self.shuffle_checkbox.isChecked(),
                'min_samples': self.min_samples_spin.value(),
                'target_column': self.stratify_combo.currentIndex()
            })
        
  
        elif method == "K-Fold":
            params.update({
                'n_splits': self.k_fold_spin.value(),
                'random_state': self.kf_random_state_spin.value(),
                'shuffle': self.kf_shuffle_checkbox.isChecked(),
                'target_column': 0
            })
        
  
        elif method == "LOGO":
            params.update({
                'groups_column': self.logo_group_combo.currentText(),
                'target_column': 0
            })
        
  
        elif method == "Time Series":
            params.update({
                'n_splits': self.n_splits_spin.value(),
                'test_size': self.random_test_size_spin.value(),
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
