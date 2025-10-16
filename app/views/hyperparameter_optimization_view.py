# app/views/hyperparameter_optimization_view.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel,
    QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit
)
import json

class HyperparameterOptimizationView(QWidget):
    def __init__(self):
        super(HyperparameterOptimizationView, self).__init__()
        self.layout = QVBoxLayout()

  
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "No Optimization",
            "Grid Search",
            "Random Search",
            "Genetic Algorithm"
        ])
        self.layout.addWidget(QLabel("Select Hyperparameter Optimization Method:"))
        self.layout.addWidget(self.method_combo)

  
        self.params_form = QFormLayout()
        self.param_widgets = {}

        self.init_param_widgets()
        self.layout.addLayout(self.params_form)

  
        self.apply_button = QPushButton("Apply Optimization")
        self.layout.addWidget(self.apply_button)

  
        self.best_params_text = QTextEdit()
        self.best_params_text.setReadOnly(True)
        self.layout.addWidget(QLabel("Best Parameters:"))
        self.layout.addWidget(self.best_params_text)

        self.setLayout(self.layout)

  
        self.method_combo.currentIndexChanged.connect(self.on_method_change)

    def init_param_widgets(self):
        # Grid Search parameters
        self.param_grid_text = QTextEdit()
        self.param_grid_text.setPlaceholderText('{"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}')

        # Random Search parameters
        self.param_distributions_text = QTextEdit()
        self.param_distributions_text.setPlaceholderText('{"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}')
        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(1, 1000)
        self.n_iter_spin.setValue(10)

        # Genetic Algorithm parameters
        self.param_ranges_text = QTextEdit()
        self.param_ranges_text.setPlaceholderText('{"C": [0.1, 10], "kernel": ["linear", "rbf"]}')
        self.n_population_spin = QSpinBox()
        self.n_population_spin.setRange(10, 1000)
        self.n_population_spin.setValue(20)
        self.n_generations_spin = QSpinBox()
        self.n_generations_spin.setRange(1, 100)
        self.n_generations_spin.setValue(10)

  
        self.param_widgets['param_grid'] = self.param_grid_text
        self.param_widgets['param_distributions'] = self.param_distributions_text
        self.param_widgets['n_iter'] = self.n_iter_spin
        self.param_widgets['param_ranges'] = self.param_ranges_text
        self.param_widgets['n_population'] = self.n_population_spin
        self.param_widgets['n_generations'] = self.n_generations_spin

    def on_method_change(self):
  
        for i in reversed(range(self.params_form.count())):
            self.params_form.removeRow(i)

        method = self.method_combo.currentText()
        if method == "Grid Search":
            self.params_form.addRow("Parameter Grid (JSON):", self.param_grid_text)
        elif method == "Random Search":
            self.params_form.addRow("Parameter Distributions (JSON):", self.param_distributions_text)
            self.params_form.addRow("Number of Iterations:", self.n_iter_spin)
        elif method == "Genetic Algorithm":
            self.params_form.addRow("Parameter Ranges (JSON):", self.param_ranges_text)
            self.params_form.addRow("Population Size:", self.n_population_spin)
            self.params_form.addRow("Number of Generations:", self.n_generations_spin)
        else:
  
            pass

    def get_selected_method(self):
        method = self.method_combo.currentText()
        method_mapping = {
            "No Optimization": "no_optimization",
            "Grid Search": "grid_search",
            "Random Search": "random_search",
            "Genetic Algorithm": "genetic_algorithm"
        }
        return method_mapping.get(method, None)

    def get_parameters(self):
        params = {}
        method = self.get_selected_method()
        if method == 'grid_search':
            try:
                param_grid = json.loads(self.param_grid_text.toPlainText())
                params['param_grid'] = param_grid
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for parameter grid.")
        elif method == 'random_search':
            try:
                param_distributions = json.loads(self.param_distributions_text.toPlainText())
                params['param_distributions'] = param_distributions
                params['n_iter'] = self.n_iter_spin.value()
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for parameter distributions.")
        elif method == 'genetic_algorithm':
            try:
                param_ranges = json.loads(self.param_ranges_text.toPlainText())
                params['param_ranges'] = param_ranges
                params['n_population'] = self.n_population_spin.value()
                params['n_generations'] = self.n_generations_spin.value()
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for parameter ranges.")
        return params

    def display_message(self, message):
        QMessageBox.information(self, "Information", message)

    def display_error(self, message):
        QMessageBox.critical(self, "Error", message)
