# app/views/evaluation_view.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QGroupBox, QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem

class EvaluationView(QWidget):
    def __init__(self):
        super(EvaluationView, self).__init__()
        self.layout = QVBoxLayout()

  
        self.evaluate_button = QPushButton("Evaluate Model")
        self.layout.addWidget(self.evaluate_button)

  
        self.results_label = QLabel("Evaluation Results:")
        self.layout.addWidget(self.results_label)

        self.setLayout(self.layout)

    def update_evaluation_results(self, results):
        display_text = "Evaluation Results:\n"
        for key, value in results.items():
            if key == 'Confusion Matrix':
                display_text += f"{key}:\n{value}\n"
            elif key == 'Classification Report':
                display_text += f"{key}:\n{value}\n"
            else:
                display_text += f"{key}: {value}\n"
        self.results_label.setText(display_text)

    def display_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Evaluation metrics
        metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QVBoxLayout()
        
        # Quantitative metrics
        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("Quantitative Metrics:"))
        self.quant_metrics = QComboBox()
        self.quant_metrics.addItems([
            "Root Mean Square Error (RMSE)",
            "Mean Absolute Error (MAE)",
            "R-squared (R²)",
            "Correlation Coefficient (R)",
            "Relative Error (RE)"
        ])
        quant_layout.addWidget(self.quant_metrics)
        metrics_layout.addLayout(quant_layout)
        
        # Qualitative metrics
        qual_layout = QHBoxLayout()
        qual_layout.addWidget(QLabel("Qualitative Metrics:"))
        self.qual_metrics = QComboBox()
        self.qual_metrics.addItems([
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Area Under ROC Curve (AUC)"
        ])
        qual_layout.addWidget(self.qual_metrics)
        metrics_layout.addLayout(qual_layout)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        # Plot type
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "Scatter Plot",
            "Residual Plot",
            "ROC Curve",
            "Confusion Matrix",
            "Learning Curve"
        ])
        plot_type_layout.addWidget(self.plot_type)
        viz_layout.addLayout(plot_type_layout)
        
        # Plot widget
        self.plot_widget = PlotWidget()
        viz_layout.addWidget(self.plot_widget)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.evaluate_button = QPushButton("Evaluate")
        self.save_results_button = QPushButton("Save Results")
        self.export_plot_button = QPushButton("Export Plot")
        
        button_layout.addWidget(self.evaluate_button)
        button_layout.addWidget(self.save_results_button)
        button_layout.addWidget(self.export_plot_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def get_parameters(self):
        """Get evaluation parameters"""
        return {
            'quantitative_metric': self.quant_metrics.currentText(),
            'qualitative_metric': self.qual_metrics.currentText(),
            'plot_type': self.plot_type.currentText()
        }
        
    def update_results(self, results):
        """Update evaluation results"""
        self.results_table.setRowCount(len(results))
        for i, (metric, value) in enumerate(results.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            
    def update_plot(self, plot_data):
        """Update evaluation plot"""
        self.plot_widget.clear()
        
        plot_type = self.plot_type.currentText()
        if plot_type == "Scatter Plot":
            self.plot_widget.scatter(plot_data['x'], plot_data['y'])
            self.plot_widget.set_xlabel("True Values")
            self.plot_widget.set_ylabel("Predicted Values")
            self.plot_widget.axhline(y=0, color='r', linestyle='--')
            self.plot_widget.axvline(x=0, color='r', linestyle='--')
            
        elif plot_type == "Residual Plot":
            self.plot_widget.scatter(plot_data['x'], plot_data['residuals'])
            self.plot_widget.set_xlabel("Predicted Values")
            self.plot_widget.set_ylabel("Residuals")
            self.plot_widget.axhline(y=0, color='r', linestyle='--')
            
        elif plot_type == "ROC Curve":
            self.plot_widget.plot(plot_data['fpr'], plot_data['tpr'])
            self.plot_widget.plot([0, 1], [0, 1], 'r--')
            self.plot_widget.set_xlabel("False Positive Rate")
            self.plot_widget.set_ylabel("True Positive Rate")
            
        elif plot_type == "Confusion Matrix":
            self.plot_widget.imshow(plot_data['matrix'], cmap='Blues')
            self.plot_widget.set_xlabel("Predicted")
            self.plot_widget.set_ylabel("True")
            
        elif plot_type == "Learning Curve":
            self.plot_widget.plot(plot_data['train_sizes'], plot_data['train_scores'], label="Training Score")
            self.plot_widget.plot(plot_data['train_sizes'], plot_data['test_scores'], label="Cross-validation Score")
            self.plot_widget.set_xlabel("Training Examples")
            self.plot_widget.set_ylabel("Score")
            self.plot_widget.legend()
            
        self.plot_widget.tight_layout()
