import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QTabWidget, QSizePolicy
from PyQt5.QtCore import Qt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class VisualizationWindow(QMainWindow):
    def __init__(self, parent=None, title="Visualization Results"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        
        # Create main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create layout and add tab widget
        layout = QVBoxLayout(self.main_widget)
        layout.addWidget(self.tabs)
        
        # Initialize tab counter
        self.tab_count = 0
        
        # Set window flags
        self.setWindowFlags(Qt.Window)

    def clear_plots(self):
        # Remove all tabs and reset counter
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)
        self.tab_count = 0

    def add_figure_tab(self, fig, title):
        # Create a canvas for the figure
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create a layout to contain canvas and toolbar
        tab_layout = QVBoxLayout()
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar(canvas, self)
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        
        # Create a widget to host the layout
        tab = QWidget()
        tab.setLayout(tab_layout)
        
        # Add tab with the given title
        tab_index = self.tabs.addTab(tab, title)
        self.tabs.setCurrentIndex(tab_index)
        
        # Increment tab counter
        self.tab_count += 1
        
        return tab_index

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        # Ensure numpy arrays
        import numpy as np
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        
        # Set title
        ax.set_title(title)
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_feature_importance(self, feature_importance, feature_names=None, top_n=None, title='Feature Importance'):
        # Ensure numpy array
        if not isinstance(feature_importance, np.ndarray):
            feature_importance = np.array(feature_importance)
        
        # Sort features by importance
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        
        # Create sorted indices
        sorted_idx = np.argsort(feature_importance)
        
        # Apply top_n limit if specified
        if top_n is not None and top_n < len(sorted_idx):
            sorted_idx = sorted_idx[-top_n:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot feature importance
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, feature_importance[sorted_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_data(self, X, y=None, title='Data Visualization', method='pca', n_components=2):
        if X is None or len(X) == 0:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Reduce dimensionality if needed
        if X.shape[1] > n_components:
            if method.lower() == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
            elif method.lower() == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components)
            else:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
            
            X_reduced = reducer.fit_transform(X)
        else:
            X_reduced = X
        
        # Plot the data
        if n_components == 2:
            if y is not None:
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.6)
                # Add colorbar
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            if y is not None:
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.6)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        
        ax.set_title(title)
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_learning_curve(self, train_sizes, train_scores, test_scores, title='Learning Curve'):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate mean and std of scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_roc_curve(self, fpr, tpr, auc=None, title='ROC Curve'):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})' if auc is not None else 'ROC curve')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_regression_results(self, y_true, y_pred, title='Regression Results'):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot regression results (true vs predicted)
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(title)
        
        # Calculate and display metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics_text = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_residuals(self, y_true, y_pred, title='Residuals Plot'):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Plot residuals
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(title)
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_spectra(self, wavelengths, spectra, labels=None, title='Spectral Data'):
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check if spectra is a single spectrum or multiple spectra
        if spectra.ndim == 1:
            # Single spectrum
            ax.plot(wavelengths, spectra)
        else:
            # Multiple spectra
            for i, spectrum in enumerate(spectra):
                if labels is not None and i < len(labels):
                    ax.plot(wavelengths, spectrum, label=str(labels[i]))
                else:
                    ax.plot(wavelengths, spectrum, label=f'Sample {i+1}')
        
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        
        # Add legend if multiple spectra
        if spectra.ndim > 1 and spectra.shape[0] > 1:
            ax.legend()
        
        # Add the figure to the window
        self.add_figure_tab(fig, title)

    def plot_spectra_by_class(self, wavelengths, spectra, classes, class_names=None, title='Spectra by Class'):
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique classes
        unique_classes = np.unique(classes)
        
        # Plot spectra by class
        for cls in unique_classes:
            # Get indices for this class
            indices = np.where(classes == cls)[0]
            
            # Calculate mean spectrum for this class
            mean_spectrum = np.mean(spectra[indices], axis=0)
            
            # Get class name
            if class_names is not None and cls < len(class_names):
                class_label = class_names[cls]
            else:
                class_label = f'Class {cls}'
            
            # Plot mean spectrum for this class
            ax.plot(wavelengths, mean_spectrum, label=class_label)
        
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        ax.legend()
        
        # Add the figure to the window
        self.add_figure_tab(fig, title) 