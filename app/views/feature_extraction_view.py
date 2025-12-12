# -*- coding: utf-8 -*-
"""
Feature Extraction View (Unsupervised)
Addresses Editor Comment 4: Separate PCA as unsupervised feature extraction
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit,
    QCheckBox, QProgressBar, QMessageBox, QTabWidget, QFormLayout,
    QSlider, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager

class FeatureExtractionView(QWidget):
    """Unsupervised feature extraction view - no labels required"""
    
    # Signals
    extraction_requested = pyqtSignal(str, dict)  # method, params
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.extraction_results = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the feature extraction UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("Feature Extraction (Unsupervised)")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Info label
        info_label = QLabel("Extract features without requiring labels or data partitioning")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(info_label)
        
        # Create tabs for different extraction methods
        tabs = QTabWidget()
        
        # PCA Tab
        pca_tab = self.create_pca_tab()
        tabs.addTab(pca_tab, "Principal Component Analysis")
        
        main_layout.addWidget(tabs)
        
        # Results area
        results_group = QGroupBox("Extraction Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Visualization area（响应式）
        from app.utils.responsive_matplotlib import create_responsive_figure
        self.figure = create_responsive_figure(base_width=10, base_height=6)
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        self.setLayout(main_layout)
    
    def create_pca_tab(self):
        """Create PCA configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Configuration group
        config_group = QGroupBox("PCA Configuration")
        config_layout = QFormLayout()
        
        # Number of components
        components_layout = QHBoxLayout()
        
        self.auto_components = QCheckBox("Auto (based on CEV threshold)")
        self.auto_components.setChecked(True)
        self.auto_components.toggled.connect(self.on_auto_components_toggled)
        components_layout.addWidget(self.auto_components)
        
        self.manual_components = QSpinBox()
        self.manual_components.setRange(2, 100)
        self.manual_components.setValue(10)
        self.manual_components.setEnabled(False)
        components_layout.addWidget(self.manual_components)
        
        config_layout.addRow("Components:", components_layout)
        
        # CEV threshold
        self.cev_threshold = QDoubleSpinBox()
        self.cev_threshold.setRange(0.5, 0.99)
        self.cev_threshold.setValue(0.95)
        self.cev_threshold.setSingleStep(0.05)
        self.cev_threshold.setDecimals(2)
        self.cev_threshold.setSuffix(" (95%)")
        config_layout.addRow("CEV Threshold:", self.cev_threshold)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()
        
        self.standardize = QCheckBox("Standardize data")
        self.standardize.setChecked(True)
        self.standardize.setToolTip("Recommended for spectral data with different scales")
        advanced_layout.addRow("Preprocessing:", self.standardize)
        
        self.show_loadings = QCheckBox("Show feature loadings")
        self.show_loadings.setChecked(True)
        advanced_layout.addRow("Output:", self.show_loadings)
        
        advanced_group.setLayout(advanced_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        layout.addWidget(advanced_group)
        
        # Extract button
        extract_layout = QHBoxLayout()
        self.extract_button = QPushButton("Extract PCA Features")
        self.extract_button.clicked.connect(self.extract_pca_features)
        self.extract_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        extract_layout.addStretch()
        extract_layout.addWidget(self.extract_button)
        layout.addLayout(extract_layout)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def on_auto_components_toggled(self, checked):
        """Handle auto components toggle"""
        self.manual_components.setEnabled(not checked)
        self.cev_threshold.setEnabled(checked)
    
    def extract_pca_features(self):
        """Extract PCA features"""
        # Get parameters
        params = {
            'n_components': None if self.auto_components.isChecked() else self.manual_components.value(),
            'cev_threshold': self.cev_threshold.value(),
            'standardize': self.standardize.isChecked(),
            'show_loadings': self.show_loadings.isChecked()
        }
        
        # Emit signal to request extraction
        self.extraction_requested.emit('pca', params)
    
    def display_pca_results(self, results):
        """Display PCA extraction results"""
        self.extraction_results['pca'] = results
        
        # Update results text
        text = f"PCA Feature Extraction Results\n"
        text += f"=" * 40 + "\n\n"
        text += f"Components extracted: {results['n_components']}\n"
        text += f"Total variance explained: {results['cumulative_explained_variance'][-1]:.1%}\n"
        text += f"Original features: {results['transformed_data'].shape[1] if 'transformed_data' in results else 'N/A'}\n"
        text += f"Extracted features: {results['n_components']}\n\n"
        
        # Component-wise variance
        text += "Component Variance:\n"
        for i, var in enumerate(results['explained_variance_ratio']):
            text += f"  PC{i+1}: {var:.1%}\n"
        
        text += "\n"
        
        # Feature importance (top 10)
        if 'analysis' in results and 'feature_importance' in results['analysis']:
            text += "Top 10 Most Important Features:\n"
            for i, feat_info in enumerate(results['analysis']['feature_importance'][:10]):
                text += f"  {i+1}. {feat_info['feature']}: {feat_info['importance']:.4f}\n"
        
        # Recommendations
        if 'analysis' in results and 'recommendations' in results['analysis']:
            if results['analysis']['recommendations']:
                text += "\nRecommendations:\n"
                for rec in results['analysis']['recommendations']:
                    text += f"  • {rec}\n"
        
        self.results_text.setPlainText(text)
        
        # Create visualization
        self.create_pca_visualization(results)
    
    def create_pca_visualization(self, results):
        """Create PCA visualization plots"""
        self.figure.clear()
        
        # Create subplots
        if len(results['explained_variance_ratio']) > 1:
            gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = self.figure.add_subplot(gs[0, 0])  # Scree plot
            ax2 = self.figure.add_subplot(gs[0, 1])  # Cumulative variance
            ax3 = self.figure.add_subplot(gs[1, :])  # Loadings plot
        else:
            gs = self.figure.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
            ax1 = None
            ax2 = self.figure.add_subplot(gs[0, 0])
            ax3 = self.figure.add_subplot(gs[0, 1])
        
        # Scree plot
        if ax1 is not None:
            components = range(1, len(results['explained_variance_ratio']) + 1)
            ax1.plot(components, results['explained_variance_ratio'], 'bo-', linewidth=2, markersize=6)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Scree Plot')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(components)
        
        # Cumulative explained variance
        components = range(1, len(results['cumulative_explained_variance']) + 1)
        ax2.plot(components, results['cumulative_explained_variance'], 'ro-', linewidth=2, markersize=6)
        ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
        ax2.legend(fontsize=font_sizes.get('legend.fontsize', 11))
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(components)
        ax2.set_ylim(0, 1.05)
        
        # Feature loadings plot (top features for first two components)
        if 'feature_loadings' in results and ax3 is not None:
            loadings = results['feature_loadings']
            
            if loadings.shape[1] >= 2:
                # Get top features for PC1 and PC2
                pc1_loadings = loadings.iloc[:, 0].abs().sort_values(ascending=False)[:15]
                pc2_loadings = loadings.iloc[:, 1].abs().sort_values(ascending=False)[:15]
                
                # Combine and get unique features
                top_features = list(set(pc1_loadings.index.tolist() + pc2_loadings.index.tolist()))[:20]
                
                # Plot loadings
                x_pos = np.arange(len(top_features))
                pc1_values = [loadings.loc[feat, 'PC1'] if feat in loadings.index else 0 for feat in top_features]
                pc2_values = [loadings.loc[feat, 'PC2'] if feat in loadings.index else 0 for feat in top_features]
                
                width = 0.35
                ax3.bar(x_pos - width/2, pc1_values, width, label='PC1', alpha=0.8)
                ax3.bar(x_pos + width/2, pc2_values, width, label='PC2', alpha=0.8)
                
                ax3.set_xlabel('Features')
                ax3.set_ylabel('Loading Value')
                ax3.set_title('Feature Loadings (Top Features)')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels([f"F{i+1}" for i in range(len(top_features))], rotation=45)
                font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
                ax3.legend(fontsize=font_sizes.get('legend.fontsize', 11))
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Need at least 2 components\nfor loadings plot', 
                        transform=ax3.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax3.set_title('Feature Loadings')
        
        self.canvas.draw()
    
    def display_message(self, message, title="Information"):
        """Display message to user"""
        QMessageBox.information(self, title, message)
    
    def display_error(self, error_message, title="Error"):
        """Display error message to user"""
        QMessageBox.critical(self, title, error_message)
    
    def clear_results(self):
        """Clear all results"""
        self.extraction_results.clear()
        self.results_text.clear()
        self.figure.clear()
        self.canvas.draw()
    
    def get_extraction_results(self):
        """Get current extraction results"""
        return self.extraction_results.copy()
