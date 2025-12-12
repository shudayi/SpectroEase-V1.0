# -*- coding: utf-8 -*-
"""
Enhanced Preprocessing Visualization with Y-axis locking and complete legends
Addresses Editor Comment 3 about preprocessing visualization issues
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt
from app.views.responsive_dialog import ResponsiveDialog

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager

# Import visualization design tokens for responsive visualization
from app.config.visualization_design_tokens import VDT

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedPreprocessingVisualization(ResponsiveDialog):
    """Enhanced preprocessing visualization dialog with locked Y-axis and complete legends."""
    
    def __init__(self, data_model, wavelengths, labels, methods_applied, parent=None):
        super().__init__(parent, base_width=1100, base_height=850)
        self.data_model = data_model
        self.wavelengths = wavelengths
        self.labels = labels
        self.methods_applied = methods_applied
        
        self.setWindowTitle("Enhanced Preprocessing Visualization")
        
        self.init_ui()
        self.y_axis_locked = True  # Default to locked Y-axis
        self.original_y_limits = None
        
        self.plot_preprocessing_comparison()
        
    def init_ui(self):
        """Initialize the visualization UI"""
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Y-axis lock checkbox
        self.y_lock_checkbox = QCheckBox("Lock Y-axis (prevents intensity scaling artifacts)")
        self.y_lock_checkbox.setChecked(True)
        self.y_lock_checkbox.toggled.connect(self.toggle_y_axis_lock)
        control_layout.addWidget(self.y_lock_checkbox)
        
        # Processing queue display
        self.queue_label = QLabel("Processing Queue: None")
        self.queue_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        control_layout.addWidget(self.queue_label)
        
        # Scaling indicator
        self.scaling_indicator = QLabel("")
        self.scaling_indicator.setStyleSheet("QLabel { color: #ff6600; font-weight: bold; }")
        control_layout.addWidget(self.scaling_indicator)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Matplotlib figure（响应式）
        from app.utils.responsive_matplotlib import create_responsive_figure
        self.figure = create_responsive_figure(base_width=12, base_height=8)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def toggle_y_axis_lock(self, checked):
        """Toggle Y-axis locking"""
        self.y_axis_locked = checked
        if checked:
            self.y_lock_checkbox.setText("Lock Y-axis (prevents intensity scaling artifacts) ✓")
        else:
            self.y_lock_checkbox.setText("Lock Y-axis (prevents intensity scaling artifacts)")
    
    def update_processing_queue(self, methods_list):
        """Update the processing queue display"""
        if methods_list:
            queue_text = " → ".join(methods_list)
            self.queue_label.setText(f"Processing Queue: {queue_text}")
        else:
            self.queue_label.setText("Processing Queue: None")
    
    def update_scaling_indicator(self, has_scaling_normalization):
        """Update scaling/normalization indicator"""
        if has_scaling_normalization:
            self.scaling_indicator.setText("SCALING/NORMALIZATION ACTIVE")
        else:
            self.scaling_indicator.setText("")
    
    def plot_preprocessing_comparison(self):
        """Plot preprocessing comparison with enhanced features."""
        self.figure.clear()

        if self.data_model.X is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Error: Original data (X) not found in data model.', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='red')
            self.canvas.draw()
            return
        
        original_spectra = self.data_model.X.values
        
        if self.data_model.X_processed is not None:
            processed_spectra = self.data_model.X_processed.values
        else:
            processed_spectra = original_spectra  # Use original as placeholder
            logger.info("X_processed is not available. Using original data (X) as a placeholder for visualization.")
            print("Log: X_processed is not available. Using original data (X) as a placeholder for visualization.")

        # Use a unified data alignment tool with import protection
        try:
            from app.utils.data_alignment import DataDimensionAligner
            
            aligned_wavelengths, aligned_original, aligned_processed = DataDimensionAligner.align_spectral_data(
                self.wavelengths, original_spectra, processed_spectra
            )
            wavelengths = aligned_wavelengths
            original_spectra = aligned_original
            processed_spectra = aligned_processed
            
        except ImportError:
            ax = self.figure.add_subplot(111)
            error_text = "Failed to import 'DataDimensionAligner'.\nPlease ensure 'app/utils/data_alignment.py' exists and is correct."
            ax.text(0.5, 0.5, error_text, horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='red', wrap=True)
            self.canvas.draw()
            return
        except Exception as e:
            ax = self.figure.add_subplot(111)
            # This relies on a static method in DataDimensionAligner, which might not be available if import failed.
            # A simple text error is safer.
            error_text = f"An error occurred during data alignment:\n{e}"
            ax.text(0.5, 0.5, error_text, horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='red', wrap=True)
            self.canvas.draw()
            return
        
        # Update processing queue display
        if self.methods_applied:
            self.update_processing_queue(self.methods_applied)
            scaling_methods = ['normalization', 'standardization', 'min-max', 'vector', 'area', 'maximum', 'z-score', 'robust']
            has_scaling = any(any(sm in method.lower() for sm in scaling_methods) for method in self.methods_applied)
            self.update_scaling_indicator(has_scaling)
        
        ax = self.figure.add_subplot(111)
        
        # Use get_cmap for type safety
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
        
        # Plot both original and processed spectra for comparison
        self._plot_spectra_set(ax, wavelengths, original_spectra, self.labels, colors, "Original", 'solid')
        self._plot_spectra_set(ax, wavelengths, processed_spectra, self.labels, colors, "Processed", 'dashed')
        
        # Add mean comparison with responsive line width
        original_mean = np.mean(original_spectra, axis=0)
        processed_mean = np.mean(processed_spectra, axis=0)
        
        ax.plot(wavelengths, original_mean, color='blue', 
               linewidth=VDT.LINE_DATA_THICK, 
               label='Mean (Original)', alpha=0.8, zorder=10)
        ax.plot(wavelengths, processed_mean, color='red', 
               linewidth=VDT.LINE_DATA_THICK, 
               label='Mean (Processed)', alpha=0.8, zorder=10)
        
        # Add standard deviation bands (不添加到图例，用半透明表示)
        original_std = np.std(original_spectra, axis=0)
        processed_std = np.std(processed_spectra, axis=0)
        
        ax.fill_between(wavelengths, 
                      original_mean - original_std, 
                      original_mean + original_std, 
                      color='blue', alpha=0.15, zorder=1)
        ax.fill_between(wavelengths, 
                      processed_mean - processed_std, 
                      processed_mean + processed_std, 
                      color='red', alpha=0.15, zorder=1)
        
        # Use responsive font sizes
        ax.set_xlabel('Wavelength (nm)', fontsize=VDT.FONT_LABEL_MEDIUM, fontweight='medium')
        ax.set_ylabel('Absorbance', fontsize=VDT.FONT_LABEL_MEDIUM, fontweight='medium')
        ax.set_title('Preprocessing Comparison', fontsize=VDT.FONT_TITLE_MEDIUM, fontweight='bold')
        
        if self.y_axis_locked:
            if self.original_y_limits is None and original_spectra is not None and original_spectra.size > 0:
                y_min, y_max = np.min(original_spectra), np.max(original_spectra)
                y_padding = (y_max - y_min) * 0.05
                self.original_y_limits = (y_min - y_padding, y_max + y_padding)
            
            if self.original_y_limits:
                ax.set_ylim(self.original_y_limits)
        else:
            self.original_y_limits = None
        
        legend_elements = ax.get_legend_handles_labels()
        if legend_elements[0]:
            # 使用响应式图例字体，并优化显示
            legend = ax.legend(loc='upper right', 
                             frameon=True, 
                             framealpha=0.9, 
                             fontsize=VDT.FONT_LEGEND,
                             ncol=2 if len(legend_elements[0]) > 6 else 1,  # 超过6项时分两列
                             borderpad=0.5,
                             labelspacing=0.4,
                             columnspacing=1.0)
            
            # 添加线型说明文本
            ax.text(0.02, 0.98, 'Solid: Original | Dashed: Processed\nShaded: ±1 SD', 
                   transform=ax.transAxes, 
                   fontsize=VDT.FONT_TEXT_MEDIUM, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.85, edgecolor='gray', linewidth=0.8))
        
        ax.grid(True, linestyle='-', linewidth=VDT.LINE_GRID_LIGHT, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(VDT.LINE_SPINE)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _plot_spectra_set(self, ax, wavelengths, spectra, labels, colors, prefix, linestyle):
        """Helper to plot a set of spectra with consistent styling."""
        if spectra is None or spectra.size == 0:
            return
        max_spectra_to_show = min(5, spectra.shape[0])
        
        # 用于追踪已添加图例的类别
        added_legend_labels = set()
        
        for i in range(max_spectra_to_show):
            color = colors[i % len(colors)]
            
            # 只为每个类别的第一条线添加图例
            if labels is not None and i < len(labels):
                class_label = str(labels[i])
                # 如果这个类别还没有图例项，添加它
                if class_label not in added_legend_labels:
                    label = f"{class_label} ({prefix})"
                    added_legend_labels.add(class_label)
                else:
                    label = None  # 同一类别的后续线不显示图例
            else:
                # 只为第一条无标签光谱添加图例
                if i == 0:
                    label = f"{prefix}"
                else:
                    label = None
            
            ax.plot(wavelengths, spectra[i], color=color, linewidth=1.5, 
                   linestyle=linestyle, label=label, alpha=0.7)
    
    def reset_y_axis_limits(self):
        """Reset stored Y-axis limits."""
        self.original_y_limits = None