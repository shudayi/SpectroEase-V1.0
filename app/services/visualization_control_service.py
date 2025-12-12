# -*- coding: utf-8 -*-
"""
Visualization Control Service
Addresses Editor Comment 3: Y-axis locking and comprehensive legends
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PyQt5.QtCore import QObject, pyqtSignal

class VisualizationControlService(QObject):
    """Service for managing visualization controls and settings"""
    
    # Signals for UI updates
    y_axis_lock_changed = pyqtSignal(bool)
    legend_update_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.y_axis_locked = True  # Default to locked as specified in response
        self.original_y_limits = None
        self.current_preprocessing_queue = []
        self.scaling_active = False
        
    def set_y_axis_lock(self, locked: bool):
        """Set Y-axis lock state"""
        if self.y_axis_locked != locked:
            self.y_axis_locked = locked
            self.y_axis_lock_changed.emit(locked)
            print(f"🔒 Y-axis lock {'enabled' if locked else 'disabled'}")
    
    def toggle_y_axis_lock(self):
        """Toggle Y-axis lock state"""
        self.set_y_axis_lock(not self.y_axis_locked)
    
    def calculate_y_limits(self, *data_arrays, padding_factor=0.05):
        """Calculate appropriate Y-axis limits for given data"""
        if not data_arrays or all(arr is None for arr in data_arrays):
            return None, None
            
        # Collect all valid data points
        all_values = []
        for arr in data_arrays:
            if arr is not None:
                if hasattr(arr, 'values'):  # pandas DataFrame/Series
                    arr = arr.values
                if isinstance(arr, (list, tuple)):
                    arr = np.array(arr)
                if arr.size > 0:
                    # Handle 2D arrays (multiple spectra)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    # Remove NaN and infinite values
                    valid_values = arr[np.isfinite(arr)]
                    if len(valid_values) > 0:
                        all_values.extend(valid_values)
        
        if not all_values:
            return None, None
            
        all_values = np.array(all_values)
        y_min, y_max = np.min(all_values), np.max(all_values)
        
        # Add padding
        y_range = y_max - y_min
        if y_range == 0:
            y_range = abs(y_max) * 0.1 if y_max != 0 else 1.0
        
        # 🔧 V1.2.1 Fix: Auto-detect small range data (e.g., after Standard Scale)
        if y_range < 50:
            print(f"⚠️  Visualization service: Detected small range data (range={y_range:.2f}), auto-adjusting padding")
            # For small range data, use at least 10% padding to ensure visibility
            padding = max(y_range * 0.1, 0.1)
        else:
            padding = y_range * padding_factor
        
        return y_min - padding, y_max + padding
    
    def set_original_y_limits(self, *data_arrays):
        """Set the original Y-axis limits for locking"""
        y_min, y_max = self.calculate_y_limits(*data_arrays)
        if y_min is not None and y_max is not None:
            self.original_y_limits = (y_min, y_max)
            print(f"🎯 Original Y-limits set: [{y_min:.4f}, {y_max:.4f}]")
    
    def get_y_limits(self, *current_data_arrays):
        """Get Y-axis limits based on lock state"""
        if self.y_axis_locked and self.original_y_limits:
            return self.original_y_limits
        else:
            return self.calculate_y_limits(*current_data_arrays)
    
    def update_preprocessing_queue(self, methods: List[str], active_scaling: bool = False):
        """Update the preprocessing methods queue for display"""
        self.current_preprocessing_queue = methods.copy()
        self.scaling_active = active_scaling
        print(f"📋 Preprocessing queue updated: {methods}")
        if active_scaling:
            print(f"⚖️ Scaling/normalization active")
    
    def get_preprocessing_queue_text(self):
        """Get formatted text for preprocessing queue display"""
        if not self.current_preprocessing_queue:
            return "No preprocessing applied"
        
        queue_text = " → ".join(self.current_preprocessing_queue)
        if self.scaling_active:
            queue_text += " [Scaling Active]"
        
        return queue_text
    
    def generate_legend_info(self, comparison_mode=False, show_individual=True, show_means=True):
        """Generate comprehensive legend information"""
        legend_info = {
            'elements': [],
            'explanations': [],
            'title': 'Legend'
        }
        
        if comparison_mode:
            legend_info['title'] = 'Preprocessing Comparison'
            legend_info['explanations'] = [
                'Solid lines: Original spectra',
                'Dashed lines: Processed spectra'
            ]
            
            if show_means:
                legend_info['elements'].extend([
                    {'label': 'Original (Mean)', 'color': 'blue', 'linestyle': '-', 'linewidth': 3},
                    {'label': 'Processed (Mean)', 'color': 'red', 'linestyle': '--', 'linewidth': 3}
                ])
            
            if show_individual:
                legend_info['explanations'].append('Individual spectra shown with thinner lines')
        
        else:
            legend_info['title'] = 'Spectral Data'
            if show_individual:
                legend_info['explanations'].append('Each line represents one spectrum')
        
        return legend_info
    
    def create_plot_annotations(self, ax, comparison_mode=False):
        """Add annotations to plot for better understanding"""
        annotations = []
        
        # Add preprocessing queue annotation
        if self.current_preprocessing_queue:
            queue_text = self.get_preprocessing_queue_text()
            annotation = ax.text(0.02, 0.02, f'Processing: {queue_text}', 
                               transform=ax.transAxes, fontsize=8, 
                               verticalalignment='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            annotations.append(annotation)
        
        # Add Y-axis lock status
        lock_status = "Y-axis: Locked" if self.y_axis_locked else "Y-axis: Auto-scale"
        lock_annotation = ax.text(0.98, 0.02, lock_status,
                                transform=ax.transAxes, fontsize=8,
                                verticalalignment='bottom', horizontalalignment='right',
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='lightgreen' if self.y_axis_locked else 'lightyellow', 
                                        alpha=0.7))
        annotations.append(lock_annotation)
        
        # Add line style explanation for comparison mode
        if comparison_mode:
            style_explanation = ax.text(0.02, 0.98, 'Solid: Original\nDashed: Processed', 
                                      transform=ax.transAxes, fontsize=9, 
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            annotations.append(style_explanation)
        
        return annotations
    
    def get_control_state(self):
        """Get current state of all visualization controls"""
        return {
            'y_axis_locked': self.y_axis_locked,
            'original_y_limits': self.original_y_limits,
            'preprocessing_queue': self.current_preprocessing_queue,
            'scaling_active': self.scaling_active
        }

# Global visualization control service instance
visualization_control = VisualizationControlService()
