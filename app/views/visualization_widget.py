from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QGroupBox, QSplitter, QFileDialog, QToolBar, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QSize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from plugins.preprocessing.spectrum_visualizer import SpectrumVisualizer
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix
from app.utils.unified_data_processor import UnifiedDataProcessor
import logging
matplotlib.use('Qt5Agg')

# V1.2.1: Import data flow tracker
from app.utils.data_flow_tracker import data_flow_tracker

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager

class VisualizationWidget(QWidget):
    """Spectral data visualization enhancement component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # V1.2.1: Dynamically adapt to display DPI
        optimal_dpi = self._get_optimal_dpi()
        self.figure = plt.figure(figsize=(10, 7), dpi=optimal_dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Initialize attributes (must be after DPI detection)
        self.current_theme = "default"
        self.current_plot_type = "Spectrum Plot"
        self.unified_processor = UnifiedDataProcessor()
        
        # Spectral type configuration
        self.spectral_type = None
        self.spectral_config = {}
        
        # Initialize visualization control service (Editor Comment 3)
        from app.services.visualization_control_service import visualization_control
        self.viz_control = visualization_control
        
        # Connect to visualization control signals
        self.viz_control.y_axis_lock_changed.connect(self.on_y_axis_lock_changed)
        
        # Continue UI initialization
        self.init_ui()
    
    def _get_optimal_dpi(self) -> int:
        """Intelligently select optimal DPI based on display resolution"""
        try:
            from PyQt5.QtWidgets import QApplication
            
            # Get primary screen information
            screen = QApplication.primaryScreen()
            if screen is None:
                return 150  # Default value
            
            # Get device pixel ratio (high DPI screens typically >1)
            pixel_ratio = screen.devicePixelRatio()
            
            # Intelligently select DPI based on pixel ratio
            if pixel_ratio >= 2.0:  # 4K/Retina display
                optimal_dpi = 200
                screen_type = "4K/Retina"
            elif pixel_ratio >= 1.5:  # 2K display
                optimal_dpi = 175
                screen_type = "2K"
            else:  # 1080p and below
                optimal_dpi = 150
                screen_type = "1080p"
            
            print(f"📺 Display adaptation: {screen_type} (pixel ratio={pixel_ratio:.2f}, DPI={optimal_dpi})")
            return optimal_dpi
            
        except Exception as e:
            print(f"⚠️ DPI detection failed, using default 150: {e}")
            return 150  # Use safe default on failure
    
    def init_ui(self):
        """Initialize the user interface (continue initialization)"""
        pass  # Original init_ui content has been completed in __init__
        
    def toggle_y_axis_lock(self):
        """Toggle Y-axis lock state (Editor Comment 3)"""
        self.viz_control.toggle_y_axis_lock()
    
    def on_y_axis_lock_changed(self, locked):
        """Handle Y-axis lock state change"""
        self.y_axis_lock_btn.setChecked(locked)
        self.y_axis_lock_btn.setText("🔒 Y-Axis Locked" if locked else "🔓 Y-Axis Auto")
        
        # Refresh current plot if available
        if hasattr(self, 'current_plot_data'):
            self.refresh_current_plot()
    
    def _get_axis_labels(self):
        """Get appropriate axis labels based on spectral type"""
        
        # Default labels
        xlabel = "Wavelength (nm)"
        ylabel = "Intensity (a.u.)"
        
        # If spectral config is available, use it
        if self.spectral_config:
            xlabel = self.spectral_config.get('x_label', xlabel)
            ylabel = self.spectral_config.get('y_label', ylabel)
            return xlabel, ylabel
        
        # If spectral type is set, infer labels
        if self.spectral_type:
            spectral_type = self.spectral_type.lower()
            
            if "nir" in spectral_type:
                xlabel = "Wavelength (nm)"
                ylabel = "Absorbance"
            elif "vis" in spectral_type:
                xlabel = "Wavelength (nm)"
                ylabel = "Absorbance"
            elif "raman" in spectral_type:
                xlabel = "Raman Shift (cm⁻¹)"
                ylabel = "Intensity (a.u.)"
            elif "mir" in spectral_type or "ftir" in spectral_type or "infrared" in spectral_type:
                xlabel = "Wavenumber (cm⁻¹)"
                ylabel = "Transmittance / Absorbance"
            elif "uv" in spectral_type:
                xlabel = "Wavelength (nm)"
                ylabel = "Absorbance"
            else:
                xlabel = "Wavelength / Wavenumber"
                ylabel = "Intensity"
        
        return xlabel, ylabel
        
    def init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add Y-axis lock control (Editor Comment 3)
        self.y_axis_lock_btn = QPushButton("🔒 Y-Axis Locked")
        self.y_axis_lock_btn.setCheckable(True)
        self.y_axis_lock_btn.setChecked(True)  # Default to locked
        self.y_axis_lock_btn.setToolTip("Toggle Y-axis locking for preprocessing comparison")
        self.y_axis_lock_btn.clicked.connect(self.toggle_y_axis_lock)
        toolbar_layout.addWidget(self.y_axis_lock_btn)
        
        plot_controls = QGroupBox()
        plot_controls.setStyleSheet("""
            QGroupBox { 
                border: none; 
                background-color: transparent;
            }
        """)
        plot_controls_layout = QHBoxLayout()
        plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        plot_controls_layout.setSpacing(12)
        
        plot_type_label = QLabel("Plot Type:")
        plot_type_label.setStyleSheet("""
            font-weight: bold;
            color: #666666;
            font-size: 9pt;
            font-family: 'Microsoft YaHei UI';
        """)
        plot_controls_layout.addWidget(plot_type_label)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Spectrum Plot", "Line Plot", "Scatter Plot", "Heatmap", "3D Plot", "Bar Chart"])
        self.plot_type_combo.setCurrentText("Spectrum Plot")  # Set default option to spectrum plot
        self.plot_type_combo.setFixedWidth(150)
        self.plot_type_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: white;
                font-size: 9pt;
                font-family: 'Microsoft YaHei UI';
            }
            QComboBox:hover {
                border: 1px solid #a0a0a0;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border-left: none;
            }
        """)
        self.plot_type_combo.currentTextChanged.connect(self.update_plot_type)
        plot_controls_layout.addWidget(self.plot_type_combo)
        
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("""
            font-weight: bold;
            color: #666666;
            font-size: 9pt;
            font-family: 'Microsoft YaHei UI';
        """)
        plot_controls_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["default", "ggplot", "seaborn-v0_8", "bmh", "grayscale"])
        self.theme_combo.setFixedWidth(150)
        self.theme_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: white;
                font-size: 9pt;
                font-family: 'Microsoft YaHei UI';
            }
            QComboBox:hover {
                border: 1px solid #a0a0a0;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border-left: none;
            }
        """)
        self.theme_combo.currentTextChanged.connect(self.update_theme)
        plot_controls_layout.addWidget(self.theme_combo)
        
        plot_controls_layout.addStretch()
        plot_controls.setLayout(plot_controls_layout)
        toolbar_layout.addWidget(plot_controls)
        
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f5f5f5;
                spacing: 5px;
                border: none;
                border-radius: 4px;
            }
            QToolButton {
                background-color: #a0a0a0;
                border: none;
                color: white;
                padding: 4px;
                margin: 2px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #888888;
                border-radius: 3px;
            }
            QToolButton:pressed {
                background-color: #707070;
            }
        """)
        toolbar_layout.addWidget(self.nav_toolbar)
        
        main_layout.addLayout(toolbar_layout)
        
        plot_area = QGroupBox()
        plot_area.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d9d9d9;
                border-radius: 6px;
                background-color: white;
                margin-top: 5px;
            }
        """)
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(10, 10, 10, 10)
        plot_layout.addWidget(self.canvas)
        plot_area.setLayout(plot_layout)
        
        main_layout.addWidget(plot_area)
        
        self.setup_plot_style()
        
    def setup_plot_style(self):
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = '#f8f8f8'
        plt.rcParams['figure.facecolor'] = '#ffffff'
        plt.rcParams['grid.color'] = '#e0e0e0'
        plt.rcParams['text.color'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#666666'
        plt.rcParams['ytick.color'] = '#666666'
        
        # Use responsive font sizes
        from app.utils.responsive_matplotlib import apply_responsive_font_sizes
        apply_responsive_font_sizes()
        
        self.figure.patch.set_facecolor('#ffffff')
        self.ax.set_facecolor('#f8f8f8')
        self.ax.grid(True, linestyle='-', alpha=0.1)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#d9d9d9')
        self.ax.spines['left'].set_color('#d9d9d9')
        self.ax.tick_params(colors='#666666', which='both')
        self.canvas.draw()
        
    def update_theme(self, theme):
        """Update plot theme"""
        self.current_theme = theme
        plt.style.use(theme)
        self.ax.clear()
        if hasattr(self, 'wavelengths') and hasattr(self, 'spectra'):
            self.plot_spectra(self.wavelengths, self.spectra)
        else:
            self.setup_plot_style()
            self.canvas.draw()
        
    def update_plot_type(self, plot_type):
        """Update plot type with error handling and validation"""
        # V1.2.1 fix: Save current state for complete recovery
        backup_plot_type = self.current_plot_type
        backup_is_3d = getattr(self, '_is_3d_plot', False)
        backup_wavelengths = getattr(self, 'wavelengths', None)
        backup_spectra = getattr(self, 'spectra', None)
        
        try:
            # Save current labels if available
            current_labels = getattr(self, 'current_labels', None)
            
            # Validate data availability
            if not hasattr(self, 'wavelengths') or not hasattr(self, 'spectra'):
                logging.warning(f"Cannot switch to {plot_type}: No data available")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Data", 
                                  "Please load data first before changing plot type.")
                # Reset to default
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText("Spectrum Plot")
                self.plot_type_combo.blockSignals(False)
                return
            
            # Validate data suitability for specific plot types
            if plot_type == "3D Plot" and self.spectra.shape[0] < 2:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Insufficient Data", 
                                  "3D Plot requires at least 2 spectra.")
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText(self.current_plot_type)
                self.plot_type_combo.blockSignals(False)
                return
            
            if plot_type == "Heatmap" and self.spectra.shape[0] < 3:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Insufficient Data", 
                                  "Heatmap requires at least 3 spectra.")
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText(self.current_plot_type)
                self.plot_type_combo.blockSignals(False)
                return
            
            # Restore 2D axis if switching from 3D
            if hasattr(self, '_is_3d_plot') and self._is_3d_plot and plot_type != "3D Plot":
                try:
                    self.ax.remove()
                    self.ax = self.figure.add_subplot(111)
                    self._is_3d_plot = False
                    logging.info("Restored 2D axis from 3D plot")
                except Exception as ax_error:
                    logging.error(f"Failed to restore 2D axis: {ax_error}")
                    raise  # Trigger exception handling recovery mechanism
            
            # Update current plot type
            self.current_plot_type = plot_type
            
            # Plot with new type
            self.plot_spectra(self.wavelengths, self.spectra, labels=current_labels)
            logging.info(f"Successfully switched to plot type: {plot_type}")
            
        except Exception as e:
            logging.error(f"Error switching plot type to {plot_type}: {e}", exc_info=True)
            
            # V1.2.1 fix: Complete state recovery mechanism
            try:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Plot Error", 
                                   f"Failed to switch to {plot_type}. Reverting to previous state.\n\nError: {str(e)[:100]}")
                
                # Restore data (prevent data corruption)
                if backup_wavelengths is not None and backup_spectra is not None:
                    self.wavelengths = backup_wavelengths
                    self.spectra = backup_spectra
                
                # Restore plot type
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText(backup_plot_type)
                self.plot_type_combo.blockSignals(False)
                self.current_plot_type = backup_plot_type
                
                # Restore axis type (2D/3D)
                current_is_3d = getattr(self, '_is_3d_plot', False)
                if backup_is_3d != current_is_3d:
                    self.ax.remove()
                    if backup_is_3d:
                        from mpl_toolkits.mplot3d import Axes3D
                        self.ax = self.figure.add_subplot(111, projection='3d')
                        self._is_3d_plot = True
                    else:
                        self.ax = self.figure.add_subplot(111)
                        self._is_3d_plot = False
                    logging.info(f"Restored axis type to {'3D' if backup_is_3d else '2D'}")
                
                # Try to redraw
                if hasattr(self, 'wavelengths') and hasattr(self, 'spectra'):
                    current_labels = getattr(self, 'current_labels', None)
                    self.plot_spectra(self.wavelengths, self.spectra, labels=current_labels)
                    logging.info("Successfully recovered to previous plot state")
                    
            except Exception as recovery_error:
                logging.error(f"Failed to recover plot: {recovery_error}", exc_info=True)
                # Last attempt: display error message
                try:
                    self.ax.clear()
                    self.ax.text(0.5, 0.5, "Plot recovery failed. Please reload data or restart.", 
                               ha='center', va='center', fontsize=11, color='red',
                               bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.8))
                    self.canvas.draw()
                except:
                    pass  # Cannot recover
        
    def plot_spectra(self, wavelengths, spectra, title="Spectrum Data", labels=None, 
                     original_spectra=None, comparison_mode=False):
        """Plot spectra data
        
        Args:
            wavelengths: wavelength data array
            spectra: spectral data array
            title: chart title
            labels: samples labels array, used to select spectra to display based on category
            original_spectra: original spectra for comparison mode
            comparison_mode: whether to show comparison between original and processed
        """
        
        # Save labels for plot type switching
        self.current_labels = labels
        
        if labels is not None:
            try:
                labels = np.array([str(label) for label in labels], dtype='<U50')
                logging.info(f"Labels safely converted to string array: {labels[:3] if len(labels) > 0 else []}")
            except Exception as label_error:
                logging.warning(f"Label conversion warning: {label_error}")
                labels = None
        
        # Use data alignment tool to handle dimension issues
        if wavelengths is not None and spectra is not None:
            try:
                from app.utils.data_alignment import DataDimensionAligner
                
                # Validate data
                validation_info = DataDimensionAligner.validate_spectral_data(wavelengths, spectra, "processed_spectra")
                
                if not validation_info['valid']:
                    error_msg = '; '.join(validation_info['errors'])
                    DataDimensionAligner.create_error_plot(self.ax, error_msg, "Visualization Error")
                    self.canvas.draw()
                    return
                
                # Print warning messages
                for warning in validation_info['warnings']:
                    logging.warning(warning)
                
                # Align data dimensions
                if comparison_mode and original_spectra is not None:
                    # Comparison mode: align three arrays
                    aligned_wavelengths, aligned_original, aligned_processed = DataDimensionAligner.align_spectral_data(
                        wavelengths, original_spectra, spectra
                    )
                    wavelengths = aligned_wavelengths
                    original_spectra = aligned_original
                    spectra = aligned_processed
                    
                    logging.info("Comparison mode dimension alignment completed:")
                    logging.info(f"   Wavelengths: {len(wavelengths)}")
                    logging.info(f"   Original spectra: {original_spectra.shape}")
                    logging.info(f"   Processed spectra: {spectra.shape}")
                    
                else:
                    # Normal mode: align two arrays
                    aligned_wavelengths, aligned_spectra = DataDimensionAligner.align_spectral_data(
                        wavelengths, spectra
                    )
                    wavelengths = aligned_wavelengths
                    spectra = aligned_spectra
                    
                    logging.info("Normal mode dimension alignment completed:")
                    logging.info(f"   Wavelengths: {len(wavelengths)}")
                    logging.info(f"   Spectra: {spectra.shape}")
            
            except Exception as e:
                logging.error(f"Error in data dimension alignment: {e}", exc_info=True)
                # Create error display
                DataDimensionAligner.create_error_plot(self.ax, str(e), "Visualization Error")
                self.canvas.draw()
                return
        
        logging.debug("Visualization received wavelength data")
        if wavelengths is not None:
            logging.debug(f"   Wavelength count: {len(wavelengths)}")
            logging.debug(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f}")
            logging.debug(f"   First 5 wavelengths: {wavelengths[:5]}")
            logging.debug(f"   Last 5 wavelengths: {wavelengths[-5:]}")
        else:
            logging.debug("   Wavelengths is None!")
        
        self.wavelengths = wavelengths
        self.spectra = spectra
        
        self.ax.clear()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        if wavelengths is None or spectra is None or len(wavelengths) == 0 or spectra.shape[0] == 0:
            self.ax.text(0.5, 0.5, "No valid spectral data", 
                       horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        if not np.all(np.diff(wavelengths) > 0):
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            spectra = spectra[:, sort_idx]
        
        selected_indices = []
        
        if labels is not None:
            labels = np.array(labels).flatten()
            unique_labels = np.unique(labels)
            
            try:
                def safe_convert_to_float(arr):
                    """Safely convert array to float, if contains non-numeric return original array"""
                    result = []
                    has_string_labels = False
                    
                    for item in arr:
                        try:
                            str_item = str(item).strip()
                            str_item = str_item.strip("'\"")
                            if str_item and not str_item.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                has_string_labels = True
                                break
                            float_val = float(item)
                            result.append(float_val)
                        except (ValueError, TypeError):
                            has_string_labels = True
                            break
                    
                    if has_string_labels:
                        return np.array(arr)
                    else:
                        return np.array(result) if result else None
                
                numeric_labels = safe_convert_to_float(labels)
                
                if numeric_labels is not None and numeric_labels.dtype.kind in ['U', 'S', 'O']:
                    is_categorical = True
                    logging.info("Label classification detection: Non-numeric labels detected, determined as categorical")
                elif numeric_labels is not None:
                    try:
                        if numeric_labels.dtype.kind in ['U', 'S', 'O']:
                            is_categorical = True
                        else:
                            unique_count = len(unique_labels)
                            total_count = len(labels)
                            
                            condition1 = unique_count <= 20
                            condition2 = unique_count < total_count * 0.3
                            condition3 = np.sum(np.equal(numeric_labels, np.round(numeric_labels))) > total_count * 0.9
                            
                            try:
                                numeric_unique = np.array([float(x) for x in unique_labels])
                                sorted_unique = np.sort(numeric_unique)
                                
                                if len(sorted_unique) >= 2:
                                    is_sequential = np.allclose(np.diff(sorted_unique), 1) and all(x == int(x) for x in sorted_unique)
                                    is_small_integers = all(x == int(x) and 0 <= x <= 100 for x in sorted_unique)
                                    condition4 = is_sequential or (is_small_integers and unique_count <= 10)
                                else:
                                    condition4 = False
                            except (ValueError, TypeError):
                                condition4 = False
                            
                            is_categorical = condition1 and condition2 and (condition3 or condition4)
                            
                            logging.debug("Classification analysis:")
                            logging.debug(f"  Unique count: {unique_count}, Total count: {total_count}")
                            logging.debug(f"  Condition1 (≤20 unique): {condition1}")
                            logging.debug(f"  Condition2 (<30% unique): {condition2}")
                            logging.debug(f"  Condition3 (>90% integers): {condition3}")
                            logging.debug(f"  Condition4 (classification pattern): {condition4}")
                            logging.debug(f"  Final decision: {'Categorical' if is_categorical else 'Continuous (Quantitative)'}")
                    except Exception as e:
                        logging.error(f"Error in numeric label analysis: {e}", exc_info=True)
                        is_categorical = True
                                     
                    logging.info(f"Final classification: {'Categorical labels' if is_categorical else 'Continuous values (Quantitative task)'}")
                else:
                    is_categorical = True
                    logging.info("Label classification detection: Conversion failed, determined as categorical")
            except:
                is_categorical = True
                logging.info("Label classification detection: Non-numeric labels, determined as categorical")
            
            if is_categorical:
                logging.info(f"Plotting by category, {len(unique_labels)} categories total")
                
                # Qualitative task display strategy: determine samples per category based on number of categories
                if len(unique_labels) < 5:
                    samples_per_category = 3  # <5 categories, show 3 per category
                else:
                    samples_per_category = 1  # ≥5 categories, show 1 per category
                
                logging.info(f"Qualitative task display strategy: {len(unique_labels)} categories, {samples_per_category} samples per category")
                
                for label in unique_labels:
                    label_indices = np.where(labels == label)[0]
                    samples_to_select = min(len(label_indices), samples_per_category)
                    selected_from_category = label_indices[:samples_to_select].tolist()
                    selected_indices.extend(selected_from_category)
                    logging.debug(f"  Category {label}: Selected {samples_to_select} samples from {len(label_indices)} available")
                
                logging.info(f"✅ Actually plotted {len(selected_indices)} spectra (total {spectra.shape[0]})")
            else:
                # Quantitative task display strategy: fixed display of 5
                display_count = min(5, spectra.shape[0])
                selected_indices = list(range(display_count))
                logging.info(f"Quantitative task display strategy: showing {display_count}/{spectra.shape[0]} spectra")
        else:
            # No label data: fixed 5
            display_count = min(5, spectra.shape[0])
            selected_indices = list(range(display_count))
            logging.info(f"No label data display strategy: showing {display_count}/{spectra.shape[0]} spectra")
        
        if len(selected_indices) == 0:
            selected_indices = list(range(min(5, spectra.shape[0])))
        
        if self.current_plot_type == "Line Plot" or self.current_plot_type == "Spectrum Plot":
            if comparison_mode and original_spectra is not None:
                # Comparison mode: plot original and processed spectra
                # First draw original spectra (solid lines)
                for i, idx in enumerate(selected_indices):
                    if labels is not None and is_categorical:
                        label = f"Original - {labels[idx]}"
                    else:
                        label = f"Original - Spectrum {i+1}"
                    
                    self.ax.plot(wavelengths, original_spectra[idx], linewidth=1.5, alpha=0.7, 
                               color=colors[i % len(colors)], linestyle='-',
                               label=label)
                
                # Then draw processed spectra (dashed lines)
                for i, idx in enumerate(selected_indices):
                    if labels is not None and is_categorical:
                        label = f"Processed - {labels[idx]}"
                    else:
                        label = f"Processed - Spectrum {i+1}"
                    
                    self.ax.plot(wavelengths, spectra[idx], linewidth=1.5, alpha=0.7, 
                               color=colors[i % len(colors)], linestyle='--',
                               label=label)
                
                # Add mean comparison
                original_mean = np.mean(original_spectra[selected_indices], axis=0)
                processed_mean = np.mean(spectra[selected_indices], axis=0)
                
                self.ax.plot(wavelengths, original_mean, color='blue', linewidth=3, 
                           label='Original (Mean)', alpha=0.9, linestyle='-')
                self.ax.plot(wavelengths, processed_mean, color='red', linewidth=3, 
                           label='Processed (Mean)', alpha=0.9, linestyle='--')
                
                # Calculate Y-axis range (including original data, processed data, and mean data)
                all_spectra = np.vstack([original_spectra[selected_indices], spectra[selected_indices]])
                # Also include mean data to ensure complete display
                all_data_with_means = np.vstack([all_spectra, [original_mean], [processed_mean]])
                y_min, y_max = np.min(all_data_with_means), np.max(all_data_with_means)
                
            else:
                # V1.2.1 fix: Simplify plotting, each spectrum has legend when count is small
                # Normal mode: plot selected spectra
                
                for i, idx in enumerate(selected_indices):
                    # Qualitative task shows category labels, quantitative task shows Sample labels
                    if labels is not None and is_categorical:
                        legend_label = f"{labels[idx]}"  # Qualitative task: show category label
                    else:
                        legend_label = f"Sample {i+1}"  # Quantitative task: show Sample label
                    
                    # Plot spectrum (each has legend, because count is already small)
                    self.ax.plot(wavelengths, spectra[idx], 
                               linewidth=1.8,  # Slightly thicker for better visibility
                               alpha=0.85, 
                               color=colors[i % len(colors)],
                               label=legend_label)
                
                logging.info(f"✅ Plotted {len(selected_indices)} spectra, each with legend")
                
                displayed_spectra = spectra[selected_indices]
                y_min, y_max = np.min(displayed_spectra), np.max(displayed_spectra)
            
            # Apply Y-axis limits using visualization control service (Editor Comment 3)
            if comparison_mode:
                # Set original Y-limits for locking
                self.viz_control.set_original_y_limits(original_spectra[selected_indices], spectra[selected_indices])
                
                # Get Y-limits based on lock state
                y_min, y_max = self.viz_control.get_y_limits(
                    original_spectra[selected_indices], 
                    spectra[selected_indices],
                    [original_mean], 
                    [processed_mean]
                )
            else:
                # For normal mode, calculate Y-limits
                y_min, y_max = self.viz_control.calculate_y_limits(spectra[selected_indices])
            
            # V1.2.1 critical fix: Automatically detect data range and adjust Y-axis
            actual_y_min = np.min(spectra[selected_indices])
            actual_y_max = np.max(spectra[selected_indices])
            data_range = actual_y_max - actual_y_min
            
            logging.info(f"📊 Y-axis range check: data range=({actual_y_min:.2f}, {actual_y_max:.2f}), range width={data_range:.2f}")
            
            # If data range is small (<50), likely normalized data
            if data_range < 50:
                logging.warning(f"⚠️  Detected small range data (range={data_range:.2f}), automatically adjusting Y-axis")
                # Use actual data range and add 10% margin
                margin = max(data_range * 0.1, 0.1)  # At least 0.1 margin
                y_min = actual_y_min - margin
                y_max = actual_y_max + margin
                logging.info(f"   Adjusted Y-axis: ({y_min:.2f}, {y_max:.2f})")
            
            if y_min is not None and y_max is not None:
                self.ax.set_ylim(y_min, y_max)
            
            # Add visualization control annotations
            annotations = self.viz_control.create_plot_annotations(self.ax, comparison_mode)
            
            # Update preprocessing queue for display
            if comparison_mode:
                self.viz_control.update_preprocessing_queue(
                    getattr(self, 'current_preprocessing_methods', []),
                    getattr(self, 'scaling_active', False)
                )
            
            self.ax.ticklabel_format(useOffset=False, style='plain')
            
            # Use dynamic axis labels (Line Plot / Spectrum Plot)
            xlabel, ylabel = self._get_axis_labels()
            font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
            self.ax.set_xlabel(xlabel, fontsize=font_sizes['axes.labelsize'], fontweight='bold')
            self.ax.set_ylabel(ylabel, fontsize=font_sizes['axes.labelsize'], fontweight='bold')
            
            # Add explanation text in comparison mode
            if comparison_mode and original_spectra is not None:
                self.ax.text(0.02, 0.98, 'Solid lines: Original spectra\nDashed lines: Processed spectra', 
                           transform=self.ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        elif self.current_plot_type == "Scatter Plot":
            if len(wavelengths) > 500:
                step = max(1, len(wavelengths) // 100)
            else:
                step = 1
            
            point_size = 30 if step > 1 else 15
            alpha = 0.8
            
            logging.info(f"Scatter plot sampling info: Original wavelength count = {len(wavelengths)}, Sampling step = {step}, Points after sampling = {len(wavelengths[::step])}")
            
            # V1.2.1 fix: Simplify scatter plot, each has legend
            for i, idx in enumerate(selected_indices):
                # Qualitative task shows category labels, quantitative task shows Sample labels
                if labels is not None and is_categorical:
                    legend_label = f"{labels[idx]}"  # Qualitative task: show category label
                else:
                    legend_label = f"Sample {i+1}"  # Quantitative task: show Sample label
                
                # Plot scatter points (each has legend)
                scatter = self.ax.scatter(
                    wavelengths[::step], 
                    spectra[idx, ::step], 
                    s=point_size, 
                    alpha=alpha,
                    color=colors[i % len(colors)],
                    label=legend_label,
                    marker='o'
                )
                
                logging.debug(f"Spectrum {idx+1}: {len(wavelengths[::step])} points")
            
            logging.info(f"✅ Plotted {len(selected_indices)} scatter spectra")
            
            self.ax.set_xlim(np.min(wavelengths), np.max(wavelengths))
            
            displayed_spectra = spectra[selected_indices]
            y_min, y_max = np.min(displayed_spectra), np.max(displayed_spectra)
            y_range = y_max - y_min
            y_padding = max(y_range * 0.05, 0.001)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            self.ax.grid(True, linestyle='-', linewidth=0.8, alpha=0.5)
            
            self.ax.ticklabel_format(useOffset=False, style='plain')
            
            # Use dynamic axis labels (Scatter Plot)
            xlabel, ylabel = self._get_axis_labels()
            font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
            self.ax.set_xlabel(xlabel, fontsize=font_sizes['axes.labelsize'], fontweight='bold')
            self.ax.set_ylabel(ylabel, fontsize=font_sizes['axes.labelsize'], fontweight='bold')
            
        elif self.current_plot_type == "Heatmap":
            max_spectra = min(spectra.shape[0], 20)
            
            extent = [np.min(wavelengths), np.max(wavelengths), 0, max_spectra]
            
            im = self.ax.imshow(spectra[:max_spectra], aspect='auto', 
                              interpolation='nearest', cmap='viridis',
                              extent=tuple(extent))
                              
            self.ax.ticklabel_format(axis='x', useOffset=False, style='plain')
            
            self.ax.set_yticks(np.arange(0.5, max_spectra, 1))
            self.ax.set_yticklabels(np.arange(1, max_spectra+1, 1))
            
            # Use dynamic axis labels (Heatmap - X-axis)
            xlabel, ylabel_intensity = self._get_axis_labels()
            self.ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            self.ax.set_ylabel("Spectrum Index", fontsize=14, fontweight='bold')
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label(ylabel_intensity, fontsize=12)
            
        elif self.current_plot_type == "3D Plot":
            # Mark as 3D plot and create/recreate 3D axis
            if not hasattr(self.ax, 'zaxis'):  # Check if already 3D
                self.ax.remove()
                from mpl_toolkits.mplot3d import Axes3D
                self.ax = self.figure.add_subplot(111, projection='3d')
            self._is_3d_plot = True
            
            if len(wavelengths) > 500:
                step = len(wavelengths) // 500
            else:
                step = 1
            
            max_spectra = min(spectra.shape[0], 10)
            
            for i in range(max_spectra):
                if labels is not None and is_categorical:
                    label = f"{labels[i]}"
                else:
                    label = f"Spectrum {i+1}"
                
                self.ax.plot(wavelengths[::step], [i] * len(wavelengths[::step]), 
                          spectra[i, ::step], linewidth=2, 
                          color=colors[i % len(colors)],
                          label=label)
            
            # Use dynamic axis labels (3D Plot)
            xlabel, ylabel = self._get_axis_labels()
            self.ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            self.ax.set_ylabel("Spectrum Index", fontsize=14, fontweight='bold')
            self.ax.set_zlabel(ylabel, fontsize=14, fontweight='bold')
            self.ax.legend(loc='best', fontsize='small')
            
        elif self.current_plot_type == "Bar Chart":
            means = np.mean(spectra, axis=1)
            max_spectra = min(len(means), 15)
            
            x_labels = []
            for i in range(max_spectra):
                if labels is not None and is_categorical:
                    x_labels.append(f"{labels[i]}")
                else:
                    x_labels.append(f"Spectrum {i+1}")
            
            bars = self.ax.bar(range(1, max_spectra+1), means[:max_spectra], 
                      color=colors[:max_spectra], alpha=0.85, width=0.6)
            
            self.ax.set_xticks(range(1, max_spectra+1))
            self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            
            font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
            self.ax.set_xlabel("Spectrum", fontsize=font_sizes['axes.labelsize'], fontweight='medium')
            self.ax.set_ylabel("Average Absorbance", fontsize=font_sizes['axes.labelsize'], fontweight='medium')
        
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
        self.ax.set_title(title, fontsize=font_sizes['axes.titlesize'], fontweight='bold')
        
        self.ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
        
        # V1.2.1 fix: Intelligent legend display with responsive font size
        if self.current_plot_type in ["Line Plot", "Scatter Plot", "Spectrum Plot"]:
            # Check if there are legend entries
            handles, labels_list = self.ax.get_legend_handles_labels()
            if handles and labels_list:
                # Get responsive font sizes from ui_scaling_manager
                font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
                legend_fontsize = font_sizes.get('legend.fontsize', 11)
                
                # Legend style optimization: more compact
                legend = self.ax.legend(
                    loc='upper right',  # Fixed at upper right, avoid blocking data
                    frameon=True, 
                    framealpha=0.85,  # Slightly increase transparency
                    fontsize=legend_fontsize,  # Responsive font size
                    ncol=1 if len(labels_list) <= 5 else 2,  # Two columns when more than 5
                    borderpad=0.5,  # Reduce padding
                    labelspacing=0.3,  # Reduce label spacing
                    handlelength=1.5,  # Shorten legend line length
                    columnspacing=1.0  # Column spacing
                )
                logging.info(f"Display legend: {len(labels_list)} entries, font size: {legend_fontsize}")
            else:
                logging.info("No legend entries, skip legend display")
            
        if wavelengths is not None and len(wavelengths) > 0:
            x_min, x_max = np.min(wavelengths), np.max(wavelengths)
            self.ax.set_xlim(x_min, x_max)
            
            if spectra is not None and spectra.size > 0:
                y_min, y_max = np.min(spectra), np.max(spectra)
                y_padding = (y_max - y_min) * 0.05
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_spectra_comparison(self, wavelengths, original_spectra, processed_spectra, 
                               title="Spectra Comparison"):
        """Plot original and processed spectra comparison"""
        if (wavelengths is None or original_spectra is None or processed_spectra is None or 
            len(wavelengths) == 0 or original_spectra.size == 0 or processed_spectra.size == 0):
            self.ax.text(0.5, 0.5, "No valid comparison data", 
                      horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        try:
            wavelengths = np.array(wavelengths, dtype=float)
            
            def safe_extract_numeric_data(data):
                """Safely extract numerical data, excluding label columns"""
                if isinstance(data, (list, tuple)):
                    data = np.array(data)
                
                if data.ndim == 1:
                    try:
                        return np.array(data, dtype=float)
                    except (ValueError, TypeError):
                        try:
                            return np.array(data[1:], dtype=float)
                        except:
                            raise ValueError("Data contains non-numeric values that cannot be converted")
                else:
                    try:
                        return np.array(data, dtype=float)
                    except (ValueError, TypeError):
                        try:
                            if data.shape[1] > 1:
                                numeric_data = data[:, 1:]
                                return np.array(numeric_data, dtype=float)
                            else:
                                raise ValueError("Data appears to be labels only")
                        except:
                            raise ValueError("Data contains non-numeric values in spectral columns")
            
            original_spectra = safe_extract_numeric_data(original_spectra)
            processed_spectra = safe_extract_numeric_data(processed_spectra)
            
            logging.info("Successfully extracted numerical spectral data:")
            logging.info(f"    Original spectrum: {original_spectra.shape}")
            logging.info(f"    Processed spectrum: {processed_spectra.shape}")
            logging.info(f"    Wavelength count: {len(wavelengths)}")
            
        except Exception as conversion_error:
            logging.error(f"Spectral data conversion failed: {conversion_error}", exc_info=True)
            self.ax.text(0.5, 0.5, f"Data conversion failed: {str(conversion_error)}", 
                      horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        
        if original_spectra.ndim == 1:
            original_spectra = original_spectra.reshape(1, -1)
        if processed_spectra.ndim == 1:
            processed_spectra = processed_spectra.reshape(1, -1)
            
        if len(wavelengths) != original_spectra.shape[1]:
            if len(wavelengths) > original_spectra.shape[1]:
                wavelengths = wavelengths[:original_spectra.shape[1]]
            else:
                original_spectra = original_spectra[:, :len(wavelengths)]
                
        if len(wavelengths) != processed_spectra.shape[1]:
            if len(wavelengths) > processed_spectra.shape[1]:
                wavelengths = wavelengths[:processed_spectra.shape[1]]
            else:
                processed_spectra = processed_spectra[:, :len(wavelengths)]
                
        if not np.all(np.diff(wavelengths) > 0):
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            original_spectra = original_spectra[:, sort_idx]
            processed_spectra = processed_spectra[:, sort_idx]
                
        self.ax.clear()
        
        original_mean = np.mean(original_spectra, axis=0)
        processed_mean = np.mean(processed_spectra, axis=0)
        
        self.ax.plot(wavelengths, original_mean, color='#1f77b4', linestyle='-', 
                   linewidth=2.0, alpha=0.9, 
                   label="Original (Mean)")
        self.ax.plot(wavelengths, processed_mean, color='#ff7f0e', linestyle='-', 
                   linewidth=2.0, alpha=0.9, 
                   label="Processed (Mean)")
        
        if original_spectra.shape[0] > 1:
            for i in range(min(3, original_spectra.shape[0])):
                self.ax.plot(wavelengths, original_spectra[i], color='#1f77b4', 
                           linestyle='--', linewidth=0.8, alpha=0.3)
                self.ax.plot(wavelengths, processed_spectra[i], color='#ff7f0e', 
                           linestyle='--', linewidth=0.8, alpha=0.3)
        
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
            
        self.ax.set_xlim(np.min(wavelengths), np.max(wavelengths))
        
        all_data = np.concatenate([original_spectra.flatten(), processed_spectra.flatten()])
        self.ax.set_ylim(np.min(all_data), np.max(all_data))
        
        self.ax.ticklabel_format(useOffset=False, style='plain')
            
        # Use dynamic axis labels (Comparison plot)
        xlabel, ylabel = self._get_axis_labels()
        self.ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        self.ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
        
        self.ax.legend(loc='best', frameon=True, framealpha=0.7, fontsize=10)
        
        self.ax.set_facecolor('#f8f8f8')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def zoom_in(self):
        """Zoom in the plot"""
        self.ax.set_autoscale_on(False)
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.ax.set_xlim(x_min + x_range * 0.1, x_max - x_range * 0.1)
        self.ax.set_ylim(y_min + y_range * 0.1, y_max - y_range * 0.1)
        
        self.canvas.draw()
        
    def zoom_out(self):
        """Zoom out the plot"""
        self.ax.set_autoscale_on(False)
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        self.ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
        self.ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        self.canvas.draw()
        
    def save_plot(self, file_path):
        """Save current plot to file with high resolution"""
        try:
            # Increase save DPI to 600 (publication quality)
            self.figure.savefig(file_path, dpi=600, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            logging.info(f"Plot saved successfully at 600 DPI: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving plot: {e}", exc_info=True)
            return False

    def update_preprocessing_preview(self, data_model, wavelengths, preview_mode, methods_applied=None):
        """
        Updates the plot to show original, processed, or a comparison of spectral data
        based on the selected preview mode, with improved legend handling.
        
        V1.2.1: Add detailed data source verification
        """
        print("=" * 60)
        print(f"📺 Visualization preview: mode={preview_mode}")
        self.ax.clear()

        try:
            from app.utils.data_alignment import DataDimensionAligner
        except ImportError:
            self.ax.text(0.5, 0.5, "Error: DataDimensionAligner not found.", ha='center', va='center', color='red')
            self.canvas.draw()
            return

        # 1. Get data from data_model
        X_original = data_model.get_X()
        X_processed = data_model.X_processed
        y_labels = data_model.get_y()
        
        # V1.2.1: Data source verification
        print(f"   Data source: data_model.get_X() → {'exists' if X_original is not None else 'not exists'}")
        print(f"   Preprocessed data: data_model.X_processed → {'exists' if X_processed is not None else 'not exists'}")
        if X_original is not None:
            print(f"   Original data shape: {X_original.shape if hasattr(X_original, 'shape') else 'N/A'}")
        if X_processed is not None:
            print(f"   Processed data shape: {X_processed.shape if hasattr(X_processed, 'shape') else 'N/A'}")

        # 2. Handle missing data
        if X_original is None or wavelengths is None:
            self.ax.text(0.5, 0.5, "No original data (X) available for preview.", ha='center', va='center')
            self.canvas.draw()
            return
        
        if y_labels is not None:
            y_labels = np.array(y_labels)

        # V1.2.1 fix: If no preprocessed data, should not display, but warn
        if X_processed is None and preview_mode in ['Processed', 'Comparison']:
            self.ax.text(0.5, 0.5, 
                        "Preprocessed data unavailable\n\nPlease apply preprocessing methods first\nThen you can view preprocessing effects", 
                        ha='center', va='center', 
                        fontsize=14, color='red',
                        bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.8))
            self.canvas.draw()
            logging.warning("Attempted to view Processed/Comparison mode without preprocessed data")
            return
        
        # If Original mode, X_processed can be missing
        if X_processed is None and preview_mode == 'Original':
            X_processed = X_original.copy()  # Only for placeholder, will not be displayed

        # 3. Align dimensions
        try:
            X_original_np = np.array(X_original)
            X_processed_np = np.array(X_processed)
            wavelengths_np = np.array(wavelengths)

            aligned_wavelengths, aligned_original, aligned_processed = DataDimensionAligner.align_spectral_data(
                wavelengths_np, X_original_np, X_processed_np
            )
        except Exception as e:
            DataDimensionAligner.create_error_plot(self.ax, f"Dimension Alignment Failed: {e}", "Data Error")
            self.canvas.draw()
            return

        # 4. Plotting logic
        show_original = preview_mode in ["Original", "Comparison"]
        show_processed = preview_mode in ["Processed", "Comparison"]
        
        print(f"   Show Original: {show_original}")
        print(f"   Show Processed: {show_processed}")
        print("=" * 60)
        
        # V1.2.1 fix: Preprocessing visualization should also distinguish qualitative/quantitative tasks
        selected_indices = []
        is_categorical = False
        
        # Determine if it's a qualitative task
        if y_labels is not None:
            unique_labels = np.unique(y_labels)
            # Simple judgment: if number of labels < 30% of samples, consider it qualitative task
            if len(unique_labels) < len(y_labels) * 0.3:
                is_categorical = True
                
                # Qualitative task: select by category
                if len(unique_labels) < 5:
                    samples_per_category = 3  # <5 categories, 3 per category
                else:
                    samples_per_category = 1  # ≥5 categories, 1 per category
                
                logging.info(f"Preprocessing visualization - qualitative task: {len(unique_labels)} categories, {samples_per_category} per category")
                
                for label in unique_labels:
                    label_indices = np.where(y_labels == label)[0]
                    samples_to_select = min(len(label_indices), samples_per_category)
                    selected_from_category = label_indices[:samples_to_select].tolist()
                    selected_indices.extend(selected_from_category)
            else:
                # Quantitative task: fixed 5
                selected_indices = list(range(min(5, aligned_original.shape[0])))
                logging.info(f"Preprocessing visualization - quantitative task: showing {len(selected_indices)} spectra")
        else:
            # No labels: fixed 5
            selected_indices = list(range(min(5, aligned_original.shape[0])))
            logging.info(f"Preprocessing visualization - no labels: showing {len(selected_indices)} spectra")
        
        cmap = plt.get_cmap("tab10")
        mean_original, mean_processed = None, None

        # Plot individual spectra with proper labels
        for i, idx in enumerate(selected_indices):
            # Qualitative task shows category labels, quantitative task shows Sample labels
            if y_labels is not None and is_categorical:
                sample_label = f"{y_labels[idx]}"  # Qualitative task: show category label
            else:
                sample_label = f"Sample {i+1}"  # Quantitative task: show Sample label
            
            if show_original:
                label = f"{sample_label} (Original)" if preview_mode == "Comparison" else str(sample_label)
                self.ax.plot(aligned_wavelengths, aligned_original[idx], color=cmap(i % 10), linestyle='-', linewidth=0.8, alpha=0.7, label=label)

            if show_processed:
                label = f"{sample_label} (Processed)" if preview_mode == "Comparison" else str(sample_label)
                linestyle = '--' if preview_mode == "Comparison" else '-'
                self.ax.plot(aligned_wavelengths, aligned_processed[idx], color=cmap(i % 10), linestyle=linestyle, linewidth=1.0, alpha=0.8, label=label)

        # Plot mean and std bands without adding to legend in comparison mode
        if show_original:
            mean_original = np.mean(aligned_original, axis=0)
            std_original = np.std(aligned_original, axis=0)
            self.ax.plot(aligned_wavelengths, mean_original, color='black', linestyle='-', linewidth=2.5, alpha=0.6, label='Original Mean' if preview_mode != "Comparison" else '_nolegend_')
            self.ax.fill_between(aligned_wavelengths, mean_original - std_original, mean_original + std_original, color='gray', alpha=0.15, label='_nolegend_')

        if show_processed:
            mean_processed = np.mean(aligned_processed, axis=0)
            std_processed = np.std(aligned_processed, axis=0)
            linestyle = '--' if preview_mode == "Comparison" else '-'
            self.ax.plot(aligned_wavelengths, mean_processed, color='black', linestyle=linestyle, linewidth=2.5, alpha=0.6, label='Processed Mean' if preview_mode != "Comparison" else '_nolegend_')
            self.ax.fill_between(aligned_wavelengths, mean_processed - std_processed, mean_processed + std_processed, color='gray', alpha=0.15, label='_nolegend_')

        # 5. Set title and labels
        if preview_mode == "Comparison":
            title = "Preprocessing Comparison"
        elif preview_mode == "Original":
            title = "Original Spectra"
        else: # Processed
            title = "Processed Spectra"
        
        if methods_applied:
            title += f": {', '.join(methods_applied)}"
        
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.ax.set_xlabel("Wavelength", fontsize=11)
        self.ax.set_ylabel("Intensity", fontsize=11)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # 6. Handle Y-axis locking
        if hasattr(self, 'viz_control'):
             self.viz_control.set_original_y_limits(aligned_original, aligned_processed)
             
             original_data_for_lim = [mean_original] if mean_original is not None else []
             processed_data_for_lim = [mean_processed] if mean_processed is not None else []

             y_min, y_max = self.viz_control.get_y_limits(
                 aligned_original if show_original else None, 
                 aligned_processed if show_processed else None,
                 original_data_for_lim,
                 processed_data_for_lim
             )
             if y_min is not None and y_max is not None:
                self.ax.set_ylim(y_min, y_max)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear current plot content"""
        self.ax.clear()
        self.canvas.draw()
        return True
        
    def plot_scatter(self, x_data, y_data, title="Scatter Plot", xlabel="X", ylabel="Y"):
        """Plot scatter plot, used for actual value vs prediction value in regression analysis
        
        Args:
            x_data: X-axis data (usually actual value)
            y_data: Y-axis data (usually prediction value)
            title: Chart title
            xlabel: X-axis labels
            ylabel: Y-axis labels
        """
        self.ax.clear()
        
        self.ax.scatter(x_data, y_data, s=20, alpha=0.7, c='#1f77b4', edgecolors='k', linewidths=0.5)
        
        min_val = min(np.min(x_data), np.min(y_data))
        max_val = max(np.max(x_data), np.max(y_data))
        self.ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_confusion_matrix(self, confusion_matrix, class_names=None, title="Confusion Matrix"):
        """Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Chart title
        """
        # V1.2.1 critical fix: Completely reset figure to avoid colorbar accumulation causing left shift
        # Save current axes position
        original_position = self.ax.get_position()
        
        # Clear all axes (including colorbar axes)
        self.figure.clear()
        
        # Recreate main axes, restore original position
        self.ax = self.figure.add_subplot(111)
        self.ax.set_position(original_position)
        
        if confusion_matrix is None:
            logging.error("Confusion matrix data is empty")
            self.ax.text(0.5, 0.5, "No confusion matrix data available", 
                       horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        logging.debug("Confusion matrix plotting debug info:")
        logging.debug(f"VisualizationWidget: Input class_names: {class_names}")
        logging.debug(f"Confusion matrix shape: {confusion_matrix.shape}")
        
        matrix_size = confusion_matrix.shape[0]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(matrix_size)]
            logging.warning("No class names provided, using default indices")
        
        if len(class_names) != matrix_size:
            logging.warning(f"Label count mismatch: {len(class_names)} vs {matrix_size}")
            
            if len(class_names) > matrix_size:
                class_names = class_names[:matrix_size]
                logging.info(f"Truncate labels to {matrix_size} classes")
            else:
                original_count = len(class_names)
                for i in range(matrix_size - original_count):
                    class_names.append(f"Missing_{original_count + i}")
                logging.info(f"Supplement labels to {matrix_size} classes")
        
        cleaned_class_names = []
        for i, name in enumerate(class_names):
            str_name = str(name).strip()
            
            if len(str_name) > 15:
                str_name = str_name[:13] + ".."
            
            str_name = str_name.replace('\n', ' ').replace('\t', ' ')
            
            cleaned_class_names.append(str_name)
        
        class_names = cleaned_class_names
        logging.debug(f"Cleaned labels: {class_names}")
        
        if len(class_names) != matrix_size:
            logging.warning("Final label count still mismatch, force repair")
            if len(class_names) > matrix_size:
                class_names = class_names[:matrix_size]
            else:
                while len(class_names) < matrix_size:
                    class_names.append(f"Fix_{len(class_names)}")
        
        logging.debug(f"Final label list ({len(class_names)}): {class_names}")
        
        num_classes = matrix_size
        
        if num_classes <= 10:
            logging.info("Using small confusion matrix plotting")
            self._plot_small_confusion_matrix(confusion_matrix, class_names, title)
        elif num_classes <= 25:
            logging.info("Using medium confusion matrix plotting")
            self._plot_medium_confusion_matrix(confusion_matrix, class_names, title)
        else:
            logging.info("Using large confusion matrix plotting")
            self._plot_large_confusion_matrix(confusion_matrix, class_names, title)
        
        logging.info(f"VisualizationWidget: Confusion matrix plotting completed, classes: {num_classes}")
        
    def _plot_small_confusion_matrix(self, cm, class_names, title):
        """Plot small confusion matrix (≤10 classes)"""
        logging.debug(f"Plotting small confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.get_cmap("Blues")
        
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            # Use automatic layout, do not specify fixed values
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            logging.warning(f"Color bar addition failed: {e}")
        
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        self.ax.set_xlabel('Predicted', fontsize=10)
        self.ax.set_ylabel('True', fontsize=10)
        
        tick_marks = np.arange(len(class_names))
        self.ax.set_xticks(tick_marks)
        self.ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        self.ax.set_yticks(tick_marks)
        self.ax.set_yticklabels(class_names, fontsize=9)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=8)
        
        self.ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.5)
        
        # Use tight_layout to automatically center and adjust layout
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_medium_confusion_matrix(self, cm, class_names, title):
        """Plot medium confusion matrix (11-25 classes) - improved version"""
        logging.debug(f"Plotting medium confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.get_cmap("Blues")
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            # Use automatic layout, do not specify fixed values
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            logging.warning(f"Color bar addition failed: {e}")
        
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        self.ax.set_xlabel('Predicted', fontsize=9)
        self.ax.set_ylabel('True', fontsize=9)
        
        tick_marks = np.arange(len(class_names))
        self.ax.set_xticks(tick_marks)
        self.ax.set_xticklabels(class_names, rotation=90, ha='center', fontsize=7)
        self.ax.set_yticks(tick_marks)
        self.ax.set_yticklabels(class_names, fontsize=7)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j or cm[i, j] > thresh or cm[i, j] > 0:
                    self.ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontsize=6)
        
        # Use tight_layout to automatically center and adjust layout
        self.figure.tight_layout()
        
        self.canvas.draw()
        
    def _plot_large_confusion_matrix(self, cm, class_names, title):
        """Plot large confusion matrix (>25 classes) - improved version"""
        logging.debug(f"Plotting large confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.get_cmap("Blues")
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            # Use automatic layout, do not specify fixed values
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            logging.warning(f"Color bar addition failed: {e}")
        
        self.ax.set_title(title, fontsize=10, fontweight='bold')
        self.ax.set_xlabel('Predicted', fontsize=8)
        self.ax.set_ylabel('True', fontsize=8)
        
        num_classes = len(class_names)
        if num_classes > 30:
            step = max(1, num_classes // 15)
            
            tick_indices = list(range(0, num_classes, step))
            if tick_indices[-1] != num_classes - 1:
                tick_indices.append(num_classes - 1)
            
            tick_labels = [class_names[i] for i in tick_indices]
            
            self.ax.set_xticks(tick_indices)
            self.ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=6)
            self.ax.set_yticks(tick_indices)
            self.ax.set_yticklabels(tick_labels, fontsize=6)
        else:
            tick_marks = np.arange(len(class_names))
            self.ax.set_xticks(tick_marks)
            self.ax.set_xticklabels(class_names, rotation=90, ha='center', fontsize=5)
            self.ax.set_yticks(tick_marks)
            self.ax.set_yticklabels(class_names, fontsize=5)
        
        # Use tight_layout to automatically center and adjust layout
        self.figure.tight_layout()
        self.canvas.draw() 