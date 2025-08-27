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
matplotlib.use('Qt5Agg')

class VisualizationWidget(QWidget):
    """Spectral data visualization enhancement component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.init_ui()
        self.current_theme = "default"
        self.current_plot_type = "Spectrum Plot"
        self.unified_processor = UnifiedDataProcessor()
        
    def init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
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
        """Update plot type"""
        self.current_plot_type = plot_type
        if hasattr(self, 'wavelengths') and hasattr(self, 'spectra'):
            self.plot_spectra(self.wavelengths, self.spectra)
        
    def plot_spectra(self, wavelengths, spectra, title="Spectrum Data", labels=None):
        """Plot spectra data
        
        Args:
            wavelengths: wavelength data array
            spectra: spectral data array
            title: chart title
            labels: sampleslabels array, used to select spectra to display based on category
        """
        
        if labels is not None:
            try:
                labels = np.array([str(label) for label in labels], dtype='<U50')
                print(f"🔧 Labels safely converted to string array: {labels[:3] if len(labels) > 0 else []}")
            except Exception as label_error:
                print(f"⚠️ Label conversion warning: {label_error}")
                labels = None
        
        if wavelengths is not None and spectra is not None:
            try:
                wavelengths = np.array(wavelengths, dtype=float)
                spectra = np.array(spectra, dtype=float)
                
                if spectra.ndim == 1:
                    spectra = spectra.reshape(1, -1)
                    
                if len(wavelengths) != spectra.shape[1]:
                    if len(wavelengths) > spectra.shape[1]:
                        wavelengths = wavelengths[:spectra.shape[1]]
                    else:
                        spectra = spectra[:, :len(wavelengths)]
            
            except Exception as e:
                print(f"Error in wavelength conversion: {e}")
                return
        
        print(f"🔧 DEBUG: Visualization received wavelength data")
        if wavelengths is not None:
            print(f"   Wavelength count: {len(wavelengths)}")
            print(f"   Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f}")
            print(f"   First 5 wavelengths: {wavelengths[:5]}")
            print(f"   Last 5 wavelengths: {wavelengths[-5:]}")
        else:
            print("   Wavelengths is None!")
        
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
                    print(f"Label classification detection: Non-numeric labels detected, determined as categorical")
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
                            
                            print(f"Classification analysis:")
                            print(f"  Unique count: {unique_count}, Total count: {total_count}")
                            print(f"  Condition1 (≤20 unique): {condition1}")
                            print(f"  Condition2 (<30% unique): {condition2}")
                            print(f"  Condition3 (>90% integers): {condition3}")
                            print(f"  Condition4 (classification pattern): {condition4}")
                            print(f"  Final decision: {'Categorical' if is_categorical else 'Continuous (Quantitative)'}")
                    except Exception as e:
                        print(f"Error in numeric label analysis: {e}")
                        is_categorical = True
                                     
                    print(f"Final classification: {'Categorical labels' if is_categorical else 'Continuous values (Quantitative task)'}")
                else:
                    is_categorical = True
                    print(f"Label classification detection: Conversion failed, determined as categorical")
            except:
                is_categorical = True
                print(f"Label classification detection: Non-numeric labels, determined as categorical")
            
            if is_categorical:
                print(f"Plotting by category, {len(unique_labels)} categories total")
                for label in unique_labels:
                    label_indices = np.where(labels == label)[0]
                    if len(label_indices) > 0:
                        selected_idx = label_indices[0]
                        selected_indices.append(selected_idx)
                        print(f"  Category {label}: Selected sample {selected_idx}")
            else:
                print(f"Quantitative task: Plotting first {min(10, spectra.shape[0])} spectra")
                selected_indices = list(range(min(10, spectra.shape[0])))
        else:
            is_categorical = False
            print("No labels, selecting first spectra")
            selected_indices = list(range(min(10, spectra.shape[0])))
        
        if len(selected_indices) == 0:
            selected_indices = list(range(min(10, spectra.shape[0])))
        
        if self.current_plot_type == "Line Plot" or self.current_plot_type == "Spectrum Plot":
            for i, idx in enumerate(selected_indices):
                if labels is not None and is_categorical:
                    label = f"{labels[idx]}"
                else:
                    label = f"Spectrum {i+1}"
                
                self.ax.plot(wavelengths, spectra[idx], linewidth=1.5, alpha=0.85, 
                           color=colors[i % len(colors)],
                           label=label)
            
            self.ax.set_xlim(np.min(wavelengths), np.max(wavelengths))
            
            displayed_spectra = spectra[selected_indices]
            y_min, y_max = np.min(displayed_spectra), np.max(displayed_spectra)
            y_range = y_max - y_min
            
            y_padding = max(y_range * 0.05, 0.001)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            self.ax.ticklabel_format(useOffset=False, style='plain')
            
            self.ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
            self.ax.set_ylabel("Absorbance", fontsize=11, fontweight='medium')
            
        elif self.current_plot_type == "Scatter Plot":
            if len(wavelengths) > 500:
                step = max(1, len(wavelengths) // 100)
            else:
                step = 1
            
            point_size = 30 if step > 1 else 15
            alpha = 0.8
            
            print(f"Scatter plot sampling info: Original wavelength count = {len(wavelengths)}, Sampling step = {step}, Points after sampling = {len(wavelengths[::step])}")
            
            for i, idx in enumerate(selected_indices):
                if labels is not None and is_categorical:
                    label = f"{labels[idx]}"
                else:
                    label = f"Spectrum {i+1}"
                
                scatter = self.ax.scatter(
                    wavelengths[::step], 
                    spectra[idx, ::step], 
                    s=point_size, 
                    alpha=alpha,
                    color=colors[i % len(colors)],
                    label=label,
                    marker='o'
                )
                
                print(f"Spectrum {idx+1}: {len(wavelengths[::step])} points")
            
            self.ax.set_xlim(np.min(wavelengths), np.max(wavelengths))
            
            displayed_spectra = spectra[selected_indices]
            y_min, y_max = np.min(displayed_spectra), np.max(displayed_spectra)
            y_range = y_max - y_min
            y_padding = max(y_range * 0.05, 0.001)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            self.ax.grid(True, linestyle='-', linewidth=0.8, alpha=0.5)
            
            self.ax.ticklabel_format(useOffset=False, style='plain')
            
            self.ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
            self.ax.set_ylabel("Absorbance", fontsize=11, fontweight='medium')
            
        elif self.current_plot_type == "Heatmap":
            max_spectra = min(spectra.shape[0], 20)
            
            extent = [np.min(wavelengths), np.max(wavelengths), 0, max_spectra]
            
            im = self.ax.imshow(spectra[:max_spectra], aspect='auto', 
                              interpolation='nearest', cmap='viridis',
                              extent=extent)
                              
            self.ax.ticklabel_format(axis='x', useOffset=False, style='plain')
            
            self.ax.set_yticks(np.arange(0.5, max_spectra, 1))
            self.ax.set_yticklabels(np.arange(1, max_spectra+1, 1))
            
            self.ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
            self.ax.set_ylabel("Spectrum Index", fontsize=11, fontweight='medium')
            cbar = self.figure.colorbar(im, ax=self.ax)
            cbar.set_label("Absorbance", fontsize=10)
            
        elif self.current_plot_type == "3D Plot":
            self.ax.remove()
            self.ax = self.figure.add_subplot(111, projection='3d')
            
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
            
            self.ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
            self.ax.set_ylabel("Spectrum Index", fontsize=11, fontweight='medium')
            self.ax.set_zlabel("Absorbance", fontsize=11, fontweight='medium')
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
            
            self.ax.set_xlabel("Spectrum", fontsize=11, fontweight='medium')
            self.ax.set_ylabel("Average Absorbance", fontsize=11, fontweight='medium')
        
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        
        self.ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
        
        if self.current_plot_type in ["Line Plot", "Scatter Plot", "Spectrum Plot"]:
            self.ax.legend(loc='best', frameon=True, framealpha=0.7, fontsize='small')
            
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
            
            print(f"🔧 Successfully extracted numerical spectral data:")
            print(f"    Original spectrum: {original_spectra.shape}")
            print(f"    Processed spectrum: {processed_spectra.shape}")
            print(f"    Wavelength count: {len(wavelengths)}")
            
        except Exception as conversion_error:
            print(f"❌ Spectral data conversion failed: {conversion_error}")
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
            
        self.ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
        self.ax.set_ylabel("Absorbance", fontsize=11, fontweight='medium')
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        
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
        """Save current plot to file"""
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
            return False

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
        
        self.ax.scatter(x_data, y_data, alpha=0.7, c='#1f77b4', edgecolors='k', linewidths=0.5)
        
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
        if confusion_matrix is None:
            print("Error: Confusion matrix data is empty")
            self.ax.text(0.5, 0.5, "No confusion matrix data available", 
                       horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        print("🔍 Confusion matrix plotting debug info:")
        print(f"VisualizationWidget: Input class_names: {class_names}")
        print(f"Confusion matrix shape: {confusion_matrix.shape}")
        
        matrix_size = confusion_matrix.shape[0]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(matrix_size)]
            print("⚠️ No class names provided, using default indices")
        
        if len(class_names) != matrix_size:
            print(f"⚠️ Label count mismatch: {len(class_names)} vs {matrix_size}")
            
            if len(class_names) > matrix_size:
                class_names = class_names[:matrix_size]
                print(f"✂️ Truncate labels to {matrix_size} classes")
            else:
                original_count = len(class_names)
                for i in range(matrix_size - original_count):
                    class_names.append(f"Missing_{original_count + i}")
                print(f"➕ Supplement labels to {matrix_size} classes")
        
        cleaned_class_names = []
        for i, name in enumerate(class_names):
            str_name = str(name).strip()
            
            if len(str_name) > 15:
                str_name = str_name[:13] + ".."
            
            str_name = str_name.replace('\n', ' ').replace('\t', ' ')
            
            cleaned_class_names.append(str_name)
        
        class_names = cleaned_class_names
        print(f"🧹 Cleaned labels: {class_names}")
        
        if len(class_names) != matrix_size:
            print("❌ Final label count still mismatch, force repair")
            if len(class_names) > matrix_size:
                class_names = class_names[:matrix_size]
            else:
                while len(class_names) < matrix_size:
                    class_names.append(f"Fix_{len(class_names)}")
        
        print(f"✅ Final label list ({len(class_names)}): {class_names}")
        
        num_classes = matrix_size
        
        if num_classes <= 10:
            print("📊 Using small confusion matrix plotting")
            self._plot_small_confusion_matrix(confusion_matrix, class_names, title)
        elif num_classes <= 25:
            print("📊 Using medium confusion matrix plotting")
            self._plot_medium_confusion_matrix(confusion_matrix, class_names, title)
        else:
            print("📊 Using large confusion matrix plotting")
            self._plot_large_confusion_matrix(confusion_matrix, class_names, title)
        
        print(f"✅ VisualizationWidget: Confusion matrix plotting completed, classes: {num_classes}")
        
    def _plot_small_confusion_matrix(self, cm, class_names, title):
        """Plot small confusion matrix (≤10 classes)"""
        print(f"🎨 Plotting small confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.cm.Blues
        
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            print(f"⚠️ Color bar addition failed: {e}")
        
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
        
        self.canvas.draw()
        
    def _plot_medium_confusion_matrix(self, cm, class_names, title):
        """Plot medium confusion matrix (11-25 classes) - improved version"""
        print(f"🎨 Plotting medium confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.cm.Blues
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            print(f"⚠️ Color bar addition failed: {e}")
        
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
        
        plt.tight_layout()
        
        self.canvas.draw()
        
    def _plot_large_confusion_matrix(self, cm, class_names, title):
        """Plot large confusion matrix (>25 classes) - improved version"""
        print(f"🎨 Plotting large confusion matrix: {cm.shape}, labels: {len(class_names)}")
        
        cmap = plt.cm.Blues
        im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        try:
            cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count', rotation=270, labelpad=15)
        except Exception as e:
            print(f"⚠️ Color bar addition failed: {e}")
        
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
        
        plt.tight_layout()
        
        self.canvas.draw() 