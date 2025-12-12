# utils/visualization_window.py

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager

def safe_float_convert(val):
    """Safely convert value to float, return NaN if conversion fails"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan



def safe_convert_to_float(arr):
    """Safely convert array to float, return original array if contains non-numeric values"""
    result = []
    has_string_labels = False
    
    for item in arr:
        try:
            # **CRITICAL FIX: Check if item is string-based before conversion**
            str_item = str(item).strip()
            # Remove any quotes that might be present
            str_item = str_item.strip("'\"")
            if str_item and not str_item.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                # This is a string label like 'ClassA', 'ClassB', etc.
                has_string_labels = True
                break
            float_val = float(item)
            result.append(float_val)
        except (ValueError, TypeError):
            # If contains non-numeric labels (like string labels), return original array
            has_string_labels = True
            break
    
    # If contains string labels, return original array instead of None or converted array
    if has_string_labels:
        return np.array(arr)  # Return original array to keep string labels
    else:
        return np.array(result) if result else None



class VisualizationWindow(QMainWindow):
    def __init__(self, title="Visualization"):
        super(VisualizationWindow, self).__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout(self.main_widget)

        # Add a label for messages
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            padding: 8px; 
            background-color: #fffbe6; 
            border: 1px solid #ffe58f; 
            border-radius: 4px;
            font-size: 9pt;
        """)
        self.message_label.hide() # Hide by default
        self.layout.addWidget(self.message_label)

        # Matplotlib Figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Save button
        self.save_button = QPushButton("Save Image")
        self.layout.addWidget(self.save_button)

        # Connect save button signal
        self.save_button.clicked.connect(self.save_image)

    def set_message(self, text):
        """Set a message to display at the top of the window."""
        if text:
            self.message_label.setText(text)
            self.message_label.show()
        else:
            self.message_label.hide()

    def save_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
                                                   options=options)
        if file_path:
            self.figure.savefig(file_path)

    def plot(self, data, title="Plot"):
        # Clear previous plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data)
        ax.set_title(title)
        self.canvas.draw()
        
    def plot_feature_importance(self, importances, feature_names, title="Feature Importance"):
        """Plot feature importance visualization"""
        # Clear previous plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if importances is None or feature_names is None:
            ax.text(0.5, 0.5, "No feature importance data available", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        # Ensure numpy is imported
        import numpy as np
        
        # Unify conversion to list - ensure correct handling of all input formats
        # Check importances format
        if isinstance(importances, np.ndarray):
            importances = importances.tolist()
        elif isinstance(importances, (tuple, list)) and len(importances) > 0 and isinstance(importances[0], (tuple, list)):
            # If importance is a tuple/list of lists (compound structure), extract values
            # Detected compound structure for importance, extracting values
            try:
                importances = [float(item[1]) if isinstance(item, (tuple, list)) and len(item) > 1 else 0 
                             for item in importances]
            except Exception as e:
                # Failed to extract importance values, using default values
                importances = [1.0/len(importances)] * len(importances)
        
        # Check feature_names format
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        elif isinstance(feature_names, (tuple, list)) and len(feature_names) > 0 and isinstance(feature_names[0], (tuple, list)):
            # If feature names are tuple/list of lists (compound structure), extract names
            # Detected compound structure for feature names, extracting names
            feature_names = [str(item[0]) if isinstance(item, (tuple, list)) and len(item) > 0 else f"Feature_{i}" 
                           for i, item in enumerate(feature_names)]
        
        # Ensure all feature names are string type
        feature_names = [str(name) for name in feature_names]
            
        if len(importances) == 0 or len(feature_names) == 0:
            ax.text(0.5, 0.5, "No features were selected", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        # Debug feature importance values (only in development mode)
        # print(f"Feature importance values: {importances[:10]}")
        # print(f"Feature names: {feature_names[:10]}")
        
        # Check if importance values are all zero or close to zero
        all_zeros = all(abs(imp) < 1e-10 for imp in importances)
        if all_zeros:
            # Warning: All feature importance values are close to zero
            # Prompt user to check data preprocessing
            ax.text(0.5, 0.5, "All feature importance values are zero\nPlease check:\n1. Is data preprocessing correct\n2. Does task type match\n3. Data quality", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            self.canvas.draw()
            return
            
        # Ensure data lengths are consistent
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        feature_names = feature_names[:n]
        
        try:
            # Convert to numpy array to use argsort
            importances_array = np.array([float(imp) for imp in importances])
            
            # If data contains NaN or infinite values, replace them
            if np.isnan(importances_array).any() or np.isinf(importances_array).any():
                # Warning: Feature importance contains NaN or infinite values, replacing with zero
                importances_array = np.nan_to_num(importances_array, nan=0.0, posinf=0.0, neginf=0.0)
                importances = importances_array.tolist()
            
            # Sort by importance
            sorted_idx = importances_array.argsort()[::-1]  # Descending sort indices
            
            # Handle special case: if all importance values are equal
            if len(set(importances)) == 1:
                # Use original order
                position = range(len(importances))
                colors = ['#1f77b4'] * len(importances)  # Use same color
            else:
                # Use sorted order
                position = range(len(sorted_idx))
                importances = [importances[i] for i in sorted_idx]
                feature_names = [feature_names[i] for i in sorted_idx]
                # Generate gradient colors
                import matplotlib.cm as cm
                max_importance = max(importances) if max(importances) > 0 else 1.0
                colors = cm.viridis(np.array(importances) / max_importance)
            
            # Create horizontal bar chart
            bars = ax.barh(position, importances, align='center', color=colors, alpha=0.8)
            
            # Set Y-axis labels as feature names
            ax.set_yticks(position)
            ax.set_yticklabels(feature_names)
            
            # Set title and labels
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            # Add grid lines for better readability
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Optimize figure layout
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            import traceback
            print(f"Failed to plot feature importance chart: {e}")
            traceback.print_exc()
            # Display error message
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error plotting feature importance: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center', color='red')
            self.canvas.draw()
        
    def plot_confusion_matrix(self, confusion_matrix, class_names=None, title="Confusion Matrix"):
        """Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Chart title
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if confusion_matrix is None:
            ax.text(0.5, 0.5, "No confusion matrix data available", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        # If no class names provided, use numeric indices
        if class_names is None:
            class_names = [str(i) for i in range(confusion_matrix.shape[0])]
            
        # Use better color map
        cmap = plt.cm.Blues
        
        # Create normalized version for side-by-side display
        if confusion_matrix.shape[0] <= 20:  # Only for reasonable size matrices
            # Create a subplot for counts (original)
            ax_counts = self.figure.add_subplot(121)
            ax_norm = self.figure.add_subplot(122)
            
            # Plot original counts
            im_counts = ax_counts.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
            cbar_counts = self.figure.colorbar(im_counts, ax=ax_counts)
            cbar_counts.set_label('Counts', rotation=270, labelpad=15)
            
            # Plot normalized version
            if np.sum(confusion_matrix) > 0:  # Avoid division by zero
                norm_cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
                im_norm = ax_norm.imshow(norm_cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
                cbar_norm = self.figure.colorbar(im_norm, ax=ax_norm)
                cbar_norm.set_label('Normalized', rotation=270, labelpad=15)
            else:
                ax_norm.text(0.5, 0.5, "Cannot normalize empty matrix", 
                           horizontalalignment='center', verticalalignment='center')
                
            # Set ticks and labels for both subplots
            for subplot_ax, subplot_title in [(ax_counts, "Counts"), (ax_norm, "Normalized")]:
                # Set axis ticks
                tick_marks = np.arange(len(class_names))
                subplot_ax.set_xticks(tick_marks)
                subplot_ax.set_xticklabels(class_names, rotation=45, ha='right')
                subplot_ax.set_yticks(tick_marks)
                subplot_ax.set_yticklabels(class_names)
                
                # Set title and labels
                subplot_ax.set_title(subplot_title, fontsize=12)
                subplot_ax.set_ylabel('True', fontsize=10)
                subplot_ax.set_xlabel('Predicted', fontsize=10)
                
                # Add text annotations only for count matrix and if matrix is reasonably sized
                if subplot_title == "Counts" and confusion_matrix.shape[0] <= 10:
                    fmt = 'd'  # 'd' for integer format
                    thresh = confusion_matrix.max() / 2.
                    for i in range(confusion_matrix.shape[0]):
                        for j in range(confusion_matrix.shape[1]):
                            subplot_ax.text(j, i, format(confusion_matrix[i, j], fmt),
                                         ha="center", va="center",
                                         color="white" if confusion_matrix[i, j] > thresh else "black")
                                         
            # Set the main title for the whole figure
            self.figure.suptitle(title, fontsize=14, fontweight='bold')
            
        else:
            # For very large matrices, just show the original without annotations
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label('Count', rotation=270, labelpad=15)
            
            # Set title and labels
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('True', fontsize=12)
            ax.set_xlabel('Predicted', fontsize=12)
            
            # For large matrices, don't show all class labels
            if len(class_names) > 20:
                # Pick evenly spaced indices
                indices = np.linspace(0, len(class_names)-1, 20, dtype=int)
                ax.set_xticks(indices)
                ax.set_xticklabels([class_names[i] for i in indices], rotation=45, ha='right')
                ax.set_yticks(indices)
                ax.set_yticklabels([class_names[i] for i in indices])
            else:
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_yticklabels(class_names)
        
        # Optimize layout
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_scatter(self, x_data, y_data, title="Scatter Plot", xlabel="X", ylabel="Y"):
        """Plot scatter plot, mainly used for regression model's actual vs predicted values
        
        Args:
            x_data: X-axis data (usually actual values)
            y_data: Y-axis data (usually predicted values)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if x_data is None or y_data is None:
            ax.text(0.5, 0.5, "No data available for scatter plot", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        
        # Convert data types for compatibility
        x_data = np.array(x_data).flatten()
        y_data = np.array(y_data).flatten()
        
        # Print debugging information
        print(f"Scatter plot data point count: {len(x_data)}")
        print(f"x_data range: [{np.min(x_data)}, {np.max(x_data)}]")
        print(f"y_data range: [{np.min(y_data)}, {np.max(y_data)}]")
            
        # Plot scatter plot, increase point size and adjust opacity
        ax.scatter(x_data, y_data, s=50, alpha=0.8, c='#1f77b4', edgecolors='k', linewidths=0.5)
        
        # Add diagonal line (ideal case where prediction = actual)
        min_val = min(np.min(x_data), np.min(y_data))
        max_val = max(np.max(x_data), np.max(y_data))
        
        # Add some padding to make the plot more aesthetic
        padding = (max_val - min_val) * 0.05
        plot_min = min_val - padding
        plot_max = max_val + padding
        
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5)
        
        # Ensure axis range includes all data points and has padding
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Set title and axis labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Add grid lines for easier reading
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set equal axis to make the plot more square
        ax.set_aspect('equal', adjustable='box')
        
        # Adjust layout and draw
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_spectra(self, wavelengths, spectra, title="Spectrum Data", labels=None):
        """Plot spectral data, selectively displaying spectrum lines based on label types
        
        Args:
            wavelengths: Wavelength data array
            spectra: Spectral data array
            title: Chart title
            labels: Sample label array, used to select displayed spectra based on categories
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # **CRITICAL FIX: Safe label preprocessing to prevent ClassC conversion errors**
        if labels is not None:
            try:
                # Ensure labels are always treated as strings to prevent any float conversion
                labels = np.array([str(label) for label in labels], dtype='<U50')
                # print(f"🔧 Utils Points Labels safely converted to string array: {labels[:3] if len(labels) > 0 else []}")
            except Exception as label_error:
                # print(f"⚠️ Utils Points Label conversion warning: {label_error}")
                labels = np.array([f"Sample_{i}" for i in range(len(spectra))], dtype='<U50')
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Ensure wavelength and spectral data match
        if wavelengths is not None and spectra is not None:
            # Check data types and dimensions
            wavelengths = np.array(wavelengths, dtype=float)
            spectra = np.array(spectra, dtype=float)
            
            # Handle one-dimensional spectral data
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
                
            # Check and fix dimension mismatch issues
            if len(wavelengths) != spectra.shape[1]:
                if len(wavelengths) > spectra.shape[1]:
                    # Wavelength array is longer, truncate
                    wavelengths = wavelengths[:spectra.shape[1]]
                else:
                    # Spectral data has more columns, truncate
                    spectra = spectra[:, :len(wavelengths)]
        
        # More elegant color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Ensure data is valid
        if wavelengths is None or spectra is None or len(wavelengths) == 0 or spectra.shape[0] == 0:
            ax.text(0.5, 0.5, "No valid spectral data", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        # Ensure wavelengths are sorted - this is important for spectral plots
        if not np.all(np.diff(wavelengths) > 0):
            # If wavelengths are not increasing, sort them
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            spectra = spectra[:, sort_idx]
        
        # Select sample indices to plot
        selected_indices = []
        
        # Detect label type
        try:
            # Check if labels are numeric or categorical
            if labels is not None and len(labels) > 0:
                # Try to detect label type more robustly
                sample_labels = labels[:min(10, len(labels))]
                
                # Check for non-numeric content
                try:
                    # Try converting to float
                    numeric_labels = [float(label) for label in sample_labels]
                    
                    # Check for specific problematic values like "Verde"
                    sample_values = [str(label) for label in sample_labels]
                    if any('verde' in str(val).lower() or 'albari' in str(val).lower() for val in sample_values):
                        # print(f"🚨 Detected Verde-like labels in numeric_labels: {sample_values}")
                        is_categorical = True
                    else:
                        # Check if values suggest categorical vs continuous
                        unique_labels = np.unique(labels)
                        is_categorical = len(unique_labels) <= max(10, len(labels) * 0.1)
                except (ValueError, TypeError) as math_error:
                    # print(f"❌ Mathematical operation failed on labels (Verde conversion error): {math_error}")
                    is_categorical = True
                
                # print(f"Label classification detection: Unique values = {len(unique_labels)}, Sample count = {len(labels)}")
                # print(f"Unique label values: {unique_labels}")
                # print(f"Detected as: {'Categorical labels' if is_categorical else 'Continuous values'}")
                
            else:
                is_categorical = True
        except Exception as e:
            # print(f"Label classification detection: Exception occurred ({e}), determined as categorical")
            is_categorical = True
        
        if is_categorical and labels is not None:
            # print(f"Plotting by category, {len(unique_labels)} categories total")
            # Plot by category
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = (labels == label)
                if np.any(mask):
                    selected_idx = np.where(mask)[0]
                    # print(f"  Category {label}: Selected sample {selected_idx}")
                    for idx in selected_idx[:1]:  # Only plot first sample per category
                        ax.plot(wavelengths, spectra[idx], color=colors[i], label=f'{label}', alpha=0.8)
        else:
            # print(f"Plotting first {min(10, spectra.shape[0])} spectra")
            # Plot first few spectra
            colors = plt.cm.viridis(np.linspace(0, 1, min(10, spectra.shape[0])))
            for i in range(min(10, spectra.shape[0])):
                label = f'Sample {i+1}' if labels is None else f'{labels[i]}'
                ax.plot(wavelengths, spectra[i], color=colors[i], label=label, alpha=0.7)
        
        # Set axis labels and title
        ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight='medium')
        ax.set_ylabel("Absorbance", fontsize=11, fontweight='medium')
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Add grid lines
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add legend
        ax.legend(loc='best', frameon=True, framealpha=0.7, fontsize='small')
        
        # Set axis ranges
        if wavelengths is not None and len(wavelengths) > 0:
            x_min, x_max = np.min(wavelengths), np.max(wavelengths)
            ax.set_xlim(x_min, x_max)
            
            if spectra is not None and spectra.size > 0:
                y_min, y_max = np.min(spectra), np.max(spectra)
                y_padding = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Optimize figure layout
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_spectra_with_points(self, wavelengths, spectra, title="Raw Spectrum Data", labels=None, show_points=True):
        """Plot spectral data, showing original data points to prove no smoothing processing
        
        Args:
            wavelengths: Wavelength data array
            spectra: Spectral data array  
            title: Chart title
            labels: Sample label array, used to select displayed spectra based on categories
            show_points: Whether to show original data points
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # **CRITICAL FIX: Safe label preprocessing to prevent ClassC conversion errors**
        if labels is not None:
            try:
                # Ensure labels are always treated as strings to prevent any float conversion
                labels = np.array([str(label) for label in labels], dtype='<U50')
                print(f"🔧 Utils Labels safely converted to string array: {labels[:3] if len(labels) > 0 else []}")
            except Exception as label_error:
                print(f"⚠️ Utils Label conversion warning: {label_error}")
                labels = None
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Ensure wavelength and spectral data match
        if wavelengths is not None and spectra is not None:
            # Check data types and dimensions
            wavelengths = np.array(wavelengths, dtype=float)
            spectra = np.array(spectra, dtype=float)
            
            # Handle one-dimensional spectral data
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
                
            # Check and fix dimension mismatch issues
            if len(wavelengths) != spectra.shape[1]:
                if len(wavelengths) > spectra.shape[1]:
                    # Wavelength array is longer, truncate
                    wavelengths = wavelengths[:spectra.shape[1]]
                else:
                    # Spectral data has more columns, truncate
                    spectra = spectra[:, :len(wavelengths)]
        
        # More elegant color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Ensure data is valid
        if wavelengths is None or spectra is None or len(wavelengths) == 0 or spectra.shape[0] == 0:
            ax.text(0.5, 0.5, "No valid spectral data", 
                   horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
            
        # Ensure wavelengths are sorted - this is important for spectral plots
        if not np.all(np.diff(wavelengths) > 0):
            # If wavelengths are not increasing, sort them
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            spectra = spectra[:, sort_idx]
        
        # Select sample indices to plot
        selected_indices = []
        
        # Determine selection method based on label type
        if labels is not None:
            labels = np.array(labels).flatten()
            unique_labels = np.unique(labels)
            
            # Enhanced detection method to determine if labels are categorical
            try:
                # Try to convert labels to numeric type
                numeric_labels = safe_convert_to_float(labels)
                
                # If numeric_labels is string array (non-numeric), directly judge as classification labels
                if numeric_labels is not None and numeric_labels.dtype.kind in ['U', 'S', 'O']:
                    is_categorical = True
                    print(f"Label classification detection: Non-numeric labels detected, determined as categorical")
                elif numeric_labels is not None:
                    # **CRITICAL FIX: Ensure numeric_labels is truly numeric before mathematical operations**
                    try:
                        # Double-check that numeric_labels is actually numeric
                        if numeric_labels.dtype.kind in ['U', 'S', 'O']:
                            # Still string type, treat as categorical
                            is_categorical = True
                        else:
                            # **ADDITIONAL FIX: Check for Verde and other string labels before math operations**
                            sample_values = [str(val) for val in numeric_labels[:5]]
                            has_string_like_verde = any('Verde' in str(val) or 'Class' in str(val) or 
                                                       not str(val).replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                       for val in sample_values if str(val).strip())
                            
                            if has_string_like_verde:
                                print(f"🚨 Detected Verde-like labels in numeric_labels: {sample_values}")
                                is_categorical = True
                            else:
                                # Numeric label classification judgment - safe to do math operations
                                is_categorical = (len(unique_labels) <= 10 or  # Few categories
                                                 len(unique_labels) < len(labels) * 0.3 or  # Much fewer categories than samples
                                                 np.sum(np.equal(numeric_labels, np.round(numeric_labels))) > len(labels) * 0.8)  # Mostly integer values
                    except Exception as math_error:
                        print(f"❌ Mathematical operation failed on labels (Verde conversion error): {math_error}")
                        is_categorical = True
                    
                    # Print classification detection details for debugging
                    print(f"Label classification detection: Unique values = {len(unique_labels)}, Sample count = {len(labels)}")
                    print(f"Unique label values: {unique_labels}")
                    print(f"Detected as: {'Categorical labels' if is_categorical else 'Continuous values'}")
                else:
                    # Conversion failed, judge as classification labels
                    is_categorical = True
                    print(f"Label classification detection: Conversion failed, determined as categorical")
            except Exception as e:
                # Cannot convert to numeric, definitely non-numeric categorical labels
                is_categorical = True
                print(f"Label classification detection: Exception occurred ({e}), determined as categorical")
            
            if is_categorical:
                # Categorical labels: select first spectrum per class (ensure consistency)
                print(f"Plotting by category, {len(unique_labels)} categories total")
                for label in unique_labels:
                    label_indices = np.where(labels == label)[0]
                    if len(label_indices) > 0:
                        # Select first sample from each category (ensure consistency)
                        selected_idx = label_indices[0]
                        selected_indices.append(selected_idx)
                        print(f"  Category {label}: Selected sample {selected_idx}")
            else:
                # Continuous value labels or too many label categories: select first samples (ensure consistency)
                print(f"Plotting first {min(10, spectra.shape[0])} spectra")
                selected_indices = list(range(min(10, spectra.shape[0])))
        else:
            # No labels: select first samples (ensure consistency)
            is_categorical = False
            print("No labels, selecting first spectra")
            selected_indices = list(range(min(10, spectra.shape[0])))
        
        # Ensure selected_indices is not empty
        if len(selected_indices) == 0:
            selected_indices = list(range(min(10, spectra.shape[0])))
        
        # Plot spectral lines
        for i, idx in enumerate(selected_indices):
            # Set label
            if labels is not None and is_categorical:
                # Categorical values: directly display label value
                label = f"{labels[idx]}"
            else:
                # Continuous values: use simple sequence number
                label = f"Spectrum {i+1}"
            
            # Plot spectral lines - ensure direct connection of original data points, no smoothing or interpolation
            ax.plot(wavelengths, spectra[idx], 
                   linewidth=2.0, 
                   alpha=0.7, 
                   color=colors[i % len(colors)],
                   label=label,
                   linestyle='-',
                   antialiased=False)  # Turn off antialiasing to show most original lines
            
            # 2. If enabled, overlay original data points
            if show_points:
                # For performance, downsample if too many data points
                if len(wavelengths) > 200:
                    step = len(wavelengths) // 200
                    point_wavelengths = wavelengths[::step]
                    point_spectra = spectra[idx, ::step]
                else:
                    point_wavelengths = wavelengths
                    point_spectra = spectra[idx]
                
                ax.scatter(point_wavelengths, point_spectra, 
                          s=12, alpha=0.8, color=colors[i % len(colors)], 
                          marker='o', edgecolors='white', linewidths=0.5,
                          zorder=5)  # Ensure points are above lines
        
        # Set axis labels and title
        ax.set_xlabel("Wavelength (nm)", fontsize=12, fontweight='medium')
        ax.set_ylabel("Absorbance", fontsize=12, fontweight='medium')
        
        if show_points:
            ax.set_title(f"{title} (Lines + Raw Data Points)", fontsize=13, fontweight='bold')
        else:
            ax.set_title(f"{title} (Direct Point-to-Point Connection)", fontsize=13, fontweight='bold')
        
        # Add explanatory text
        info_text = f"Data points: {len(wavelengths)}"
        if show_points and len(wavelengths) > 200:
            info_text += f" (showing every {len(wavelengths) // 200}th point)"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Add grid lines
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Add legend with responsive font size
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
        legend_fontsize = font_sizes.get('legend.fontsize', 11)
        ax.legend(loc='best', frameon=True, framealpha=0.8, fontsize=legend_fontsize)
        
        # Set axis ranges
        if wavelengths is not None and len(wavelengths) > 0:
            x_min, x_max = np.min(wavelengths), np.max(wavelengths)
            ax.set_xlim(x_min, x_max)
            
            if spectra is not None and spectra.size > 0:
                y_min, y_max = np.min(spectra), np.max(spectra)
                y_padding = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Optimize figure layout
        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear the current figure"""
        self.figure.clear()
        self.canvas.draw()
