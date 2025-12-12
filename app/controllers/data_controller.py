from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox
# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
from app.utils.exceptions import DataServiceError
from app.utils.logger import setup_logger
from app.services.data_service import DataService
from app.models.data_model import DataModel
from app.utils.label_processor import LabelProcessor, EnhancedLabelProcessor
import numpy as np
import os
import pandas as pd
import time

class DataController:
    def __init__(self, view, translator, data_partitioning_plugins):
        self.view = view
        self.translator = translator
        self.data_service = DataService(data_partitioning_plugins)
        self.data_model = DataModel()
        self.label_processor = LabelProcessor()
        self.logger = setup_logger()
        self.current_spectral_config = None  # Store current spectral configuration

    def load_data(self):
        try:
            # NEW: First show spectral type pre-selection dialog
            from app.views.spectral_type_pre_selection_dialog import SpectralTypePreSelectionDialog
            
            # Show spectral type selection dialog
            type_dialog = SpectralTypePreSelectionDialog(self.view)
            if type_dialog.exec_() != QDialog.Accepted:
                return False  # User cancelled
            
            selected_type = type_dialog.get_selected_type()
            type_name = type_dialog.get_type_name()
            
            # Now show file selection dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self.view, f"Select {type_name} Data File", "",
                "CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*.*)"
            )
            
            if file_path:
                try:
                    # **NEW: Load data with NaN/Inf handling**
                    raw_data = self.data_service.load_data(file_path)
                    
                    # Check for data quality issues and handle them
                    from app.views.nan_inf_handling_dialog import detect_and_handle_data_issues
                    cleaned_data, cancelled = detect_and_handle_data_issues(raw_data, self.view)
                    
                    if cancelled:
                        return False  # User cancelled data cleaning
                    
                    # Use cleaned data
                    data = cleaned_data
                    self.data_model.set_data(data)
                    self.view.update_data_view(data)
                    
                    # NEW: Apply spectral configuration based on pre-selected type
                    self.apply_preselected_spectral_config(data, selected_type, type_name)
                    
                    return True
                except DataServiceError as e:
                    self.view.display_error(str(e))
                    return False
            else:
                return False
        except Exception as e:
            raise
    
    def apply_preselected_spectral_config(self, data, selected_type, type_name):
        """Apply spectral configuration based on pre-selected type"""
        try:
            wavelengths = self.extract_wavelengths(data)
            
            if selected_type == "auto":
                # Use auto-detection dialog
                from app.views.spectral_type_selection_dialog import SpectralTypeSelectionDialog
                dialog = SpectralTypeSelectionDialog(data, wavelengths, self.view)
                
                if dialog.exec_() == QDialog.Accepted:
                    config = dialog.get_selected_config()
                    self.apply_spectral_configuration(config)
                    self.show_configuration_info(config)
                else:
                    self.use_default_configuration()
            else:
                # Apply configuration for pre-selected type
                config = self.generate_config_for_preselected_type(selected_type, data, wavelengths)
                self.apply_spectral_configuration(config)
                
                # Show simplified confirmation
                QMessageBox.information(self.view, "Spectral Configuration", 
                    f"Data loaded successfully!\n\nSpectral Type: {type_name}\n"
                    f"Samples: {len(data)}\n"
                    f"Features: {len(data.columns)}\n\n"
                    f"Optimized preprocessing parameters have been applied automatically.")
                
        except Exception as e:
            self.logger.error(f"Failed to apply preselected spectral config: {e}")
            QMessageBox.warning(self.view, "Warning", 
                              f"Data loaded successfully, but spectral configuration failed.\n"
                              f"Using default settings.\n\nError: {str(e)}")
            self.use_default_configuration()
    
    def generate_config_for_preselected_type(self, spectral_type, data, wavelengths):
        """Generate configuration for pre-selected spectral type"""
        # Basic configuration based on spectral type
        configs = {
            "nir": {
                "spectral_type": "nir",
                "adaptation_level": "basic",
                "preprocessing_config": {
                    "Standard Normal Variate (SNV)": {"enabled": True},
                    "Baseline Correction": {"enabled": True, "method": "als", "lambda": 1e5, "p": 0.01},
                    "Savitzky-Golay Filter": {"enabled": True, "window_length": 15, "polyorder": 2},
                    "Second Derivative": {"enabled": True},
                    "Vector Normalization": {"enabled": True}
                },
                "parameter_adaptations": {
                    "als_baseline": {"lambda": 1e5, "p": 0.01},
                    "savgol_filter": {"window_length": 15, "polyorder": 2}
                }
            },
            "raman": {
                "spectral_type": "raman", 
                "adaptation_level": "basic",
                "preprocessing_config": {
                    "Despiking": {"enabled": True, "method": "mad", "threshold": 2.0},
                    "Peak Alignment": {"enabled": True, "method": "dtw", "reference": "mean"},
                    "Baseline Correction": {"enabled": True, "method": "als", "lambda": 1e6, "p": 0.001},
                    "Savitzky-Golay Filter": {"enabled": True, "window_length": 7, "polyorder": 2},
                    "Area Normalization": {"enabled": True}
                },
                "parameter_adaptations": {
                    "als_baseline": {"lambda": 1e6, "p": 0.001},
                    "despiking": {"threshold": 2.0},
                    "peak_alignment": {"method": "dtw", "reference": "mean"}
                }
            },
            "mir": {
                "spectral_type": "mir",
                "adaptation_level": "basic", 
                "preprocessing_config": {
                    "Multiplicative Scatter Correction (MSC)": {"enabled": True},
                    "Baseline Correction": {"enabled": True, "method": "als", "lambda": 1e4, "p": 0.05},
                    "First Derivative": {"enabled": True},  # MIR requires derivative for feature enhancement
                    "Savitzky-Golay Filter": {"enabled": True, "window_length": 9, "polyorder": 2},
                    "Vector Normalization": {"enabled": True}
                },
                "parameter_adaptations": {
                    "als_baseline": {"lambda": 1e4, "p": 0.05}
                }
            },
            "vis_nir": {
                "spectral_type": "vis_nir",
                "adaptation_level": "basic",
                "preprocessing_config": {
                    "Standard Normal Variate (SNV)": {"enabled": True},  # Add scatter correction
                    "Baseline Correction": {"enabled": True, "method": "als", "lambda": 1e3, "p": 0.1},  # Use ALS instead
                    "Savitzky-Golay Filter": {"enabled": True, "window_length": 11, "polyorder": 2},
                    "Min-Max Scale": {"enabled": True}
                },
                "parameter_adaptations": {
                    "als_baseline": {"lambda": 1e3, "p": 0.1}
                }
            },
            "uv_vis": {
                "spectral_type": "uv_vis",
                "adaptation_level": "basic",
                "preprocessing_config": {
                    "Baseline Correction": {"enabled": True, "method": "polynomial", "order": 1},
                    "Min-Max Scale": {"enabled": True}
                },
                "parameter_adaptations": {}
            }
        }
        
        return configs.get(spectral_type, configs["nir"])
    
    def open_spectral_type_config(self, data):
        """Open spectral type configuration dialog"""
        try:
            from app.views.spectral_type_selection_dialog import SpectralTypeSelectionDialog
            
            # Try to extract wavelength information
            wavelengths = self.extract_wavelengths(data)
            
            dialog = SpectralTypeSelectionDialog(data, wavelengths, self.view)
            
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_selected_config()
                self.apply_spectral_configuration(config)
                self.show_configuration_info(config)
            else:
                # User cancelled, use default configuration
                self.use_default_configuration()
                
        except Exception as e:
            self.logger.error(f"Spectral configuration failed: {e}")
            QMessageBox.information(self.view, "Notice", 
                                  "Spectral configuration encountered an issue, using default configuration")
    
    def extract_wavelengths(self, data):
        """Extract wavelength information from data"""
        try:
            # Assume column names contain wavelength information
            columns = data.columns
            wavelengths = []
            
            for col in columns:
                # Try to extract numerical values from column names
                import re
                numbers = re.findall(r'\d+\.?\d*', str(col))
                if numbers:
                    try:
                        wl = float(numbers[0])
                        wavelengths.append(wl)
                    except:
                        continue
            
            if len(wavelengths) > 10:  # Need at least 10 wavelength points
                return np.array(wavelengths)
            else:
                # Generate default wavelengths (NIR range)
                return np.linspace(1000, 2500, len(columns))
                
        except Exception as e:
            self.logger.warning(f"Wavelength extraction failed: {e}")
            # Return default NIR wavelength range
            return np.linspace(1000, 2500, len(data.columns))
    
    def apply_spectral_configuration(self, config):
        """Apply spectral configuration to system"""
        self.current_spectral_config = config
        self.logger.info(f"Applying spectral configuration: {config.get('spectral_type', 'unknown')}")
        
        # Notify other components to update configuration
        if hasattr(self.view, 'update_spectral_config'):
            self.view.update_spectral_config(config)
    
    def show_configuration_info(self, config):
        """Display configuration information"""
        spectral_type = config.get('spectral_type', 'unknown')
        adaptation_level = config.get('adaptation_level', 'basic')
        
        message = f"""Spectral configuration applied:
        
Spectral Type: {spectral_type.upper()}
Adaptation Level: {adaptation_level.title()}
Preprocessing Steps: {len(config.get('preprocessing_config', {}))} steps

Configuration saved and will be used for subsequent preprocessing and modeling analysis."""
        
        QMessageBox.information(self.view, "Spectral Configuration", message)
    
    def use_default_configuration(self):
        """Use default configuration"""
        default_config = {
            "spectral_type": "nir",
            "adaptation_level": "basic",
            "preprocessing_config": {},
            "parameter_adaptations": {}
        }
        self.current_spectral_config = default_config
        self.logger.info("Using default spectral configuration")
    
    def get_current_spectral_config(self):
        """Get current spectral configuration"""
        return self.current_spectral_config
    
    def has_data(self):
        """Check if data exists"""
        return self.data_model.data is not None
    
    def get_current_data(self):
        """Get current data"""
        return self.data_model.data
    
    def get_wavelengths(self):
        """Get wavelength information"""
        if self.data_model.data is not None:
            return self.extract_wavelengths(self.data_model.data)
        return None

    def apply_data_partitioning(self, method=None, params=None):
        data = self.data_model.data
        
        if params is None:
            params = {}
            
        task_type = params.get('task_type', None)
        
        if data is None:
            self.view.display_message("No data loaded.", "Error")
            return

        try:
            if method is None:
                method = "Train-Test Split"
                
            file_path = getattr(self.data_model, 'file_path', None)
            
            # Process target column
            target_column = params.get('target_column', 0)
            if isinstance(target_column, str) and target_column.isdigit():
                target_column = int(target_column)
            elif target_column == 'label':
                target_column = 0
            
            if isinstance(target_column, int) and 0 <= target_column < data.shape[1]:
                y_raw = data.iloc[:, target_column]
            else:
                y_raw = data.iloc[:, 0]
            
            # Apply data partitioning
            self.label_processor = EnhancedLabelProcessor()
            
            if task_type is None:
                detected_task_type = self.label_processor.detect_task_type(y_raw)
                
                if params.get('force_classification', False):
                    detected_task_type = 'classification'
                elif params.get('force_regression', False):
                    detected_task_type = 'regression'
                
                task_type = detected_task_type
                params['task_type'] = task_type
            
            try:
                processed_labels, label_metadata = self.label_processor.process_labels_smart(y_raw, task_type)
                # print(f"✅ Label processing completed:")  # Output disabled
                # print(f"   Task type: {label_metadata.get('task_type')}")  # Output disabled
                # print(f"   Auto-detected: {label_metadata.get('auto_detected')}")  # Output disabled
                
                if task_type == 'classification':
                    # print(f"   Number of classes: {label_metadata.get('num_classes')}")  # Output disabled
                    # print(f"   Class names: {label_metadata.get('class_names')}")  # Output disabled
                    pass  # Ensure if block has content
                else:
                    try:
                        min_val = label_metadata.get('min_value', 'N/A')
                        max_val = label_metadata.get('max_value', 'N/A')
                        if min_val != 'N/A' and max_val != 'N/A':
                            # print(f"   Value range: [{float(min_val):.3f}, {float(max_val):.3f}]")  # Output disabled
                            pass  # Ensure if block has content
                        else:
                            # print(f"   Value range: [{min_val}, {max_val}]")  # Output disabled
                            pass  # Ensure else block has content
                    except (ValueError, TypeError):
                        # print(f"   Value range: [{label_metadata.get('min_value', 'N/A')}, {label_metadata.get('max_value', 'N/A')}]")  # Output disabled
                        pass  # Ensure except block has content
                    # print(f"   Valid samples: {label_metadata.get('valid_samples', 'N/A')}")  # Output disabled
                
            except ValueError as label_error:
                # print(f"❌ Label processing error: {label_error}")  # Output disabled
                
                if "non-numeric values" in str(label_error) and task_type == 'regression':
                    # print("🔄 Attempting to switch to classification due to non-numeric labels...")  # Output disabled
                    try:
                        task_type = 'classification'
                        params['task_type'] = task_type
                        processed_labels, label_metadata = self.label_processor.process_labels_smart(y_raw, task_type)
                        # print("✅ Successfully switched to classification mode")  # Output disabled
                    except Exception as fallback_error:
                        # print(f"❌ Fallback to classification also failed: {fallback_error}")  # Output disabled
                        self.view.display_message(f"Label processing failed: {str(label_error)}", "Error")
                        return
                else:
                    self.view.display_message(f"Label processing failed: {str(label_error)}", "Error")
                    return
            
            params['label_metadata'] = label_metadata
            params['processed_labels'] = processed_labels
            params['original_labels'] = y_raw
            
            # print(f"📋 Data partitioning summary:")  # Output disabled
            # print(f"   Method: {method}")  # Output disabled
            # print(f"   Task type: {task_type}")  # Output disabled
            # print(f"   Data shape: {data.shape}")  # Output disabled
            # print(f"   Target column: {target_column}")  # Output disabled
            
            # REFACTORED CALL: Pass data and params dict which now contains all necessary info
            partitioned_data = self.data_service.partition_data(
                data=data, 
                method=method, 
                params=params
            )
            
            self.data_model.X_train = partitioned_data.get('X_train')
            self.data_model.X_test = partitioned_data.get('X_test')
            self.data_model.y_train = partitioned_data.get('y_train')
            self.data_model.y_test = partitioned_data.get('y_test')
            
            self.data_model.task_type = task_type
            self.data_model.label_metadata = label_metadata
            self.data_model.label_processor = self.label_processor
            
            # print(f"✅ Data partitioning completed successfully:")  # Output disabled
            # print(f"   Training set: {self.data_model.X_train.shape}")  # Output disabled
            # print(f"   Test set: {self.data_model.X_test.shape}")  # Output disabled
            # print(f"   Task type: {task_type}")  # Output disabled
            
            self.view.display_message(f"Data successfully partitioned using {method}. Task type: {task_type}", "Info")
            
            # print("🎯 Showing data partitioning results dialog...")  # Output disabled
            try:
                self._show_partitioning_results_dialog(partitioned_data, method)
                # print("✅ Data partitioning visualization displayed successfully")  # Output disabled
            except Exception as viz_error:
                # print(f"⚠️  Visualization display failed: {viz_error}")  # Output disabled
                self.view.display_message("Data partitioned successfully, but visualization display failed", "Warning")
            
            return partitioned_data
            
        except Exception as e:
            error_msg = f"Data partitioning failed: {str(e)}"
            # print(f"❌ {error_msg}")  # Output disabled
            self.view.display_message(error_msg, "Error")
            return None

    def _show_partitioning_results_dialog(self, partitioned_data, method):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QPushButton, QTableWidget, QTableWidgetItem, 
                                   QTabWidget, QWidget, QGroupBox,
                                   QFileDialog, QMessageBox)
        from PyQt5.QtCore import Qt
        import pandas as pd
        import numpy as np
        import os
        
        dialog = QDialog(self.view)
        dialog.setWindowTitle(f"Data Partitioning Results - {method}")
        dialog.setModal(False)
        
        X_train = partitioned_data.get('X_train')
        X_test = partitioned_data.get('X_test')
        y_train = partitioned_data.get('y_train')
        y_test = partitioned_data.get('y_test')
        
        task_type = getattr(self.data_model, 'task_type', 'classification')

        # This block prepares y_train_original and y_test_original for display,
        # mapping numeric classification labels back to their string representation.
        label_mapping = partitioned_data.get('label_mapping')
        y_train_original = y_train
        y_test_original = y_test
        if y_train is not None and task_type == 'classification':
            if label_mapping:
                index_to_label = label_mapping.get('index_to_label', {})
                if index_to_label:
                    try:
                        # Convert numeric labels back to original string labels for display
                        if isinstance(y_train, pd.Series):
                            y_train_original = y_train.map(lambda x: index_to_label.get(int(x), str(x)))
                        else:
                            y_train_original = np.array([index_to_label.get(int(label), str(label)) for label in y_train])
                        
                        if isinstance(y_test, pd.Series):
                            y_test_original = y_test.map(lambda x: index_to_label.get(int(x), str(x)))
                        else:
                            y_test_original = np.array([index_to_label.get(int(label), str(label)) for label in y_test])
                    except (ValueError, TypeError):
                        y_train_original = y_train
                        y_test_original = y_test

        # --- UI Sizing and Layout ---
        max_features_to_show = min(20, X_train.shape[1])
        optimal_width = max(1000, min(1200, 800 + max_features_to_show * 60))
        dialog.resize(optimal_width, 800)
        layout = QVBoxLayout(dialog)
        
        # --- Title ---
        title_label = QLabel("Data Partitioning Results")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # --- Partitioning Statistics ---
        stats_group = QGroupBox("Partitioning Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_text = f"""Method: {method}
Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features
Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features
Training ratio: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]):.1%}
Test ratio: {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]):.1%}
Total samples: {X_train.shape[0]+X_test.shape[0]}
Total features: {X_train.shape[1]}"""
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 11pt; background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; line-height: 1.6;")
        stats_label.setWordWrap(True)
        stats_layout.addWidget(stats_label)
        layout.addWidget(stats_group)
        
        # --- Conditional Distribution Display ---
        if y_train is not None and y_test is not None:
            if task_type == 'regression':
                dist_group = QGroupBox("Target Value Distribution (Regression)")
                dist_layout = QVBoxLayout(dist_group)
                
                stats_table = QTableWidget()
                stats_table.setRowCount(4)
                stats_table.setColumnCount(2)
                stats_table.setHorizontalHeaderLabels(["Training Set", "Test Set"])
                stats_table.setVerticalHeaderLabels(["Mean", "Std Dev", "Min", "Max"])
                
                try:
                    train_stats = pd.Series(y_train).describe()
                    test_stats = pd.Series(y_test).describe()
                    
                    stats_data = {
                        "Mean": (train_stats.get('mean', float('nan')), test_stats.get('mean', float('nan'))),
                        "Std Dev": (train_stats.get('std', float('nan')), test_stats.get('std', float('nan'))),
                        "Min": (train_stats.get('min', float('nan')), test_stats.get('min', float('nan'))),
                        "Max": (train_stats.get('max', float('nan')), test_stats.get('max', float('nan'))),
                    }

                    for i, (stat_name, values) in enumerate(stats_data.items()):
                        stats_table.setVerticalHeaderItem(i, QTableWidgetItem(stat_name))
                        train_item = QTableWidgetItem(f"{values[0]:.4f}")
                        train_item.setFlags(train_item.flags() & ~Qt.ItemIsEditable)
                        stats_table.setItem(i, 0, train_item)
                        test_item = QTableWidgetItem(f"{values[1]:.4f}")
                        test_item.setFlags(test_item.flags() & ~Qt.ItemIsEditable)
                        stats_table.setItem(i, 1, test_item)
                except Exception as e:
                    error_item = QTableWidgetItem(f"Could not compute stats: {e}")
                    stats_table.setSpan(0, 0, 4, 2)
                    stats_table.setItem(0, 0, error_item)

                stats_table.resizeColumnsToContents()
                dist_layout.addWidget(stats_table)
                layout.addWidget(dist_group)

            else:  # Classification
                label_group = QGroupBox("Label Distribution (Classification)")
                label_layout = QVBoxLayout(label_group)
                
                # Use y_train_original which has the string labels
                if hasattr(y_train_original, 'value_counts'):
                    train_counts = y_train_original.value_counts()
                    test_counts = y_test_original.value_counts() if y_test_original is not None else pd.Series(dtype=int)
                else: # fallback for numpy arrays
                    unique_train, counts_train = np.unique(y_train_original, return_counts=True)
                    train_counts = pd.Series(counts_train, index=unique_train)
                    if y_test_original is not None:
                        unique_test, counts_test = np.unique(y_test_original, return_counts=True)
                        test_counts = pd.Series(counts_test, index=unique_test)
                    else:
                        test_counts = pd.Series(dtype=int)
                
                label_table = QTableWidget()
                all_labels = sorted(set(train_counts.index) | set(test_counts.index))
                label_table.setRowCount(len(all_labels))
                # THIS IS THE FIX: 4 columns
                label_table.setColumnCount(4)
                # THIS IS THE FIX: New header
                label_table.setHorizontalHeaderLabels(["Label", "Total", "Training Set", "Test Set"])
                
                for i, label in enumerate(all_labels):
                    train_count = train_counts.get(label, 0)
                    test_count = test_counts.get(label, 0)
                    total_count = train_count + test_count

                    # Column 0: Label
                    label_item = QTableWidgetItem(str(label))
                    label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
                    label_table.setItem(i, 0, label_item)
                    
                    # Column 1: Total
                    total_item = QTableWidgetItem(str(total_count))
                    total_item.setFlags(total_item.flags() & ~Qt.ItemIsEditable)
                    label_table.setItem(i, 1, total_item)

                    # Column 2: Training Set
                    train_item = QTableWidgetItem(str(train_count))
                    train_item.setFlags(train_item.flags() & ~Qt.ItemIsEditable)
                    label_table.setItem(i, 2, train_item)
                    
                    # Column 3: Test Set
                    test_item = QTableWidgetItem(str(test_count))
                    test_item.setFlags(test_item.flags() & ~Qt.ItemIsEditable)
                    label_table.setItem(i, 3, test_item)
                
                label_table.resizeColumnsToContents()
                label_layout.addWidget(label_table)
                layout.addWidget(label_group)
        
        # --- Data Preview Tabs ---
        tab_widget = QTabWidget()
        def create_preview_table(data, y_data_disp, max_features):
            table = QTableWidget()
            table.setRowCount(min(10, data.shape[0]))
            table.setColumnCount(max_features + (1 if y_data_disp is not None else 0))
            headers = [f"Feature{i+1}" for i in range(max_features)]
            if y_data_disp is not None: headers.append("Label")
            table.setHorizontalHeaderLabels(headers)
            for i in range(min(10, data.shape[0])):
                for j in range(max_features):
                    value = data.iloc[i, j] if hasattr(data, 'iloc') else data[i, j]
                    table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))
                if y_data_disp is not None:
                    y_value = y_data_disp.iloc[i] if hasattr(y_data_disp, 'iloc') else y_data_disp[i]
                    table.setItem(i, max_features, QTableWidgetItem(str(y_value)))
            return table

        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        train_layout.addWidget(create_preview_table(X_train, y_train_original, max_features_to_show))
        tab_widget.addTab(train_tab, f"Training Set Preview ({X_train.shape[0]} samples)")

        test_tab = QWidget()
        test_layout = QVBoxLayout(test_tab)
        test_layout.addWidget(create_preview_table(X_test, y_test_original, max_features_to_show))
        tab_widget.addTab(test_tab, f"Test Set Preview ({X_test.shape[0]} samples)")
        layout.addWidget(tab_widget)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        btn_style = "QPushButton { border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; color: white; }"
        save_train_btn = QPushButton("Save Training Set")
        save_train_btn.setStyleSheet(btn_style + "QPushButton { background-color: #3498db; } QPushButton:hover { background-color: #2980b9; }")
        save_train_btn.clicked.connect(lambda: self._save_partition_data(dialog, X_train, y_train, "Training Set"))
        button_layout.addWidget(save_train_btn)

        save_test_btn = QPushButton("Save Test Set")
        save_test_btn.setStyleSheet(btn_style + "QPushButton { background-color: #27ae60; } QPushButton:hover { background-color: #229954; }")
        save_test_btn.clicked.connect(lambda: self._save_partition_data(dialog, X_test, y_test, "Test Set"))
        button_layout.addWidget(save_test_btn)

        save_all_btn = QPushButton("Save All")
        save_all_btn.setStyleSheet(btn_style + "QPushButton { background-color: #e67e22; } QPushButton:hover { background-color: #d68910; }")
        save_all_btn.clicked.connect(lambda: self._save_all_partition_data(dialog, partitioned_data))
        button_layout.addWidget(save_all_btn)

        vis_btn = QPushButton("Visualize")
        vis_btn.setStyleSheet(btn_style + "QPushButton { background-color: #8e44ad; } QPushButton:hover { background-color: #7d3c98; }")
        vis_btn.clicked.connect(lambda: self._show_partition_visualization(partitioned_data))
        button_layout.addWidget(vis_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(btn_style + "QPushButton { background-color: #95a5a6; } QPushButton:hover { background-color: #7f8c8d; }")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.show()
        
    def _save_partition_data(self, parent, X_data, y_data, data_type):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        import os
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                parent, 
                f"Save {data_type}", 
                f"{data_type.lower().replace(' ', '_')}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                if hasattr(X_data, 'copy'):
                    data_to_save = X_data.copy()
                else:
                    data_to_save = pd.DataFrame(X_data)
                
                if y_data is not None:
                    if hasattr(y_data, 'name') and y_data.name:
                        label_col_name = y_data.name
                    else:
                        label_col_name = 'target'
                    
                    if hasattr(data_to_save, 'insert'):
                        data_to_save.insert(0, label_col_name, y_data)
                    else:
                        data_to_save = pd.DataFrame(data_to_save)
                        data_to_save.insert(0, label_col_name, y_data)
                
                if file_path.endswith('.xlsx'):
                    data_to_save.to_excel(file_path, index=False)
                else:
                    data_to_save.to_csv(file_path, index=False)
                
                QMessageBox.information(parent, "Save Successful", f"{data_type} has been saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(parent, "Save Failed", f"Error saving {data_type}:\n{str(e)}")
    
    def _save_all_partition_data(self, parent, partitioned_data):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        import os
        
        try:
            dir_path = QFileDialog.getExistingDirectory(parent, "Select Save Folder")
            
            if dir_path:
                X_train = partitioned_data.get('X_train')
                X_test = partitioned_data.get('X_test')
                y_train = partitioned_data.get('y_train')
                y_test = partitioned_data.get('y_test')
                
                train_data = X_train.copy() if hasattr(X_train, 'copy') else pd.DataFrame(X_train)
                if y_train is not None:
                    train_data.insert(0, 'target', y_train)
                train_path = os.path.join(dir_path, 'train_data.csv')
                train_data.to_csv(train_path, index=False)
                
                test_data = X_test.copy() if hasattr(X_test, 'copy') else pd.DataFrame(X_test)
                if y_test is not None:
                    test_data.insert(0, 'target', y_test)
                test_path = os.path.join(dir_path, 'test_data.csv')
                test_data.to_csv(test_path, index=False)
                
                info_path = os.path.join(dir_path, 'split_info.txt')
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(f"Data Partitioning Information\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")
                    f.write(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features\n")
                    f.write(f"Training ratio: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]):.1%}\n")
                    f.write(f"Test ratio: {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]):.1%}\n")
                
                QMessageBox.information(parent, "Save Successful", 
                                      f"All data has been saved to:\n{dir_path}\n\n"
                                      f"Files:\n- train_data.csv\n- test_data.csv\n- split_info.txt")
                
        except Exception as e:
            QMessageBox.critical(parent, "Save Failed", f"Error saving data:\n{str(e)}")
    
    def _show_partition_visualization(self, partitioned_data):
        # from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
        
        vis_window = VisualizationWindow(title="Data Partitioning Visualization")
        
        X_train = partitioned_data.get('X_train')
        X_test = partitioned_data.get('X_test')
        
        if X_train is not None:
            vis_window.plot(X_train.values if hasattr(X_train, 'values') else X_train, 
                          title="Training Data")
        
        if X_test is not None:
            vis_window.plot(X_test.values if hasattr(X_test, 'values') else X_test, 
                          title="Test Data")
        
        vis_window.show()

    def partition_data(self, method, params=None, task_type=None):
        try:
            if self.data_model.data is None:
                raise ValueError("No data loaded")
            
            if params is None:
                params = {}
            
            partitioned_data = self.data_service.partition_data(
                data=self.data_model.data,
                method=method,
                params=params,
                force_classification=(task_type == 'classification'),
                file_path=getattr(self.data_model, 'file_path', None)
            )
            
            self.data_model.set_partitioned_data(partitioned_data)
            
            try:
                dialog = VisualizationWindow(
                    partitioned_data=partitioned_data,
                    method=method,
                    params=params
                )
                dialog.exec_()
            except Exception as viz_error:
                pass
            
            return partitioned_data
            
        except Exception as e:
            self.view.display_error(f"Data partitioning failed: {str(e)}")
            return None
    
    def _check_data_types(self, X_train, y_train, X_test, y_test):
        if hasattr(X_train, 'dtypes'):
            non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
            na_cols = X_train.columns[X_train.isna().any()].tolist()
        
        if y_train is not None:
            if hasattr(y_train, 'nunique'):
                unique_values = y_train.nunique()
                total_samples = len(y_train)
                # Basic data type check, no detailed output needed
            else:
                if hasattr(y_train, 'dtype') and hasattr(y_train, 'shape'):
                    unique_values = len(np.unique(y_train))
                    total_samples = y_train.shape[0]
