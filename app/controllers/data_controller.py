from PyQt5.QtWidgets import QFileDialog
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

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.view, self.translator.get_text("select_data_file"), "",
            "CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*.*)"
        )
        
        if file_path:
            try:
                data = self.data_service.load_data(file_path)
                self.data_model.set_data(data)
                self.view.display_data(data)
                return True
            except DataServiceError as e:
                self.view.show_error(str(e))
                return False
        return False

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
            
            print(f"🔄 Applying data partitioning with method: {method}")
            print(f"   Parameters: {params}")
            
            target_column = params.get('target_column', 0)
            if isinstance(target_column, str) and target_column.isdigit():
                target_column = int(target_column)
            elif target_column == 'label':
                target_column = 0
            
            if isinstance(target_column, int) and 0 <= target_column < data.shape[1]:
                y_raw = data.iloc[:, target_column]
            else:
                y_raw = data.iloc[:, 0]
            
            print(f"📊 Raw labels analysis:")
            print(f"   Sample labels: {list(y_raw.head())}")
            print(f"   Label data type: {y_raw.dtype}")
            print(f"   Unique labels: {len(y_raw.unique())}")
            
            self.label_processor = EnhancedLabelProcessor()
            
            if task_type is None:
                detected_task_type = self.label_processor.detect_task_type(y_raw)
                print(f"🤖 Auto-detected task type: {detected_task_type}")
                
                if params.get('force_classification', False):
                    detected_task_type = 'classification'
                    print("🔧 FORCED classification mode (user override)")
                elif params.get('force_regression', False):
                    detected_task_type = 'regression'
                    print("🔧 FORCED regression mode (user override)")
                
                task_type = detected_task_type
                params['task_type'] = task_type
            else:
                print(f"🎯 User-specified task type: {task_type}")
            
            try:
                processed_labels, label_metadata = self.label_processor.process_labels_smart(y_raw, task_type)
                print(f"✅ Label processing completed:")
                print(f"   Task type: {label_metadata.get('task_type')}")
                print(f"   Auto-detected: {label_metadata.get('auto_detected')}")
                
                if task_type == 'classification':
                    print(f"   Number of classes: {label_metadata.get('num_classes')}")
                    print(f"   Class names: {label_metadata.get('class_names')}")
                else:
                    try:
                        min_val = label_metadata.get('min_value', 'N/A')
                        max_val = label_metadata.get('max_value', 'N/A')
                        if min_val != 'N/A' and max_val != 'N/A':
                            print(f"   Value range: [{float(min_val):.3f}, {float(max_val):.3f}]")
                        else:
                            print(f"   Value range: [{min_val}, {max_val}]")
                    except (ValueError, TypeError):
                        print(f"   Value range: [{label_metadata.get('min_value', 'N/A')}, {label_metadata.get('max_value', 'N/A')}]")
                    print(f"   Valid samples: {label_metadata.get('valid_samples', 'N/A')}")
                
            except ValueError as label_error:
                print(f"❌ Label processing error: {label_error}")
                
                if "non-numeric values" in str(label_error) and task_type == 'regression':
                    print("🔄 Attempting to switch to classification due to non-numeric labels...")
                    try:
                        task_type = 'classification'
                        params['task_type'] = task_type
                        processed_labels, label_metadata = self.label_processor.process_labels_smart(y_raw, task_type)
                        print("✅ Successfully switched to classification mode")
                    except Exception as fallback_error:
                        print(f"❌ Fallback to classification also failed: {fallback_error}")
                        self.view.display_message(f"Label processing failed: {str(label_error)}", "Error")
                        return
                else:
                    self.view.display_message(f"Label processing failed: {str(label_error)}", "Error")
                    return
            
            params['label_metadata'] = label_metadata
            params['processed_labels'] = processed_labels
            params['original_labels'] = y_raw
            
            print(f"📋 Data partitioning summary:")
            print(f"   Method: {method}")
            print(f"   Task type: {task_type}")
            print(f"   Data shape: {data.shape}")
            print(f"   Target column: {target_column}")
            
            partitioned_data = self.data_service.partition_data(
                data, 
                method, 
                params, 
                force_classification=params.get('force_classification', False),
                file_path=file_path
            )
            
            self.data_model.X_train = partitioned_data.get('X_train')
            self.data_model.X_test = partitioned_data.get('X_test')
            self.data_model.y_train = partitioned_data.get('y_train')
            self.data_model.y_test = partitioned_data.get('y_test')
            
            self.data_model.task_type = task_type
            self.data_model.label_metadata = label_metadata
            self.data_model.label_processor = self.label_processor
            
            print(f"✅ Data partitioning completed successfully:")
            print(f"   Training set: {self.data_model.X_train.shape}")
            print(f"   Test set: {self.data_model.X_test.shape}")
            print(f"   Task type: {task_type}")
            
            self.view.display_message(f"Data successfully partitioned using {method}. Task type: {task_type}", "Info")
            
            print("🎯 Showing data partitioning results dialog...")
            try:
                self._show_partitioning_results_dialog(partitioned_data, method)
                print("✅ Data partitioning visualization displayed successfully")
            except Exception as viz_error:
                print(f"⚠️  Visualization display failed: {viz_error}")
                self.view.display_message("Data partitioned successfully, but visualization display failed", "Warning")
            
            return partitioned_data
            
        except Exception as e:
            error_msg = f"Data partitioning failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.view.display_message(error_msg, "Error")
            return None

    def _show_partitioning_results_dialog(self, partitioned_data, method):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QPushButton, QTableWidget, QTableWidgetItem, 
                                   QTabWidget, QWidget, QTextEdit, QGroupBox,
                                   QFileDialog, QMessageBox, QSplitter)
        from PyQt5.QtCore import Qt
        import pandas as pd
        import os
        
        dialog = QDialog(self.view)
        dialog.setWindowTitle(f"Data Partitioning Results - {method}")
        dialog.setModal(False)
        
        X_train = partitioned_data.get('X_train')
        X_test = partitioned_data.get('X_test')
        y_train = partitioned_data.get('y_train')
        y_test = partitioned_data.get('y_test')
        
        label_mapping = partitioned_data.get('label_mapping')
        print(f"Partition dialog - Label mapping available: {label_mapping is not None}")
        
        if y_train is not None:
            sample_labels = y_train[:5] if hasattr(y_train, '__getitem__') else list(y_train)[:5]
            are_wine_varieties = any(isinstance(label, str) and len(str(label)) > 1 and not str(label).replace('.', '').isdigit() for label in sample_labels)
            
            print(f"Raw labels are wine varieties: {are_wine_varieties}")
            if not are_wine_varieties:
                print(f"Sample labels: {list(sample_labels)}")
            
            if are_wine_varieties:
                print("Raw labels are already wine varieties, using directly")
                y_train_original = y_train
                y_test_original = y_test
            else:
                if label_mapping:
                    index_to_label = label_mapping.get('index_to_label', {})
                    print(f"Found label mapping with {len(index_to_label)} labels")
                    
                    if index_to_label:
                        try:
                            if len(y_train) > 0:
                                first_label = y_train.iloc[0] if isinstance(y_train, pd.Series) else y_train[0]
                                if isinstance(first_label, str):
                                    y_train_original = y_train
                                    print(f"Labels already in string format, using directly. y_train sample: {y_train_original[:5].tolist() if hasattr(y_train_original, '__getitem__') else list(y_train_original)[:5]}")
                                else:
                                    if isinstance(y_train, pd.Series):
                                        y_train_original = y_train.map(lambda x: index_to_label.get(int(x), str(x)))
                                    else:
                                        y_train_original = np.array([index_to_label.get(int(label), str(label)) for label in y_train])
                                    print(f"Converted y_train sample: {y_train_original[:5].tolist()}")
                            else:
                                y_train_original = y_train
                        except Exception as e:
                            print(f"Failed to convert y_train: {e}")
                            y_train_original = y_train
                        
                        try:
                            if len(y_test) > 0:
                                first_label = y_test.iloc[0] if isinstance(y_test, pd.Series) else y_test[0]
                                if isinstance(first_label, str):
                                    y_test_original = y_test
                                    print(f"Labels already in string format, using directly. y_test sample: {y_test_original[:5].tolist() if hasattr(y_test_original, '__getitem__') else list(y_test_original)[:5]}")
                                else:
                                    if isinstance(y_test, pd.Series):
                                        y_test_original = y_test.map(lambda x: index_to_label.get(int(x), str(x)))
                                    else:
                                        y_test_original = np.array([index_to_label.get(int(label), str(label)) for label in y_test])
                                    print(f"Converted y_test sample: {y_test_original[:5].tolist()}")
                            else:
                                y_test_original = y_test
                        except Exception as e:
                            print(f"Failed to convert y_test: {e}")
                            y_test_original = y_test
                    else:
                        print("No mapping and numeric labels detected")
                        print("This may be regression data being treated as classification")
                        y_train_original = y_train
                        y_test_original = y_test
                else:
                    y_train_original = y_train
                    y_test_original = y_test
        else:
            y_train_original = None
            y_test_original = None
        
        base_height = 400
        label_section_height = 0
        
        if y_train_original is not None:
            unique_labels = len(np.unique(y_train_original))
            if unique_labels > 12:
                label_section_height = 450
            else:
                label_section_height = min(250, 120 + unique_labels * 30)
        
        tab_section_height = 300
        
        total_height = base_height + label_section_height + tab_section_height
        optimal_height = max(600, min(900, total_height))
        
        max_features_to_show = min(20, X_train.shape[1])
        
        optimal_width = max(1000, min(1200, 800 + max_features_to_show * 60))
        
        dialog.resize(optimal_width, optimal_height)
        
        layout = QVBoxLayout(dialog)
        
        title_label = QLabel(f"Data Partitioning Results")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
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
        stats_label.setStyleSheet("""
            QLabel {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11pt;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                line-height: 1.6;
            }
        """)
        stats_lines = stats_text.count('\n') + 1
        calculated_height = max(150, min(300, stats_lines * 25 + 40))
        stats_label.setMinimumHeight(calculated_height)
        stats_label.setMaximumHeight(calculated_height + 20)
        stats_label.setWordWrap(True)
        stats_layout.addWidget(stats_label)
        layout.addWidget(stats_group)
        
        if y_train_original is not None:
            label_group = QGroupBox("Label Distribution")
            label_layout = QVBoxLayout(label_group)
            
            if hasattr(y_train_original, 'value_counts'):
                train_counts = y_train_original.value_counts()
                test_counts = y_test_original.value_counts() if y_test_original is not None else pd.Series(dtype=int)
            else:
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
            label_table.setColumnCount(3)
            label_table.setHorizontalHeaderLabels(["Label", "Training Set", "Test Set"])
            
            label_table.setAlternatingRowColors(True)
            label_table.setStyleSheet("""
                QTableWidget {
                    gridline-color: #d0d0d0;
                    background-color: white;
                }
                QTableWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #e0e0e0;
                }
                QTableWidget::item:selected {
                    background-color: #e3f2fd;
                }
            """)
            
            for i, label in enumerate(all_labels):
                label_item = QTableWidgetItem(str(label))
                label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
                label_table.setItem(i, 0, label_item)
                
                train_item = QTableWidgetItem(str(train_counts.get(label, 0)))
                train_item.setFlags(train_item.flags() & ~Qt.ItemIsEditable)
                label_table.setItem(i, 1, train_item)
                
                test_item = QTableWidgetItem(str(test_counts.get(label, 0)))
                test_item.setFlags(test_item.flags() & ~Qt.ItemIsEditable)
                label_table.setItem(i, 2, test_item)
            
            label_table.resizeColumnsToContents()
            label_table.setColumnWidth(0, max(150, label_table.columnWidth(0)))
            label_table.setColumnWidth(1, 100)
            label_table.setColumnWidth(2, 100)
            
            header_height = 30
            row_height = 30
            scrollbar_buffer = 20
            border_padding = 10
            
            optimal_height = header_height + (len(all_labels) * row_height) + scrollbar_buffer + border_padding
            
            min_height = max(120, min(optimal_height, 200))
            max_height = min(450, optimal_height)
            
            if len(all_labels) > 12:
                label_table.setMinimumHeight(max_height)
                label_table.setMaximumHeight(max_height)
            else:
                label_table.setMinimumHeight(min_height)
                label_table.setMaximumHeight(min_height + 20)
            
            label_layout.addWidget(label_table)
            layout.addWidget(label_group)
        
        tab_widget = QTabWidget()
        
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        
        train_info_label = QLabel(f"Full Training Set: {X_train.shape[0]} samples × {X_train.shape[1]} features (showing first 20 features)")
        train_info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 5px;
                border-radius: 3px;
                margin-bottom: 5px;
            }
        """)
        train_layout.addWidget(train_info_label)
        
        train_table = QTableWidget()
        train_table.setRowCount(min(10, X_train.shape[0]))
        train_table.setColumnCount(max_features_to_show + (1 if y_train_original is not None else 0))
        
        headers = [f"Feature{i+1}" for i in range(max_features_to_show)]
        if y_train_original is not None:
            headers.append("Label")
        train_table.setHorizontalHeaderLabels(headers)
        
        train_table.setHorizontalScrollBarPolicy(1)
        train_table.setVerticalScrollBarPolicy(0)
        train_table.horizontalHeader().setStretchLastSection(False)
        
        column_width = 80
        for i in range(max_features_to_show):
            train_table.setColumnWidth(i, column_width)
        if y_train_original is not None:
            train_table.setColumnWidth(max_features_to_show, column_width + 20)
            
        train_table.setMinimumWidth(600)
        
        for i in range(min(10, X_train.shape[0])):
            for j in range(max_features_to_show):
                value = X_train.iloc[i, j] if hasattr(X_train, 'iloc') else X_train[i, j]
                train_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))
            if y_train_original is not None:
                y_value = y_train_original.iloc[i] if hasattr(y_train_original, 'iloc') else y_train_original[i]
                train_table.setItem(i, max_features_to_show, QTableWidgetItem(str(y_value)))
        
        train_layout.addWidget(train_table)
        tab_widget.addTab(train_tab, f"Training Set Preview ({X_train.shape[0]} samples)")
        
        test_tab = QWidget()
        test_layout = QVBoxLayout(test_tab)
        
        test_info_label = QLabel(f"Full Test Set: {X_test.shape[0]} samples × {X_test.shape[1]} features (showing first 20 features)")
        test_info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 5px;
                border-radius: 3px;
                margin-bottom: 5px;
            }
        """)
        test_layout.addWidget(test_info_label)
        
        test_table = QTableWidget()
        test_table.setRowCount(min(10, X_test.shape[0]))
        test_table.setColumnCount(max_features_to_show + (1 if y_test_original is not None else 0))
        test_table.setHorizontalHeaderLabels(headers)
        
        test_table.setHorizontalScrollBarPolicy(1)
        test_table.setVerticalScrollBarPolicy(0)
        test_table.horizontalHeader().setStretchLastSection(False)
        
        for i in range(max_features_to_show):
            test_table.setColumnWidth(i, column_width)
        if y_test_original is not None:
            test_table.setColumnWidth(max_features_to_show, column_width + 20)
            
        test_table.setMinimumWidth(600)
        
        for i in range(min(10, X_test.shape[0])):
            for j in range(max_features_to_show):
                value = X_test.iloc[i, j] if hasattr(X_test, 'iloc') else X_test[i, j]
                test_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))
            if y_test_original is not None:
                y_value = y_test_original.iloc[i] if hasattr(y_test_original, 'iloc') else y_test_original[i]
                test_table.setItem(i, max_features_to_show, QTableWidgetItem(str(y_value)))
        
        test_layout.addWidget(test_table)
        tab_widget.addTab(test_tab, f"Test Set Preview ({X_test.shape[0]} samples)")
        
        layout.addWidget(tab_widget)
        
        button_layout = QHBoxLayout()
        
        save_train_btn = QPushButton("Save Training Set")
        save_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        save_train_btn.clicked.connect(lambda: self._save_partition_data(dialog, X_train, y_train, "Training Set"))
        button_layout.addWidget(save_train_btn)
        
        save_test_btn = QPushButton("Save Test Set")
        save_test_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        save_test_btn.clicked.connect(lambda: self._save_partition_data(dialog, X_test, y_test, "Test Set"))
        button_layout.addWidget(save_test_btn)
        
        save_all_btn = QPushButton("Save All")
        save_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d68910;
            }
        """)
        save_all_btn.clicked.connect(lambda: self._save_all_partition_data(dialog, partitioned_data))
        button_layout.addWidget(save_all_btn)
        
        vis_btn = QPushButton("Visualize")
        vis_btn.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
        """)
        vis_btn.clicked.connect(lambda: self._show_partition_visualization(partitioned_data))
        button_layout.addWidget(vis_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
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
            self.view.show_error(f"Data partitioning failed: {str(e)}")
            return None
    
    def _check_data_types(self, X_train, y_train, X_test, y_test):
        if hasattr(X_train, 'dtypes'):
            print("Feature data type check:")
            print(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
            print(f"X_test type: {type(X_test)}, shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}")
            
            non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Detected {len(non_numeric_cols)} non-numeric feature columns: {non_numeric_cols[:5]}...")
            
            na_cols = X_train.columns[X_train.isna().any()].tolist()
            if na_cols:
                print(f"Warning: Detected {len(na_cols)} columns with missing values: {na_cols[:5]}...")
        
        if y_train is not None:
            print("Target variable type check:")
            print(f"y_train type: {type(y_train)}, shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}")
            print(f"y_test type: {type(y_test)}, shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}")
            
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_outputs", f"check_{int(time.time())}")
                os.makedirs(debug_dir, exist_ok=True)
                
                if isinstance(X_train, pd.DataFrame):
                    X_sample = X_train.head(20)
                    X_sample.to_csv(os.path.join(debug_dir, "X_train_sample.csv"), index=False)
                
                if isinstance(y_train, pd.Series):
                    y_series = y_train.head(50)
                    y_series.to_csv(os.path.join(debug_dir, "y_train_sample.csv"), index=False)
                elif isinstance(y_train, np.ndarray):
                    pd.Series(y_train[:50], name="target").to_csv(os.path.join(debug_dir, "y_train_sample.csv"), index=False)
                
                if hasattr(y_train, 'value_counts'):
                    y_counts = y_train.value_counts()
                    y_counts.to_csv(os.path.join(debug_dir, "y_train_counts.csv"), index=True)
                elif hasattr(y_train, 'shape'):
                    unique, counts = np.unique(y_train, return_counts=True)
                    pd.DataFrame({
                        'value': unique,
                        'count': counts
                    }).to_csv(os.path.join(debug_dir, "y_train_counts.csv"), index=False)
                
                with open(os.path.join(debug_dir, "data_info.txt"), "w") as f:
                    f.write(f"X_train type: {type(X_train)}\n")
                    f.write(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n")
                    f.write(f"y_train type: {type(y_train)}\n")
                    f.write(f"y_train dtype: {y_train.dtype if hasattr(y_train, 'dtype') else 'N/A'}\n")
                    f.write(f"y_train unique values: {y_train.unique() if hasattr(y_train, 'unique') else np.unique(y_train) if hasattr(y_train, 'shape') else 'N/A'}\n")
                    
                print(f"Saved data samples to {debug_dir} for inspection")
            except Exception as e:
                print(f"Error saving debug data: {str(e)}")
            
            if hasattr(y_train, 'nunique'):
                unique_values = y_train.nunique()
                total_samples = len(y_train)
                
                if unique_values <= 10:
                    print(f"Target variable has few unique values ({unique_values}), inferred as classification task")
                    if hasattr(y_train, 'dtype') and y_train.dtype.kind == 'f':
                        print(f"Warning: Classification task using float labels ({y_train.dtype}), may need conversion to integers")
                    elif hasattr(y_train, 'dtype') and y_train.dtype.kind in ['i', 'u']:
                        print(f"Appropriate integer labels detected ({y_train.dtype}) for classification task")
                    else:
                        print(f"Label type: {y_train.dtype if hasattr(y_train, 'dtype') else type(y_train)}")
                elif unique_values > total_samples * 0.1:
                    print(f"Target variable has many unique values ({unique_values}, {unique_values/total_samples:.2%} of total), inferred as regression task")
            else:
                if hasattr(y_train, 'dtype'):
                    print(f"Target variable data type: {y_train.dtype}")
                    if hasattr(y_train, 'shape'):
                        unique_values = len(np.unique(y_train))
                        total_samples = y_train.shape[0]
                        
                        if unique_values <= 10:
                            print(f"Target variable has few unique values ({unique_values}), inferred as classification task")
                            if y_train.dtype.kind == 'f':
                                print(f"Warning: Classification task using float labels ({y_train.dtype}), may need conversion to integers")
                            elif y_train.dtype.kind in ['i', 'u']:
                                print(f"Appropriate integer labels detected ({y_train.dtype}) for classification task")
                        elif unique_values > total_samples * 0.1:
                            print(f"Target variable has many unique values ({unique_values}, {unique_values/total_samples:.2%} of total), inferred as regression task")
