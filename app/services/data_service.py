# app/services/data_service.py

from app.utils.exceptions import DataServiceError
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneGroupOut, ShuffleSplit, StratifiedShuffleSplit
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
import logging
import numpy as np
import os
import time
from typing import Dict, Any
from app.utils.label_processor import EnhancedLabelProcessor

class DataService:
    def __init__(self, plugins: dict):
        self.plugins = plugins
        self.label_mapping = None
        self.label_processor = EnhancedLabelProcessor()
        # Conservative rule thresholds for data import (Requirement D)
        from config.settings import Settings
        self.settings = Settings()
        self.data_import_stats = {}  # Store data import statistics

    def load_data(self, file_path: str, skip_first_row=False) -> pd.DataFrame:
        try:
            # Load data
            if file_path.endswith('.csv'):
                header = 1 if skip_first_row else 0
                preview = pd.read_csv(file_path, nrows=5)
                first_col_name = preview.columns[0] if len(preview.columns) > 0 else None
                category_keywords = ['category', 'class', 'type', 'label', 'variety', 'group', 'target']
                is_category_col = False
                if first_col_name and any(keyword.lower() in first_col_name.lower() for keyword in category_keywords):
                    is_category_col = True
                first_val = preview.iloc[0, 0] if len(preview) > 0 and len(preview.columns) > 0 else None
                has_label_column = is_category_col or (first_val is not None and isinstance(first_val, str) and not first_val.replace('.', '').replace('-', '').isdigit())
                data = pd.read_csv(file_path, header=header, encoding='utf-8')
                if skip_first_row or (len(data) > 0 and pd.isna(data.iloc[0]).all()):
                    data = data.iloc[1:].reset_index(drop=True)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                if skip_first_row:
                    data = pd.read_excel(file_path, header=1)
                else:
                    data = pd.read_excel(file_path)
                if skip_first_row or (len(data) > 0 and pd.isna(data.iloc[0]).all()):
                    data = data.iloc[1:].reset_index(drop=True)
            else:
                raise DataServiceError(f"Unsupported file format: {file_path}")
            
            # Data quality check and interactive handling (Reviewer 1 Comment 1)
            processed_data = self._handle_data_quality_issues(data, file_path)
            
            # Conservative rule check for data import (Requirement D)
            self.data_import_stats = self._analyze_data_quality(processed_data, file_path)
            
            # Apply data type conversion and quality checks
            processed_data = self._apply_conservative_rules(processed_data)
            
            return processed_data
        except Exception as e:
            raise DataServiceError(f"Error loading data: {str(e)}")
    
    def _handle_data_quality_issues(self, data: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Handle data quality issues with interactive dialog (Reviewer 1 Comment 1)"""
        from app.views.data_quality_dialog import DataQualityAnalyzer, DataQualityDialog
        from PyQt5.QtWidgets import QApplication
        
        # Analyze data quality
        analysis = DataQualityAnalyzer.analyze_data_quality(data)
        
        # If no issues found, return original data
        if analysis['severity'] == 'none':
            print("✅ No data quality issues detected")
            return data
        
        print(f"⚠️ Data quality issues detected (severity: {analysis['severity']})")
        
        # Check if we're in a GUI environment
        app = QApplication.instance()
        if app is None:
            # No GUI available, apply default handling
            print("🔧 No GUI available, applying default data quality handling...")
            return self._apply_default_data_quality_handling(data, analysis)
        
        # Show interactive dialog
        try:
            dialog = DataQualityDialog(data, file_path)
            
            if dialog.exec_() == dialog.Accepted:
                processed_data, action = dialog.get_processed_data()
                print(f"✅ Data quality handled via '{action}' action")
                return processed_data
            else:
                # User cancelled
                raise DataServiceError("Data loading cancelled due to quality issues")
                
        except Exception as e:
            print(f"⚠️ Dialog failed: {e}, applying default handling...")
            return self._apply_default_data_quality_handling(data, analysis)
    
    def _apply_default_data_quality_handling(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply default data quality handling when no GUI is available"""
        processed_data = data.copy()
        
        # Remove infinite values
        if 'inf' in analysis['issues']:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                processed_data = processed_data[~np.isinf(processed_data[col])]
            print("🔧 Removed infinite values")
        
        # Handle NaN values based on severity
        if 'nan' in analysis['issues']:
            nan_info = analysis['issues']['nan']
            if nan_info['max_nan_percentage'] > 50:
                # Remove columns with >50% NaN
                nan_percentages = processed_data.isnull().sum() / len(processed_data)
                cols_to_remove = nan_percentages[nan_percentages > 0.5].index
                processed_data = processed_data.drop(columns=cols_to_remove)
                print(f"🗑️ Removed {len(cols_to_remove)} columns with >50% NaN values")
            
            # Remove rows with any remaining NaN
            processed_data = processed_data.dropna()
            print("🗑️ Removed rows with NaN values")
        
        return processed_data
    
    def _analyze_data_quality(self, data: pd.DataFrame, file_path: str) -> dict:
        """Analyze data quality and generate statistics (Requirement D)"""
        stats = {
            'file_path': file_path,
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'column_analysis': {},
            'warnings': []
        }
        
        for col_name in data.columns:
            col_data = data[col_name]
            col_stats = {
                'name': col_name,
                'total_values': len(col_data),
                'missing_count': col_data.isnull().sum(),
                'missing_rate': col_data.isnull().sum() / len(col_data),
                'unique_count': col_data.nunique(),
                'data_type': str(col_data.dtype),
                'numeric_convertible_count': 0,
                'numeric_conversion_rate': 0.0,
                'conversion_decision': 'keep_original'
            }
            
            # Check numeric conversion possibility
            if col_data.dtype == 'object':
                numeric_count = 0
                for val in col_data.dropna():
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                col_stats['numeric_convertible_count'] = numeric_count
                col_stats['numeric_conversion_rate'] = numeric_count / len(col_data.dropna()) if len(col_data.dropna()) > 0 else 0.0
                
                # Apply numeric conversion threshold
                if col_stats['numeric_conversion_rate'] >= self.settings.numeric_conversion_threshold:
                    col_stats['conversion_decision'] = 'convert_to_numeric'
                    logging.info(f"Column '{col_name}': {col_stats['numeric_conversion_rate']:.1%} numeric convertible (>= {self.settings.numeric_conversion_threshold:.1%} threshold) - Will convert to numeric")
                else:
                    col_stats['conversion_decision'] = 'keep_as_string'
                    logging.info(f"Column '{col_name}': {col_stats['numeric_conversion_rate']:.1%} numeric convertible (< {self.settings.numeric_conversion_threshold:.1%} threshold) - Will keep as string")
            
            # Check missing rate warning threshold
            if col_stats['missing_rate'] > self.settings.missing_rate_warning_threshold:
                warning_msg = f"Column '{col_name}' has high missing rate: {col_stats['missing_rate']:.1%} (> {self.settings.missing_rate_warning_threshold:.1%} threshold)"
                stats['warnings'].append(warning_msg)
                logging.warning(warning_msg)
            
            stats['column_analysis'][col_name] = col_stats
        
        # Record overall statistics
        logging.info(f"Data import analysis completed for {file_path}:")
        logging.info(f"  - Total: {stats['total_rows']} rows, {stats['total_columns']} columns")
        logging.info(f"  - Warnings: {len(stats['warnings'])} issues detected")
        logging.info(f"  - Thresholds: numeric_conversion={self.settings.numeric_conversion_threshold:.1%}, missing_rate_warning={self.settings.missing_rate_warning_threshold:.1%}")
        
        return stats
    
    def _apply_conservative_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply conservative rules for data processing (Requirement D)"""
        processed_data = data.copy()
        
        for col_name, col_stats in self.data_import_stats['column_analysis'].items():
            if col_stats['conversion_decision'] == 'convert_to_numeric':
                try:
                    # Convert to numeric type
                    processed_data[col_name] = pd.to_numeric(processed_data[col_name], errors='coerce')
                    logging.info(f"Successfully converted column '{col_name}' to numeric type")
                except Exception as e:
                    logging.error(f"Failed to convert column '{col_name}' to numeric: {e}")
        
        return processed_data

    def load_data_with_first_row(self, file_path: str) -> pd.DataFrame:
        return self.load_data(file_path, skip_first_row=True)

    def _auto_detect_target_column(self, data, target_column=None):
        if target_column is not None and isinstance(target_column, int):
            if 0 <= target_column < data.shape[1]:
                return target_column
        if target_column is not None and isinstance(target_column, str):
            if target_column in data.columns:
                return data.columns.get_loc(target_column)
        category_keywords = ['label', 'class', 'target', 'category', 'type', 'variety', 'group']
        for col_idx, col_name in enumerate(data.columns):
            if any(keyword.lower() in str(col_name).lower() for keyword in category_keywords):
                return col_idx
        for col_idx in range(min(5, data.shape[1])):
            col = data.iloc[:, col_idx]
            unique_count = col.nunique()
            if 2 <= unique_count <= 20 and unique_count / len(col) < 0.1:
                return col_idx
        return 0

    def partition_data(self, data: pd.DataFrame, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refactored data partitioning method.
        This method is now a pure function that relies entirely on the controller 
        for label processing and task type determination. It only performs the split.
        """
        if data is None:
            raise ValueError("Data is required for partitioning")

        # --- Get pre-processed data from params ---
        task_type = params.get('task_type')
        target_column_index = params.get('target_column', 0)
        processed_labels = params.get('processed_labels') # Numeric labels from controller
        original_labels = params.get('original_labels')   # Original labels from controller
        label_metadata = params.get('label_metadata')

        if task_type is None or processed_labels is None or original_labels is None:
            raise ValueError("Controller did not provide necessary label processing information (task_type, processed_labels).")

        # --- Separate features (X) and labels (y) ---
        X = data.drop(data.columns[target_column_index], axis=1)
        # Use the already processed numeric labels for splitting
        y_numeric = pd.Series(processed_labels, index=X.index) 
        # Keep original labels for the final returned dictionary
        y_original = original_labels

        # --- Determine Stratification ---
        # Stratification should be based on the numeric (integer) labels for classification
        stratify_labels = y_numeric if task_type == 'classification' else None

        # --- Select and Execute Partitioning Method ---
        if method == "Train-Test Split":
            test_size = params.get('test_size', 0.3)
            random_state = params.get('random_state', 42)
            shuffle = params.get('shuffle', True)
            
            X_train, X_test, y_train_orig, y_test_orig, y_train_num, y_test_num = train_test_split(
                X, y_original, y_numeric,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_labels
            )
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train_orig, 'y_test': y_test_orig,
                'y_train_numeric': y_train_num, 'y_test_numeric': y_test_num, # For internal use
                'method': method, 'params': params, 'label_metadata': label_metadata
            }

        elif method == "K-Fold":
            n_splits = params.get('n_splits', 5)
            random_state = params.get('random_state', 42)
            shuffle = params.get('shuffle', True)
            
            if task_type == 'classification':
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splitter = kf.split(X, y_numeric)
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splitter = kf.split(X)

            # Return the first fold as a representative split
            train_idx, test_idx = next(splitter)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_orig, y_test_orig = y_original.iloc[train_idx], y_original.iloc[test_idx]
            y_train_num, y_test_num = y_numeric.iloc[train_idx], y_numeric.iloc[test_idx]
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train_orig, 'y_test': y_test_orig,
                'y_train_numeric': y_train_num, 'y_test_numeric': y_test_num,
                'method': method, 'params': params, 'label_metadata': label_metadata,
                'cv_splits': list(kf.split(X, y_numeric)) # Provide all splits
            }
        
        # 🔧 New: LOGO (Leave-One-Group-Out)
        elif method == "LOGO":
            from sklearn.model_selection import LeaveOneGroupOut
            groups_column = params.get('groups_column', 'label')
            # Use original labels as groups
            groups = y_original.values
            
            logo = LeaveOneGroupOut()
            splits = list(logo.split(X, y_numeric, groups))
            # Return first split as representative
            train_idx, test_idx = splits[0]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_orig, y_test_orig = y_original.iloc[train_idx], y_original.iloc[test_idx]
            y_train_num, y_test_num = y_numeric.iloc[train_idx], y_numeric.iloc[test_idx]
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train_orig, 'y_test': y_test_orig,
                'y_train_numeric': y_train_num, 'y_test_numeric': y_test_num,
                'method': method, 'params': params, 'label_metadata': label_metadata
            }
        
        # 🔧 New: Random Split
        elif method == "Random":
            test_size = params.get('test_size', 0.3)
            random_state = params.get('random_state', 42)
            
            X_train, X_test, y_train_orig, y_test_orig, y_train_num, y_test_num = train_test_split(
                X, y_original, y_numeric,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=None  # No stratification
            )
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train_orig, 'y_test': y_test_orig,
                'y_train_numeric': y_train_num, 'y_test_numeric': y_test_num,
                'method': method, 'params': params, 'label_metadata': label_metadata
            }
        
        # 🔧 New: Stratified Split
        elif method == "Stratified":
            test_size = params.get('test_size', 0.3)
            random_state = params.get('random_state', 42)
            
            # Force stratified split (only for classification tasks)
            if task_type != 'classification':
                raise ValueError("Stratified split is only applicable for classification tasks")
            
            X_train, X_test, y_train_orig, y_test_orig, y_train_num, y_test_num = train_test_split(
                X, y_original, y_numeric,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=y_numeric  # Force stratification
            )
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train_orig, 'y_test': y_test_orig,
                'y_train_numeric': y_train_num, 'y_test_numeric': y_test_num,
                'method': method, 'params': params, 'label_metadata': label_metadata
            }

        # --- Handle Plugin-based Methods ---
        elif method in self.plugins:
            # V1.4.1: Use instance directly (plugins now store instances, not classes)
            algorithm = self.plugins[method]
            
            # Safety check: if it's a class, instantiate it (backward compatibility)
            if isinstance(algorithm, type):
                print(f"⚠️  WARNING: Plugin '{method}' is stored as class, instantiating...")
                algorithm = algorithm()
            
            # The plugin is expected to handle the data and params
            # We pass the combined data with original labels
            combined_data = X.copy()
            combined_data[y_original.name if hasattr(y_original, 'name') and y_original.name else 'target'] = y_original
            
            # Call the partition() method defined in the interface
            X_train, X_test, y_train, y_test = algorithm.partition(combined_data, params)
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'y_train_numeric': y_train, 'y_test_numeric': y_test,  # For compatibility
                'method': method, 'params': params, 'label_metadata': label_metadata
            }
            
        else:
            raise ValueError(f"Partitioning method '{method}' not implemented or supported by this refactored service.")

    def map_predictions_to_original(self, predictions, preserve_original=True):
        if self.label_mapping is None:
            return np.array([str(pred) for pred in predictions], dtype='<U50')
        
        if self.label_mapping.get('preserve_original_labels', False):
            if isinstance(predictions, np.ndarray):
                return np.array([str(pred) for pred in predictions], dtype='<U50')
            elif isinstance(predictions, pd.Series):
                return predictions.astype(str)
            else:
                return [str(pred) for pred in predictions]
        
        index_to_label = self.label_mapping.get('index_to_label', {})
        if not index_to_label:
            return np.array([str(pred) for pred in predictions], dtype='<U50')
        
        mapped_predictions = []
        for pred in predictions:
            try:
                if isinstance(pred, str):
                    try:
                        pred_int = int(float(pred))
                        mapped_label = index_to_label.get(pred_int, pred)
                    except (ValueError, TypeError):
                        mapped_label = pred
                else:
                    pred_int = int(round(float(pred)))
                    mapped_label = index_to_label.get(pred_int, f"Unknown_Class_{pred_int}")
                mapped_predictions.append(str(mapped_label))
            except Exception as map_error:
                mapped_predictions.append(str(pred))
        
        result = np.array(mapped_predictions, dtype='<U50')
        return result

    def _is_numeric(self, val):
        if val is None:
            return False
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def map_predictions_to_original_safe(self, predictions):
        try:
            if self.label_mapping is None:
                return np.array([str(pred) for pred in predictions])
            index_to_label = self.label_mapping.get('index_to_label', {})
            if not index_to_label:
                return np.array([str(pred) for pred in predictions])
            mapped_results = []
            for pred in predictions:
                try:
                    if isinstance(pred, (int, np.integer)):
                        mapped_label = index_to_label.get(pred, str(pred))
                    elif isinstance(pred, str):
                        try:
                            pred_int = int(pred)
                            mapped_label = index_to_label.get(pred_int, pred)
                        except ValueError:
                            mapped_label = pred
                    else:
                        mapped_label = str(pred)
                    mapped_results.append(str(mapped_label))
                except Exception as map_error:
                    mapped_results.append(str(pred))
            return np.array(mapped_results, dtype='<U50')
        except Exception as e:
            return np.array([str(pred) for pred in predictions])
