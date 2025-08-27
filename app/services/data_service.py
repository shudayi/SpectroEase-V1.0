# app/services/data_service.py

from utils.exceptions import DataServiceError
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneGroupOut, ShuffleSplit, StratifiedShuffleSplit
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
import logging
import numpy as np
import os
import time
from app.utils.label_processor import EnhancedLabelProcessor

class DataService:
    def __init__(self, plugins: dict):
        self.plugins = plugins
        self.label_mapping = None
        self.label_processor = EnhancedLabelProcessor()
        # 数据导入保守规则阈值（需求D）
        from config.settings import Settings
        self.settings = Settings()
        self.data_import_stats = {}  # 存储数据导入统计信息

    def load_data(self, file_path: str, skip_first_row=False) -> pd.DataFrame:
        try:
            # 加载数据
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
            
            # 数据导入保守规则检查（需求D）
            self.data_import_stats = self._analyze_data_quality(data, file_path)
            
            # 应用数据类型转换和质量检查
            processed_data = self._apply_conservative_rules(data)
            
            return processed_data
        except Exception as e:
            raise DataServiceError(f"Error loading data: {str(e)}")
    
    def _analyze_data_quality(self, data: pd.DataFrame, file_path: str) -> dict:
        """分析数据质量并生成统计信息（需求D）"""
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
            
            # 检查数值转换可能性
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
                
                # 应用数值转换阈值
                if col_stats['numeric_conversion_rate'] >= self.settings.numeric_conversion_threshold:
                    col_stats['conversion_decision'] = 'convert_to_numeric'
                    logging.info(f"Column '{col_name}': {col_stats['numeric_conversion_rate']:.1%} numeric convertible (>= {self.settings.numeric_conversion_threshold:.1%} threshold) - Will convert to numeric")
                else:
                    col_stats['conversion_decision'] = 'keep_as_string'
                    logging.info(f"Column '{col_name}': {col_stats['numeric_conversion_rate']:.1%} numeric convertible (< {self.settings.numeric_conversion_threshold:.1%} threshold) - Will keep as string")
            
            # 检查缺失率警戒线
            if col_stats['missing_rate'] > self.settings.missing_rate_warning_threshold:
                warning_msg = f"Column '{col_name}' has high missing rate: {col_stats['missing_rate']:.1%} (> {self.settings.missing_rate_warning_threshold:.1%} threshold)"
                stats['warnings'].append(warning_msg)
                logging.warning(warning_msg)
            
            stats['column_analysis'][col_name] = col_stats
        
        # 记录总体统计
        logging.info(f"Data import analysis completed for {file_path}:")
        logging.info(f"  - Total: {stats['total_rows']} rows, {stats['total_columns']} columns")
        logging.info(f"  - Warnings: {len(stats['warnings'])} issues detected")
        logging.info(f"  - Thresholds: numeric_conversion={self.settings.numeric_conversion_threshold:.1%}, missing_rate_warning={self.settings.missing_rate_warning_threshold:.1%}")
        
        return stats
    
    def _apply_conservative_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用保守规则进行数据处理（需求D）"""
        processed_data = data.copy()
        
        for col_name, col_stats in self.data_import_stats['column_analysis'].items():
            if col_stats['conversion_decision'] == 'convert_to_numeric':
                try:
                    # 转换为数值类型
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

    def partition_data(self, data=None, method="Train-Test Split", params=None, force_classification=False, file_path=None):
        if data is None:
            raise ValueError("Data is required for partitioning")
        if params is None:
            params = {}
        
        y = None
        X = None
        
        target_column = params.get('target_column', 0)
        
        if target_column is None:
            target_column = self._auto_detect_target_column(data)
        elif isinstance(target_column, str) and target_column.isdigit():
            target_column = int(target_column)
        elif target_column == 'label' or target_column == 1:
            target_column = self._auto_detect_target_column(data)
        
        if isinstance(target_column, int) and 0 <= target_column < data.shape[1]:
            y = data.iloc[:, target_column].copy()
            X = data.drop(data.columns[target_column], axis=1).copy()
        else:
            if file_path and os.path.exists(file_path):
                try:
                    reloaded_data = self.load_data(file_path)
                    auto_target_col = self._auto_detect_target_column(reloaded_data)
                    y = reloaded_data.iloc[:, auto_target_col].copy()
                    X = reloaded_data.drop(reloaded_data.columns[auto_target_col], axis=1).copy()
                except Exception as e:
                    y = data.iloc[:, 0].copy()
                    X = data.iloc[:, 1:].copy()
            else:
                y = data.iloc[:, 0].copy()
                X = data.iloc[:, 1:].copy()
        
        if force_classification:
            is_classification = True
            task_type = 'classification'
        else:
            task_type = self.label_processor.detect_task_type(y)
            is_classification = (task_type == 'classification')
        
        unique_labels = len(y.unique()) if hasattr(y, 'unique') else len(set(y))
        
        numeric_labels = []
        string_labels = []
        boolean_labels = []
        
        for label in y.unique() if hasattr(y, 'unique') else set(y):
            try:
                if pd.isna(label):
                    continue
                float_val = float(label)
                if float_val.is_integer():
                    numeric_labels.append(int(float_val))
                else:
                    numeric_labels.append(float_val)
            except (ValueError, TypeError):
                str_label = str(label).strip()
                if str_label.lower() in ['true', 'false']:
                    boolean_labels.append(str_label.lower() == 'true')
                else:
                    string_labels.append(str_label)
        
        total_label_types = len([x for x in [numeric_labels, string_labels, boolean_labels] if x])
        
        final_unique_labels = y.unique() if hasattr(y, 'unique') else list(set(y))
        
        if len(final_unique_labels) < 2:
            raise ValueError("Insufficient label categories after cleaning, please check data quality")
        
        if is_classification:
            if y.dtype.kind not in 'OSU':
                y = y.astype(str)
            y_consistent = y.astype(str)
            unique_labels_str = [str(label) for label in final_unique_labels]
            label_to_index = {str(label): idx for idx, label in enumerate(unique_labels_str)}
            index_to_label = {idx: str(label) for idx, label in enumerate(unique_labels_str)}
            self.label_mapping = {
                'label_to_index': label_to_index,
                'index_to_label': index_to_label,
                'preserve_original_labels': True
            }
        else:
            try:
                regression_task_type = self.label_processor.detect_task_type(y)
                if regression_task_type != 'regression':
                    raise ValueError(f"Labels appear to be for {regression_task_type} task, not regression. String labels like 'ClassC' cannot be used for regression.")
                if not pd.api.types.is_numeric_dtype(y):
                    y_consistent = pd.to_numeric(y, errors='coerce')
                    nan_count = pd.isna(y_consistent).sum()
                    if nan_count > len(y_consistent) * 0.5:
                        raise ValueError(f"Too many non-numeric labels ({nan_count}/{len(y_consistent)}) for regression task. Labels like 'ClassC' suggest this is a classification task.")
                else:
                    y_consistent = y
                self.label_mapping = None
            except ValueError as ve:
                raise ve
            except Exception as e:
                raise ValueError(f"Unable to convert labels to numeric type: {str(e)}")
        
        if method == "Train-Test Split":
            test_size = params.get('test_size', 0.3)
            random_state = params.get('random_state', 42)
            shuffle = params.get('shuffle', True)
            stratify = y_consistent if is_classification and unique_labels <= 50 and unique_labels >= 2 else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_consistent,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
            )
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'method': method,
                'params': params,
                'label_mapping': self.label_mapping if hasattr(self, 'label_mapping') else None
            }
        elif method == "K-Fold":
            n_splits = params.get('n_splits', 5)
            random_state = params.get('random_state', 42)
            shuffle = params.get('shuffle', True)
            
            if is_classification:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            
            for train_idx, test_idx in kf.split(X, y_consistent):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_consistent.iloc[train_idx], y_consistent.iloc[test_idx]
                break
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'method': method,
                'params': params,
                'label_mapping': self.label_mapping if hasattr(self, 'label_mapping') else None,
                'cv_splits': list(kf.split(X, y_consistent))
            }
        elif method == "LOGO":
            groups_column = params.get('groups_column', 'label')
            logo = LeaveOneGroupOut()
            groups = y_consistent
            try:
                for train_idx, test_idx in logo.split(X, y_consistent, groups):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_consistent.iloc[train_idx], y_consistent.iloc[test_idx]
                    break
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_consistent, test_size=0.2, random_state=42,
                    stratify=y_consistent if is_classification else None
                )
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'method': method,
                'params': params,
                'label_mapping': self.label_mapping if hasattr(self, 'label_mapping') else None
            }
        elif method == "Random":
            n_splits = params.get('n_splits', 5)
            test_size = params.get('test_size', 0.2)
            random_state = params.get('random_state', 42)
            rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            for train_idx, test_idx in rs.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_consistent.iloc[train_idx], y_consistent.iloc[test_idx]
                break
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'method': method,
                'params': params,
                'label_mapping': self.label_mapping if hasattr(self, 'label_mapping') else None,
                'cv_splits': list(rs.split(X))
            }
        elif method == "Stratified":
            n_splits = params.get('n_splits', 5)
            test_size = params.get('test_size', 0.2)
            random_state = params.get('random_state', 42)
            if is_classification:
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                for train_idx, test_idx in sss.split(X, y_consistent):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_consistent.iloc[train_idx], y_consistent.iloc[test_idx]
                    break
            else:
                rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                for train_idx, test_idx in rs.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_consistent.iloc[train_idx], y_consistent.iloc[test_idx]
                    break
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'method': method,
                'params': params,
                'label_mapping': self.label_mapping if hasattr(self, 'label_mapping') else None
            }
        else:
            if method in self.plugins:
                algorithm: DataPartitioningAlgorithm = self.plugins[method]
                combined_data = X.copy()
                combined_data[y.name if hasattr(y, 'name') and y.name else 'target'] = y_consistent
                result = algorithm.split_data(combined_data, params)
                result['label_mapping'] = self.label_mapping if hasattr(self, 'label_mapping') else None
                return result
            else:
                raise ValueError(f"Partitioning method '{method}' not implemented")

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
