# app/services/evaluation_service.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
import pandas as pd
import numpy as np
from app.utils.exceptions import EvaluationError
import os
import traceback
import json
from app.utils.logger import setup_logger
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import KFold, cross_val_score
import csv
from datetime import datetime
from pathlib import Path
import sys

from app.utils.data_compatibility import (
    standardize_classification_labels, 
    standardize_regression_values,
    is_classification_task
)
from app.utils.label_processor import EnhancedLabelProcessor

  
def safe_classification_evaluation(metric_func: Callable) -> Callable:
    """
    Decorator to ensure classification metrics can handle mixed type labels
    
    Args:
        metric_func: Original evaluation function (accuracy_score, f1_score, etc.)
        
    Returns:
        Wrapped safe function
    """
    def wrapper(y_true, y_pred, *args, **kwargs):
        try:
            # Force convert all labels to string
            if not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
                
            # Use standardize function to handle mixed types
            y_true_str = np.array([str(label) if label is not None else "None" for label in y_true])
            y_pred_str = np.array([str(label) if label is not None else "None" for label in y_pred])
            
            y_true_idx, y_pred_idx, _ = standardize_classification_labels(y_true_str, y_pred_str)
            
            # Call original evaluation function
            return metric_func(y_true_idx, y_pred_idx, *args, **kwargs)
        except Exception as e:
            logging.error(f"Safe evaluation failed: {str(e)}")
            # Return error value
            if metric_func.__name__ == 'accuracy_score':
                return 0.0
            elif 'average' in kwargs and kwargs.get('average') == 'weighted':
                return 0.0
            else:
                # For other cases, try to use original function
                try:
                    return metric_func(y_true, y_pred, *args, **kwargs)
                except:
                    return 0.0
    
    return wrapper

# Create safe evaluation functions
safe_accuracy = safe_classification_evaluation(accuracy_score)
safe_precision = safe_classification_evaluation(precision_score)
safe_recall = safe_classification_evaluation(recall_score)
safe_f1 = safe_classification_evaluation(f1_score)

class EvaluationService:
    def __init__(self, app_instance=None):
        """Initialize evaluation service"""
        self.logger = logging.getLogger(__name__)
        self.app_instance = app_instance
        self.debug_dir = os.path.join(os.getcwd(), 'debug_data')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # **CRITICAL FIX: Initialize enhanced label processor for consistent handling**
        self.label_processor = EnhancedLabelProcessor()
        print("🔧 EvaluationService initialized with EnhancedLabelProcessor")
    
    def save_debug_data(self, data: Dict[str, Any], filename: str = None, format: str = 'json'):
        """
        Save debug data to file
        
        Args:
            data: Dictionary of data to save
            filename: Filename (without extension)
            format: Save format, supports 'json' or 'csv'
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"debug_data_{timestamp}"
        
        if format.lower() == 'json':
            file_path = os.path.join(self.debug_dir, f"{filename}.json")
            try:
                # Ensure all data is serializable
                serializable_data = {}
                for key, value in data.items():
                    if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
                        serializable_data[key] = value.tolist()
                    elif isinstance(value, np.int64):
                        serializable_data[key] = int(value)
                    elif isinstance(value, np.float64):
                        serializable_data[key] = float(value)
                    else:
                        try:
                            json.dumps({key: value})  # Test if serializable
                            serializable_data[key] = value
                        except:
                            serializable_data[key] = str(value)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Debug data saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save JSON debug data: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        elif format.lower() == 'csv':
            file_path = os.path.join(self.debug_dir, f"{filename}.csv")
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(data.keys())
                    
                    # Determine data length
                    max_length = 0
                    for value in data.values():
                        if isinstance(value, (list, np.ndarray, pd.Series)):
                            max_length = max(max_length, len(value))
                    
                    # Write data rows
                    for i in range(max_length):
                        row = []
                        for value in data.values():
                            if isinstance(value, (list, np.ndarray, pd.Series)) and i < len(value):
                                row.append(value[i])
                            elif i == 0:  # For scalar values, only write in first row
                                row.append(value)
                            else:
                                row.append("")
                        writer.writerow(row)
                
                self.logger.info(f"Debug data saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save CSV debug data: {str(e)}")
                self.logger.error(traceback.format_exc())
        else:
            self.logger.error(f"Unsupported format: {format}")

    def evaluate(self, y_true, y_pred, metrics: List[str] = None, task_type: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            metrics: List of metrics to calculate
            task_type: Task type, 'classification' or 'regression', auto-detected if None
            
        Returns:
            Dict[str, float]: Evaluation metrics results
        """
        self.logger.info("Starting model evaluation")
        
        try:
            # Save original data for debugging
            self.save_debug_data({
                'y_true_orig': y_true,
                'y_pred_orig': y_pred,
                'metrics': metrics,
                'task_type': task_type
            }, 'evaluate_raw_input')
            
            # **CRITICAL FIX: Use enhanced label processor for consistent task type detection**
            if task_type is None:
                task_type = self.label_processor.detect_task_type(y_true)
                self.logger.info(f"✅ Enhanced auto-detected task type: {task_type}")
            else:
                self.logger.info(f"🎯 User-specified task type: {task_type}")
            
            # Initialize default metrics (需求F：完整指标)
            if metrics is None:
                if task_type == 'classification':
                    # 必算指标：Accuracy、F1（macro与weighted）、ROC-AUC（多分类用OvR，macro）
                    metrics = ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_weighted', 'roc_auc_macro', 'confusion_matrix']
                else:  # regression
                    # 必算指标：RMSE、MAE、R²
                    metrics = ['rmse', 'mae', 'r2']
            
            # Call appropriate evaluation method based on task type
            if task_type == 'classification':
                return self.eval_classification_metrics(y_true, y_pred, metrics)
            else:
                return self.eval_regression_metrics(y_true, y_pred, metrics)
                
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Save error information for debugging
            self.save_debug_data({
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'y_true': str(type(y_true)) + " " + str(y_true)[:100],
                'y_pred': str(type(y_pred)) + " " + str(y_pred)[:100]
            }, 'evaluate_error')
            return {'error': str(e)}

    def eval_classification_metrics(self, y_true, y_pred, metrics: List[str]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for classification task
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metrics: List of metrics to calculate
            
        Returns:
            Dict[str, float]: Evaluation metrics results
        """
        try:
            # Use data compatibility tool to standardize labels
            y_true_idx, y_pred_idx, label_map = standardize_classification_labels(y_true, y_pred)
            
            # Save processed data for debugging
            self.save_debug_data({
                'y_true_idx': y_true_idx,
                'y_pred_idx': y_pred_idx,
                'label_map': label_map
            }, 'classification_metrics_input')
            
            results = {}
            
            # Return label mapping
            results['label_map'] = label_map
            
            # Check if the task is actually suitable for classification
            n_unique_true = len(np.unique(y_true_idx))
            n_unique_pred = len(np.unique(y_pred_idx))
            total_unique = len(label_map)
            
            self.logger.info(f"Classification metrics calculation - True label classes: {n_unique_true}, Predicted label classes: {n_unique_pred}, Total classes: {total_unique}")
            
            # If there are too many categories (>50), it might be a regression task mislabeled as classification
            if total_unique > 50:
                self.logger.warning(f"Detected too many categories ({total_unique}>50), this might be a regression problem incorrectly labeled as classification")
                # Return poor metrics to indicate this is likely the wrong task type
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'error': 'Too many categories - consider using regression instead'
                }
            
            # Calculate metrics using the encoded indices
            if 'accuracy' in metrics:
                results['accuracy'] = float(accuracy_score(y_true_idx, y_pred_idx))
            
            if 'precision' in metrics:
                # Handle warnings: target contains only one label
                unique_labels = np.unique(y_true_idx)
                if len(unique_labels) > 1:
                    results['precision'] = float(precision_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
                else:
                    results['precision'] = 1.0 if np.all(y_true_idx == y_pred_idx) else 0.0
            
            if 'recall' in metrics:
                # Handle warnings: target contains only one label
                unique_labels = np.unique(y_true_idx)
                if len(unique_labels) > 1:
                    results['recall'] = float(recall_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
                else:
                    results['recall'] = 1.0 if np.all(y_true_idx == y_pred_idx) else 0.0
            
            # F1分数计算（需求F：macro与weighted）
            unique_labels = np.unique(y_true_idx)
            if 'f1' in metrics or 'f1_macro' in metrics:
                if len(unique_labels) > 1:
                    results['f1_macro'] = float(f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0))
                else:
                    results['f1_macro'] = 1.0 if np.all(y_true_idx == y_pred_idx) else 0.0
            
            if 'f1' in metrics or 'f1_weighted' in metrics:
                if len(unique_labels) > 1:
                    results['f1_weighted'] = float(f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
                else:
                    results['f1_weighted'] = 1.0 if np.all(y_true_idx == y_pred_idx) else 0.0
            
            # 保持向后兼容
            if 'f1' in metrics:
                results['f1'] = results.get('f1_weighted', 0.0)
            
            # ROC-AUC计算（需求F：多分类用OvR，macro）
            if 'roc_auc' in metrics or 'roc_auc_macro' in metrics:
                try:
                    from sklearn.metrics import roc_auc_score
                    unique_labels = np.unique(y_true_idx)
                    
                    if len(unique_labels) == 2:  # Binary classification
                        roc_auc_value = float(roc_auc_score(y_true_idx, y_pred_idx))
                        results['roc_auc'] = roc_auc_value
                        results['roc_auc_macro'] = roc_auc_value
                    elif len(unique_labels) > 2:  # Multi-class classification
                        # 对于多分类，需要概率输出来计算ROC-AUC
                        # 这里我们设为None，因为只有预测标签没有概率
                        results['roc_auc_macro'] = None
                        results['roc_auc'] = None
                        self.logger.info("ROC-AUC calculation requires probability outputs for multi-class problems")
                    else:
                        results['roc_auc'] = None
                        results['roc_auc_macro'] = None
                except Exception as e:
                    self.logger.warning(f"Error calculating ROC AUC: {str(e)}")
                    results['roc_auc'] = None
                    results['roc_auc_macro'] = None
            
            if 'confusion_matrix' in metrics:
                # Calculate confusion matrix
                cm = confusion_matrix(y_true_idx, y_pred_idx)
                results['confusion_matrix'] = cm.tolist()
                
                # Generate confusion matrix heatmap
                try:
                    # **修复混淆矩阵显示问题：根据类别数量动态调整图形大小**
                    n_classes = len(np.unique(np.concatenate([y_true_idx, y_pred_idx])))
                    
                    # 根据类别数量动态调整图形大小
                    if n_classes <= 5:
                        figsize = (8, 6)
                        font_size = 12
                    elif n_classes <= 10:
                        figsize = (10, 8)
                        font_size = 10
                    elif n_classes <= 20:
                        figsize = (12, 10)
                        font_size = 8
                    else:
                        figsize = (15, 12)
                        font_size = 6
                    
                    plt.figure(figsize=figsize)
                    
                    # 使用更合适的热力图参数
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               cbar_kws={'shrink': 0.8},
                               annot_kws={'size': font_size})
                    
                    plt.xlabel('Predicted Labels', fontsize=font_size + 2)
                    plt.ylabel('True Labels', fontsize=font_size + 2)
                    plt.title('Confusion Matrix', fontsize=font_size + 4)
                    
                    # 调整布局以防止标签被截断
                    plt.tight_layout()
                    
                    # Convert image to base64 string
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    results['confusion_matrix_plot'] = image_base64
                except Exception as e:
                    self.logger.warning(f"Error generating confusion matrix heatmap: {str(e)}")
                    # 提供基本的混淆矩阵信息作为后备
                    results['confusion_matrix_error'] = str(e)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Try using safe fallback strategy
            try:
                self.logger.info("Attempting to evaluate metrics using safe fallback strategy...")
                results = {}
                
                # Use safe functions to calculate basic metrics
                if 'accuracy' in metrics:
                    results['accuracy'] = float(safe_accuracy(y_true, y_pred))
                if 'precision' in metrics:
                    results['precision'] = float(safe_precision(y_true, y_pred, average='weighted', zero_division=0))
                if 'recall' in metrics:
                    results['recall'] = float(safe_recall(y_true, y_pred, average='weighted', zero_division=0))
                if 'f1' in metrics:
                    results['f1'] = float(safe_f1(y_true, y_pred, average='weighted', zero_division=0))
                
                results['note'] = "Metrics calculated using safe fallback strategy"
                return results
            except Exception as fallback_error:
                self.logger.error(f"Fallback strategy also failed: {str(fallback_error)}")
                # Save error information for debugging
                self.save_debug_data({
                    'error_message': str(e),
                    'traceback': traceback.format_exc(),
                    'y_true': str(type(y_true)) + " " + str(y_true)[:100],
                    'y_pred': str(type(y_pred)) + " " + str(y_pred)[:100]
                }, 'classification_metrics_error')
                return {'error': str(e)}

    def eval_regression_metrics(self, y_true, y_pred, metrics: List[str]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for regression task
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate
            
        Returns:
            Dict[str, float]: Evaluation metrics results
        """
        try:
            # Use data compatibility tool to standardize regression values
            y_true_float, y_pred_float, valid_mask = standardize_regression_values(y_true, y_pred)
            
            # Only use valid data points to calculate metrics
            if valid_mask.sum() == 0:
                return {'error': 'No valid data points for evaluation'}
            
            y_true_valid = y_true_float[valid_mask]
            y_pred_valid = y_pred_float[valid_mask]
            
            # Save processed data for debugging
            self.save_debug_data({
                'y_true_float': y_true_float,
                'y_pred_float': y_pred_float,
                'valid_mask': valid_mask,
                'y_true_valid': y_true_valid,
                'y_pred_valid': y_pred_valid
            }, 'regression_metrics_input')
            
            results = {}
            
            if 'mae' in metrics:
                results['mae'] = float(mean_absolute_error(y_true_valid, y_pred_valid))
            
            if 'mse' in metrics:
                results['mse'] = float(mean_squared_error(y_true_valid, y_pred_valid))
            
            if 'rmse' in metrics:
                results['rmse'] = float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)))
            
            if 'r2' in metrics:
                results['r2'] = float(r2_score(y_true_valid, y_pred_valid))
            
            if 'explained_variance' in metrics:
                results['explained_variance'] = float(explained_variance_score(y_true_valid, y_pred_valid))
            
            if 'residual_plot' in metrics:
                try:
                    # Generate residual plot
                    residuals = y_true_valid - y_pred_valid
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_pred_valid, residuals)
                    plt.axhline(y=0, color='r', linestyle='-')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot')
                    
                    # Convert image to base64 string
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    results['residual_plot'] = image_base64
                except Exception as e:
                    self.logger.warning(f"Error generating residual plot: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Save error information for debugging
            self.save_debug_data({
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'y_true': str(type(y_true)) + " " + str(y_true)[:100],
                'y_pred': str(type(y_pred)) + " " + str(y_pred)[:100]
            }, 'regression_metrics_error')
            return {'error': str(e)}

    def evaluate_model(self, model, X, y, test_size=0.2, metrics=None, task_type=None) -> Dict[str, Any]:
        """
        Evaluate model (compatibility wrapper, calls evaluate method)
        
        Args:
            model: Trained model
            X: Feature data
            y: Target variable
            test_size: Test set proportion
            metrics: List of metrics to calculate
            task_type: Task type
            
        Returns:
            Dict[str, float]: Evaluation metrics results
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Convert test_size to float to ensure comparison works
            if test_size is not None:
                try:
                    test_size_float = float(test_size)
                except (ValueError, TypeError):
                    test_size_float = 0.2  # Default to 0.2 if conversion fails
            else:
                test_size_float = 0.0
                
            # Execute data split
            if test_size_float > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_float, random_state=42)
            else:
                X_test, y_test = X, y
            
            # Get prediction results
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                self.logger.error(f"Error predicting model: {str(e)}")
                return {'error': f"Model prediction failed: {str(e)}"}
            
            # Use evaluate method to evaluate model
            return self.evaluate(y_test, y_pred, metrics=metrics, task_type=task_type)
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def generate_reproducibility_info(self, model=None, preprocessing_params=None, 
                                    feature_selection_params=None, cv_results=None,
                                    global_random_state=None) -> dict:
        """生成复现性信息报告（需求F）
        
        Args:
            model: 训练好的模型
            preprocessing_params: 预处理参数
            feature_selection_params: 特征选择参数  
            cv_results: 交叉验证结果
            global_random_state: 全局随机种子
            
        Returns:
            dict: 复现性信息字典
        """
        try:
            import sklearn
            import numpy as np
            import sys
            from datetime import datetime
            
            reproducibility_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'software_versions': {
                    'python': sys.version.split()[0],
                    'numpy': np.__version__,
                    'scikit_learn': sklearn.__version__,
                    'spectroease': '1.1'  # 软件版本
                },
                'random_state': global_random_state or 42,
                'processing_pipeline': {
                    'steps': [],
                    'order': []
                }
            }
            
            # 处理流水线步骤
            pipeline_order = ['despiking', 'baseline_correction', 'scatter_correction', 
                            'normalization', 'standardization', 'derivative', 'feature_selection', 'modeling']
            
            if preprocessing_params:
                for step in pipeline_order:
                    if step in preprocessing_params and preprocessing_params[step].get('enabled', False):
                        step_info = {
                            'step': step,
                            'enabled': True,
                            'parameters': preprocessing_params[step]
                        }
                        reproducibility_info['processing_pipeline']['steps'].append(step_info)
                        reproducibility_info['processing_pipeline']['order'].append(step)
                        
                        # 特别记录去峰参数（需求A）
                        if step == 'despiking':
                            despiking_params = preprocessing_params[step]
                            reproducibility_info['despiking_statistics'] = {
                                'method': despiking_params.get('method', 'mad'),
                                'window': despiking_params.get('window', 11),
                                'threshold': despiking_params.get('threshold', 5.0),
                                'enabled': despiking_params.get('enabled', False)
                            }
            
            # 特征选择信息
            if feature_selection_params:
                reproducibility_info['feature_selection'] = {
                    'method': feature_selection_params.get('method', 'None'),
                    'n_features_selected': feature_selection_params.get('n_features', 'All'),
                    'selection_criteria': feature_selection_params.get('criteria', 'N/A')
                }
            
            # 交叉验证结果
            if cv_results:
                reproducibility_info['cross_validation'] = {
                    'method': cv_results.get('method', 'Unknown'),
                    'folds': cv_results.get('folds', 5),
                    'mean_score': cv_results.get('mean_score', None),
                    'std_score': cv_results.get('std_score', None),
                    'score_range': f"{cv_results.get('mean_score', 0):.4f} ± {cv_results.get('std_score', 0):.4f}" if cv_results.get('mean_score') else 'N/A'
                }
            
            # 模型信息
            if model:
                model_info = {
                    'model_type': type(model).__name__,
                    'parameters': {}
                }
                
                # 提取模型参数
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        # 只记录关键参数
                        key_params = ['random_state', 'n_estimators', 'max_depth', 'learning_rate', 
                                    'C', 'gamma', 'n_neighbors', 'max_iter']
                        for param in key_params:
                            if param in params:
                                model_info['parameters'][param] = params[param]
                    except:
                        pass
                
                reproducibility_info['model'] = model_info
            
            self.logger.info("Reproducibility information generated successfully")
            return reproducibility_info
            
        except Exception as e:
            self.logger.error(f"Error generating reproducibility info: {str(e)}")
            return {'error': f'Failed to generate reproducibility info: {str(e)}'}
    
    def format_reproducibility_report(self, reproducibility_info: dict) -> str:
        """格式化复现性信息为报告文本（需求F）"""
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("PROCESSING PIPELINE & REPRODUCIBILITY INFO")
            report_lines.append("=" * 60)
            
            # 基本信息
            report_lines.append(f"Generated: {reproducibility_info.get('timestamp', 'Unknown')}")
            report_lines.append(f"Random State: {reproducibility_info.get('random_state', 'Unknown')}")
            report_lines.append("")
            
            # 软件版本
            report_lines.append("Software Versions:")
            versions = reproducibility_info.get('software_versions', {})
            for software, version in versions.items():
                report_lines.append(f"  - {software}: {version}")
            report_lines.append("")
            
            # 处理流水线
            pipeline = reproducibility_info.get('processing_pipeline', {})
            if pipeline.get('steps'):
                report_lines.append("Processing Pipeline Steps:")
                for i, step in enumerate(pipeline['steps'], 1):
                    step_name = step['step'].replace('_', ' ').title()
                    report_lines.append(f"  {i}. {step_name}")
                    params = step.get('parameters', {})
                    for param_name, param_value in params.items():
                        if param_name != 'enabled' and param_value is not None:
                            report_lines.append(f"     - {param_name}: {param_value}")
                report_lines.append("")
            
            # 去峰统计（如果启用）
            if 'despiking_statistics' in reproducibility_info:
                despiking = reproducibility_info['despiking_statistics']
                if despiking.get('enabled', False):
                    report_lines.append("Spike Removal (Despiking):")
                    report_lines.append(f"  - Method: {despiking.get('method', 'mad')}")
                    report_lines.append(f"  - Window Size: {despiking.get('window', 11)}")
                    report_lines.append(f"  - Threshold: {despiking.get('threshold', 5.0)}")
                    report_lines.append("")
            
            # 特征选择
            if 'feature_selection' in reproducibility_info:
                fs_info = reproducibility_info['feature_selection']
                report_lines.append("Feature Selection:")
                report_lines.append(f"  - Method: {fs_info.get('method', 'None')}")
                report_lines.append(f"  - Features Selected: {fs_info.get('n_features_selected', 'All')}")
                report_lines.append("")
            
            # 交叉验证
            if 'cross_validation' in reproducibility_info:
                cv_info = reproducibility_info['cross_validation']
                report_lines.append("Cross-Validation Results:")
                report_lines.append(f"  - Method: {cv_info.get('method', 'Unknown')}")
                report_lines.append(f"  - Folds: {cv_info.get('folds', 5)}")
                report_lines.append(f"  - Score: {cv_info.get('score_range', 'N/A')}")
                report_lines.append("")
            
            # 模型信息
            if 'model' in reproducibility_info:
                model_info = reproducibility_info['model']
                report_lines.append("Model Configuration:")
                report_lines.append(f"  - Type: {model_info.get('model_type', 'Unknown')}")
                params = model_info.get('parameters', {})
                if params:
                    report_lines.append("  - Key Parameters:")
                    for param_name, param_value in params.items():
                        report_lines.append(f"    - {param_name}: {param_value}")
                report_lines.append("")
            
            report_lines.append("=" * 60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Error formatting reproducibility report: {str(e)}"
