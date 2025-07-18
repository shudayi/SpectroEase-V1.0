# app/services/evaluation_service.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
import pandas as pd
import numpy as np
from utils.exceptions import EvaluationError
import os
import traceback
import json
from utils.logger import setup_logger
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
            
            # Initialize default metrics
            if metrics is None:
                if task_type == 'classification':
                    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                else:  # regression
                    metrics = ['mae', 'mse', 'rmse', 'r2', 'explained_variance']
            
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
            
            if 'f1' in metrics:
                # Handle warnings: target contains only one label
                unique_labels = np.unique(y_true_idx)
                if len(unique_labels) > 1:
                    results['f1'] = float(f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
                else:
                    results['f1'] = 1.0 if np.all(y_true_idx == y_pred_idx) else 0.0
            
            if 'roc_auc' in metrics:
                try:
                    # Calculating multi-class ROC AUC needs probability output, skip here
                    unique_labels = np.unique(y_true_idx)
                    if len(unique_labels) == 2:  # Binary classification
                        # Use indexed version instead of original version
                        from sklearn.metrics import roc_auc_score
                        results['roc_auc'] = float(roc_auc_score(y_true_idx, y_pred_idx))
                    else:
                        results['roc_auc'] = None
                except Exception as e:
                    self.logger.warning(f"Error calculating ROC AUC: {str(e)}")
                    results['roc_auc'] = None
            
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
