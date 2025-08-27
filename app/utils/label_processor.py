#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Label Processing Utilities - Fixed Version
Enhanced label processing that properly handles classification vs regression detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Any, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)

def safe_isna(value):
    """
    Safe check for missing/null values that works with any data type
    """
    if value is None:
        return True
    try:
        if isinstance(value, (int, float)) and np.isnan(value):
            return True
    except (ValueError, TypeError):
        pass
    try:
        if pd.isna(value):
            return True
    except (ValueError, TypeError):
        pass
    return False

class EnhancedLabelProcessor:
    """Enhanced label processor that properly handles classification vs regression tasks"""
    
    def __init__(self):
        self.label_encoder = None
        self.label_mapping = None
        self.is_string_labels = False
        self.task_type = None
        self.original_labels = None
        self.auto_detected_type = None
    
    def detect_task_type(self, labels: Union[np.ndarray, pd.Series, list]) -> str:
        """
        CRITICAL FIX: Robust task type detection with clear rules
        
        Rules:
        1. String labels (ClassA, ClassB, etc.) -> Classification
        2. Few unique numeric values (≤20 and <10% of samples) -> Classification  
        3. Integer values with ≤50 unique values -> Classification
        4. Continuous numeric values -> Regression
        
        Args:
            labels: Label data
            
        Returns:
            str: 'classification' or 'regression'
        """
        if labels is None or len(labels) == 0:
            return 'classification'
            
        # Convert to array
        if isinstance(labels, pd.Series):
            labels_array = labels.values
        elif isinstance(labels, list):
            labels_array = np.array(labels)
        else:
            labels_array = labels
            
        # RULE 1: Check for string labels (most reliable indicator)
        has_string_labels = False
        numeric_values = []
        
        for label in labels_array:
            if safe_isna(label):
                continue
                
            # Try to convert to numeric
            try:
                # Check if it's obviously a string label first
                str_label = str(label).strip()
                if any(c.isalpha() for c in str_label):
                    # Contains letters - definitely a string label
                    has_string_labels = True
                    break
                    
                numeric_val = float(label)
                numeric_values.append(numeric_val)
            except (ValueError, TypeError):
                has_string_labels = True
                break
                
        # If contains string labels -> Classification
        if has_string_labels:
            logger.info("✅ CLASSIFICATION detected: String labels found")
            self.auto_detected_type = 'classification'
            return 'classification'
            
        # If no numeric values found -> Classification (default)
        if len(numeric_values) == 0:
            logger.info("✅ CLASSIFICATION detected: No valid numeric values")
            self.auto_detected_type = 'classification'
            return 'classification'
            
        # RULE 2: Analyze numeric values
        unique_count = len(set(numeric_values))
        total_count = len(numeric_values)
        
        # Few unique values -> Classification
        if unique_count <= 20 and unique_count / total_count < 0.1:
            logger.info(f"✅ CLASSIFICATION detected: Few unique values ({unique_count}/{total_count})")
            self.auto_detected_type = 'classification'
            return 'classification'
            
        # RULE 3: Check if all values are integers with reasonable unique count
        all_integers = all(float(val).is_integer() for val in numeric_values)
        if all_integers and unique_count <= 50:
            logger.info(f"✅ CLASSIFICATION detected: Integer labels with {unique_count} classes")
            self.auto_detected_type = 'classification'
            return 'classification'
            
        # RULE 4: Otherwise -> Regression
        logger.info(f"✅ REGRESSION detected: Continuous values")
        logger.info(f"   Total samples: {total_count}")
        logger.info(f"   Unique values: {unique_count}")
        logger.info(f"   Value range: [{min(numeric_values):.3f}, {max(numeric_values):.3f}]")
        self.auto_detected_type = 'regression'
        return 'regression'
    
    def process_labels_smart(self, labels: Union[pd.Series, np.ndarray, list], 
                             force_task_type: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Smart label processing with enhanced error handling and type detection
        """
        try:
            # Convert to numpy array for consistent handling
            if isinstance(labels, pd.Series):
                labels_array = labels.values
            elif isinstance(labels, list):
                labels_array = np.array(labels)
            else:
                labels_array = np.array(labels)
            
            # **CRITICAL FIX: Clean string labels, remove leading and trailing spaces**
            # Clean string labels by stripping whitespace
            cleaned_labels = []
            for label in labels_array:
                if isinstance(label, str):
                    cleaned_labels.append(label.strip())
                else:
                    cleaned_labels.append(label)
            labels_array = np.array(cleaned_labels)
            
            # Store original labels for reference
            self.original_labels = labels_array.copy()
            
            # Detect task type
            if force_task_type:
                self.task_type = force_task_type
            else:
                self.task_type = self.detect_task_type(labels_array)
            
            self.auto_detected_type = self.detect_task_type(labels_array)
            
            logger.info(f"🎯 SPECIFIED task type: {self.task_type}")
            
            if self.task_type == 'classification':
                # Get unique labels (already cleaned)
                unique_labels = np.unique(labels_array)
                
                # Create label mappings
                self.label_mapping = {
                    'label_to_index': {str(label): idx for idx, label in enumerate(unique_labels)},
                    'index_to_label': {idx: str(label) for idx, label in enumerate(unique_labels)},
                    'unique_labels': unique_labels.tolist()
                }
                
                # Convert labels to indices for training
                processed_labels = np.array([self.label_mapping['label_to_index'][str(label)] 
                                           for label in labels_array])
                
                logger.info(f"✅ Classification labels processed: {len(unique_labels)} classes")
                logger.info(f"   Classes: {unique_labels.tolist()}")
                
                return processed_labels, self.label_mapping
                
            else:  # regression
                # Try to convert to numeric
                try:
                    processed_labels = pd.to_numeric(labels_array, errors='coerce')
                    # **CRITICAL FIX: Use numpy-compatible isnan check**
                    if np.isnan(processed_labels).any():
                        raise ValueError("Cannot convert some labels to numeric")
                    
                    # **CRITICAL FIX: Safe formatting for regression labels**
                    try:
                        min_val = float(processed_labels.min())
                        max_val = float(processed_labels.max())
                        logger.info(f"✅ Regression labels processed: range {min_val:.3f} to {max_val:.3f}")
                    except (ValueError, TypeError):
                        logger.info(f"✅ Regression labels processed: {len(processed_labels)} samples")
                    
                    # **CRITICAL FIX: Handle both pandas Series and numpy arrays**
                    if hasattr(processed_labels, 'values'):
                        result_array = processed_labels.values
                    else:
                        result_array = processed_labels
                    
                    # Generate proper metadata for regression
                    metadata = {
                        'task_type': 'regression',
                        'min_value': float(result_array.min()),
                        'max_value': float(result_array.max()),
                        'mean_value': float(result_array.mean()),
                        'std_value': float(result_array.std()),
                        'valid_samples': len(result_array),
                        'auto_detected': self.auto_detected_type
                    }
                    
                    return result_array, metadata
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Regression conversion failed: {e}, falling back to classification")
                    # Fall back to classification
                    return self.process_labels_smart(labels, 'classification')
                    
        except Exception as e:
            logger.error(f"❌ Label processing failed: {e}")
            # Emergency fallback: return labels as strings
            return np.array([str(label).strip() for label in labels]), {'task_type': 'classification'}
    
    def _process_classification_labels_enhanced(self, labels: Union[np.ndarray, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        ENHANCED classification label processing
        - Preserves original string labels
        - Creates clean integer encoding for training
        - Maintains bidirectional mapping
        """
        # Convert to array while preserving types
        if isinstance(labels, pd.Series):
            labels_array = labels.values
        elif isinstance(labels, list):
            labels_array = np.array(labels, dtype=object)
        else:
            labels_array = labels
            
        # Get unique labels (preserve original types)
        unique_labels = []
        for label in labels_array:
            if not safe_isna(label) and label not in unique_labels:
                unique_labels.append(label)
                
        # Sort consistently (by string representation)
        unique_labels = sorted(unique_labels, key=str)
        
        # Create bidirectional mapping
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        
        # Convert to integer indices for model training
        processed_labels = []
        for label in labels_array:
            if safe_isna(label):
                processed_labels.append(-1)  # Use -1 for missing values
            else:
                processed_labels.append(label_to_index[label])
                
        processed_labels = np.array(processed_labels, dtype=int)
        
        # Save mapping
        self.label_mapping = {
            'label_to_index': label_to_index,
            'index_to_label': index_to_label,
            'unique_labels': unique_labels,
            'classes': unique_labels  # sklearn compatibility
        }
        
        metadata = {
            'task_type': 'classification',
            'mapping': self.label_mapping,
            'num_classes': len(unique_labels),
            'class_names': unique_labels,
            'auto_detected': self.auto_detected_type,
            'encoding': 'label_to_index'
        }
        
        logger.info(f"✅ Classification labels processed: {len(unique_labels)} classes")
        logger.info(f"   Classes: {unique_labels}")
        
        return processed_labels, metadata
    
    def _process_regression_labels_enhanced(self, labels: Union[np.ndarray, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        ENHANCED regression label processing with validation
        - Strict numeric conversion
        - Clear error reporting for non-numeric labels
        """
        # Convert to array
        if isinstance(labels, pd.Series):
            labels_array = labels.values
        elif isinstance(labels, list):
            labels_array = np.array(labels)
        else:
            labels_array = labels
            
        # Attempt numeric conversion with detailed error tracking
        processed_labels = []
        conversion_errors = []
        
        for i, label in enumerate(labels_array):
            if safe_isna(label):
                processed_labels.append(np.nan)
                continue
                
            try:
                # Check for obvious string labels first
                str_label = str(label).strip()
                if any(c.isalpha() for c in str_label):
                    # Contains letters - definitely not numeric
                    conversion_errors.append((i, label, 'String label detected'))
                    processed_labels.append(np.nan)
                    continue
                    
                numeric_val = float(label)
                processed_labels.append(numeric_val)
                
            except (ValueError, TypeError) as e:
                conversion_errors.append((i, label, str(e)))
                processed_labels.append(np.nan)
                
        # Error analysis and reporting
        if conversion_errors:
            error_samples = [f"Index {i}: '{label}' ({reason})" for i, label, reason in conversion_errors[:5]]
            error_ratio = len(conversion_errors) / len(labels_array)
            
            logger.warning(f"⚠️  Found {len(conversion_errors)} non-numeric labels in regression task ({error_ratio:.1%})")
            logger.warning(f"   Error samples: {error_samples}")
            
            # If too many errors, this is likely a misclassified task
            if error_ratio > 0.1:  # More than 10% conversion errors
                suggested_samples = [label for _, label, _ in conversion_errors[:3]]
                logger.error("❌ TASK TYPE MISMATCH: Too many non-numeric labels for regression")
                logger.error(f"   Suggested: Switch to CLASSIFICATION task")
                logger.error(f"   Non-numeric samples: {suggested_samples}")
                
                raise ValueError(
                    f"REGRESSION task requires numeric labels, but found {len(conversion_errors)} "
                    f"non-numeric values ({error_ratio:.1%} of data). "
                    f"Sample non-numeric labels: {suggested_samples}. "
                    f"Consider using CLASSIFICATION task instead."
                )
        
        processed_labels = np.array(processed_labels, dtype=float)
        
        # Calculate statistics
        valid_values = processed_labels[~np.isnan(processed_labels)]
        
        metadata = {
            'task_type': 'regression',
            'mapping': None,
            'min_value': np.min(valid_values) if len(valid_values) > 0 else np.nan,
            'max_value': np.max(valid_values) if len(valid_values) > 0 else np.nan,
            'mean_value': np.mean(valid_values) if len(valid_values) > 0 else np.nan,
            'std_value': np.std(valid_values) if len(valid_values) > 0 else np.nan,
            'valid_samples': len(valid_values),
            'missing_samples': len(processed_labels) - len(valid_values),
            'auto_detected': self.auto_detected_type
        }
        
        # **CRITICAL FIX: Safe formatting for regression metadata**
        logger.info(f"✅ Regression labels processed:")
        try:
            if not np.isnan(metadata['min_value']) and not np.isnan(metadata['max_value']):
                logger.info(f"   Range: [{metadata['min_value']:.3f}, {metadata['max_value']:.3f}]")
            if not np.isnan(metadata['mean_value']) and not np.isnan(metadata['std_value']):
                logger.info(f"   Mean: {metadata['mean_value']:.3f} ± {metadata['std_value']:.3f}")
        except (ValueError, TypeError):
            logger.info(f"   Range: {metadata['min_value']} - {metadata['max_value']}")
        logger.info(f"   Valid samples: {metadata['valid_samples']}/{len(processed_labels)}")
        
        return processed_labels, metadata
    
    def convert_predictions_back(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert model predictions back to original label format
        """
        if self.task_type == 'classification' and self.label_mapping is not None:
            index_to_label = self.label_mapping['index_to_label']
            
            # Convert indices back to original labels
            original_predictions = []
            for pred in predictions:
                if pred == -1 or safe_isna(pred):
                    original_predictions.append(None)
                else:
                    pred_int = int(round(pred)) if not safe_isna(pred) else -1
                    original_predictions.append(index_to_label.get(pred_int, f"Unknown_Class_{pred_int}"))
                    
            return np.array(original_predictions, dtype=object)
        else:
            # Regression task - return numeric predictions as-is
            return predictions
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get comprehensive task information"""
        return {
            'task_type': self.task_type,
            'auto_detected_type': self.auto_detected_type,
            'label_mapping': self.label_mapping,
            'original_labels_sample': list(self.original_labels[:10]) if self.original_labels is not None else None,
            'num_classes': len(self.label_mapping['unique_labels']) if self.label_mapping else None
        }

# Backward compatibility
class LabelProcessor(EnhancedLabelProcessor):
    """Backward compatibility wrapper"""
    
    def process_labels(self, labels, task_type=None):
        """Legacy method for backward compatibility"""
        return self.process_labels_smart(labels, task_type)
    
    def detect_task_type(self, labels):
        """Legacy method"""
        return super().detect_task_type(labels)
