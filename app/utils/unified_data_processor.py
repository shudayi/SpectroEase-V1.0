"""
Unified Data Processor - Solves duplicate data processing logic issues
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class UnifiedDataProcessor:
    """
    Unified Data Processor - Completely solves duplicate data processing logic issues
    
    Core principles:
    1. All labels always maintain original string format for display
    2. Use unified numeric indices for internal processing
    3. Provide unified conversion interface
    4. Eliminate string-to-float conversion issues in visualization components
    """
    
    def __init__(self):
        self.original_labels = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.task_type = None
        self.is_initialized = False
        
    def initialize_from_data(self, labels: Union[pd.Series, np.ndarray, list]) -> Dict[str, Any]:
        """
        Initialize processor from data
        
        Args:
            labels: Original label data
            
        Returns:
            Dict: Initialization result information
        """
        try:
            # 1. Standardize label format
            if isinstance(labels, pd.Series):
                labels_array = labels.values
            elif isinstance(labels, list):
                labels_array = np.array(labels)
            else:
                labels_array = np.array(labels)
            
            # 2. Ensure all labels are in string format
            self.original_labels = np.array([str(label).strip() for label in labels_array], dtype='<U50')
            
            # 3. Get unique labels
            unique_labels = np.unique(self.original_labels)
            
            # 4. Determine task type
            self.task_type = self.detect_task_type(unique_labels)
            
            # 5. Create label mapping
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
            
            self.is_initialized = True
            
            logger.info(f"✅ Unified data processor initialization completed:")
            logger.info(f"   Task type: {self.task_type}")
            logger.info(f"   Number of labels: {len(unique_labels)}")
            logger.info(f"   Number of samples: {len(self.original_labels)}")
            
            return {
                'task_type': self.task_type,
                'num_classes': len(unique_labels),
                'num_samples': len(self.original_labels),
                'unique_labels': unique_labels.tolist(),
                'label_mapping': {
                    'label_to_index': self.label_to_index,
                    'index_to_label': self.index_to_label
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Unified data processor initialization failed: {e}")
            raise
    
    def detect_task_type(self, labels=None):
        """Detect task type"""
        # Try to convert all labels to numbers
        numeric_labels = []
        has_non_numeric = False
        
        for label in labels:
            try:
                # Try to convert to float
                numeric_val = float(label)
                numeric_labels.append(numeric_val)
            except (ValueError, TypeError):
                has_non_numeric = True
                break
        
        # If there are non-numeric labels, it's definitely classification
        if has_non_numeric:
            return 'classification'
        
        # **CRITICAL FIX: More accurate numeric task judgment**
        # If all are numbers, judge based on stricter conditions
        unique_count = len(labels)
        total_samples = len(self.original_labels)
        
        # Condition 1: Very few classes (≤10) and mostly integers -> classification
        if unique_count <= 10:
            integer_count = sum(1 for val in numeric_labels if val == int(val))
            if integer_count / len(numeric_labels) > 0.8:
                return 'classification'
        
        # Condition 2: Moderate number of classes (11-20) but all small integers (0-100 range) -> classification
        elif unique_count <= 20:
            if all(val == int(val) and 0 <= val <= 100 for val in numeric_labels):
                return 'classification'
        
        # Condition 3: Many classes (>20) or contains decimals -> regression
        # Especially continuous values like octane numbers (83.4, 85.25, etc.)
        if unique_count > 20 or any(val != int(val) for val in numeric_labels):
            return 'regression'
        
        # Condition 4: Number of classes accounts for large proportion of samples (>50%) -> regression
        if unique_count / total_samples > 0.5:
            return 'regression'
            
        # Default: If uncertain, prefer classification
        return 'classification'
    
    def get_display_labels(self) -> np.ndarray:
        """
        Get labels for display (always strings)
        
        Returns:
            np.ndarray: String label array
        """
        if not self.is_initialized:
            raise ValueError("Processor not initialized")
        return self.original_labels.copy()
    
    def get_numeric_labels(self) -> np.ndarray:
        """
        Get numeric labels for machine learning
        
        Returns:
            np.ndarray: Numeric label array
        """
        if not self.is_initialized:
            raise ValueError("Processor not initialized")
        
        if self.task_type == 'classification':
            # Classification task: return label indices
            return np.array([self.label_to_index[label] for label in self.original_labels])
        else:
            # Regression task: try to convert to float
            try:
                return np.array([float(label) for label in self.original_labels])
            except ValueError as e:
                logger.error(f"Regression task label conversion failed: {e}")
                raise ValueError(f"Regression task labels cannot be converted to numbers: {e}")
    
    def convert_predictions_to_display(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert prediction results to display format
        
        Args:
            predictions: Model prediction results
            
        Returns:
            np.ndarray: Prediction results for display
        """
        if not self.is_initialized:
            raise ValueError("Processor not initialized")
        
        if self.task_type == 'classification':
            # Classification task: convert indices back to labels
            return np.array([self.index_to_label.get(int(pred), f"Unknown_{pred}") 
                           for pred in predictions], dtype='<U50')
        else:
            # Regression task: return numbers directly
            return predictions
    
    def get_safe_labels_for_visualization(self) -> np.ndarray:
        """
        Get safe labels for visualization (guaranteed not to trigger float conversion errors)
        
        Returns:
            np.ndarray: Safe string label array
        """
        if not self.is_initialized:
            raise ValueError("Processor not initialized")
        
        # **CRITICAL FIX: For regression tasks, do not add Class_ prefix**
        if self.task_type == 'regression':
            # Regression task: return original numeric labels directly, visualization component will handle correctly
            return self.original_labels.copy()
        else:
            # Classification task: ensure labels are safe string format, won't be mistaken as numbers
            safe_labels = []
            for label in self.original_labels:
                # If label looks like a number, add prefix to ensure it's treated as string
                if self._looks_like_number(label):
                    safe_labels.append(f"Class_{label}")
                else:
                    safe_labels.append(str(label))
            
            return np.array(safe_labels, dtype='<U50')
    
    def _looks_like_number(self, label: str) -> bool:
        """
        Check if label looks like a number
        
        Args:
            label: Label string
            
        Returns:
            bool: Whether it looks like a number
        """
        try:
            float(label)
            return True
        except (ValueError, TypeError):
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get processor information
        
        Returns:
            Dict: Processor information
        """
        if not self.is_initialized:
            return {'initialized': False}
        
        return {
            'initialized': True,
            'task_type': self.task_type,
            'num_classes': len(self.index_to_label),
            'num_samples': len(self.original_labels),
            'unique_labels': list(self.index_to_label.values()),
            'sample_labels': self.original_labels[:5].tolist()
        }

# Global instance
unified_processor = UnifiedDataProcessor() 