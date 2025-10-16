import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataTypeFixer:
    """Utility class for fixing data type inconsistencies, especially for mixed type errors"""
    
    @staticmethod
    def fix_mixed_types(y, force_type=None):
        """
        Fix mixed type data to ensure they can be properly evaluated
        
        Args:
            y: Label data to be fixed
            force_type: Force conversion type ('float', 'int', 'auto')
            
        Returns:
            Tuple: (fixed_labels, detected_task_type)
        """
        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, list):
            y_array = np.array(y)
        else:
            y_array = np.asarray(y)
        
        if len(y_array) == 0:
            return y_array, "none"
        
        dtypes = set([type(item) for item in y_array])
        
        has_int = any(issubclass(t, (int, np.integer)) for t in dtypes)
        has_float = any(issubclass(t, (float, np.floating)) for t in dtypes)
        has_str = any(issubclass(t, str) for t in dtypes)
        has_bool = any(issubclass(t, (bool, np.bool_)) for t in dtypes)
        
        if force_type == 'float':
            conversion = 'float'
        elif force_type == 'int':
            conversion = 'int'
        elif force_type == 'auto':
            if has_str:
                conversion = 'str'  # When there are strings, convert all to strings
            elif has_float and not has_int and not has_bool:
                conversion = 'float'  # Only floats
            elif has_int and not has_float and not has_bool:
                conversion = 'int'  # Only integers
            elif has_bool and not has_int and not has_float:
                conversion = 'bool'  # Only booleans
            elif has_float:
                conversion = 'float'  # Mixed with float and other types, prioritize float
            elif has_int or has_bool:
                conversion = 'int'  # Mixed with integers and booleans, convert to integers
            else:
                conversion = 'str'  # Cannot determine, convert to string
        else:
            if has_float and has_int:
                conversion = 'float'
            elif (has_int or has_bool) and not has_float and not has_str:
                conversion = 'int'
            elif has_float:
                conversion = 'float'
            elif has_str:
                conversion = 'str'
            else:
                conversion = 'int'
        
        logger.info(f"Selected conversion type: {conversion}")
        
        try:
            if conversion == 'float':
                # **CRITICAL FIX: Safe conversion that checks for string labels first**
                def safe_float_convert(val):
                    try:
                        str_val = str(val).strip()
                        # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                        if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                            raise ValueError(f"String label detected: {str_val}")
                        return float(val)
                    except (ValueError, TypeError):
                        raise ValueError(f"Cannot convert to float: {val}")
                
                result = np.array([safe_float_convert(item) for item in y_array])
                logger.info("Successfully converted to float")
            elif conversion == 'int':
                if has_float:
                    # **CRITICAL FIX: Safe conversion for mixed float/int types**
                    def safe_round_float(val):
                        try:
                            str_val = str(val).strip()
                            # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                            if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                raise ValueError(f"String label detected: {str_val}")
                            return round(float(val))
                        except (ValueError, TypeError):
                            raise ValueError(f"Cannot convert to float: {val}")
                    
                    y_array = np.array([safe_round_float(item) for item in y_array])
                result = np.array([int(item) for item in y_array])
                logger.info("Successfully converted to integer")
            elif conversion == 'bool':
                result = np.array([bool(item) for item in y_array])
            elif conversion == 'str':
                result = np.array([str(item) for item in y_array])
                logger.info("Successfully converted to string")
            else:
                logger.warning(f"Unknown conversion type: {conversion}, returning original data")
                result = y_array
            
            return result, conversion
        except Exception as e:
            logger.error(f"Error converting data type: {e}")
            return y_array, "error"
    
    @staticmethod
    def fix_for_classification(y):
        """
        Specifically fix data types for classification tasks
        
        Args:
            y: Label data to be fixed
            
        Returns:
            Fixed labels
        """
        y_fixed, conversion = DataTypeFixer.fix_mixed_types(y, 'auto')
        
        if conversion == 'float':
            diffs = np.abs(y_fixed - np.round(y_fixed))
            max_diff = np.max(diffs)
            
            if max_diff < 1e-7:
                y_fixed = np.round(y_fixed).astype(int)
            else:
                y_fixed = np.round(y_fixed).astype(int)
        
        if conversion == 'str':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            try:
                y_fixed = encoder.fit_transform(y_fixed)
                logger.info(f"Encoding string labels to numbers, category mapping: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
            except Exception as e:
                logger.error(f"Label encoding failed: {e}")
        
        return y_fixed
    
    @staticmethod
    def fix_for_regression(y):
        """
        Specifically fix data types for regression tasks
        
        Args:
            y: Label data to be fixed
            
        Returns:
            Fixed labels
        """
        y_fixed, _ = DataTypeFixer.fix_mixed_types(y, 'float')
        return y_fixed
        
    @staticmethod
    def check_and_fix_evaluation_data(model, X, y, task_type=None):
        """
        Comprehensive check and fix for evaluation data
        
        Args:
            model: Model object
            X: Features data
            y: Labels data
            task_type: Task type ("classification" or "regression")
            
        Returns:
            Cleaned data, and task type
        """
        if task_type is None:
            model_class = model.__class__.__name__
            if any(clf in model_class for clf in ['Classifier', 'SVC', 'LogisticRegression']):
                task_type = "classification"
            elif any(reg in model_class for reg in ['Regressor', 'SVR', 'LinearRegression']):
                task_type = "regression"
            else:
                unique_values = np.unique(y)
                if len(unique_values) < 10:
                    task_type = "classification"
                else:
                    task_type = "regression"
        
        if task_type == "classification":
            y_fixed = DataTypeFixer.fix_for_classification(y)
        else:
            y_fixed = DataTypeFixer.fix_for_regression(y)
        
        return X, y_fixed, task_type 