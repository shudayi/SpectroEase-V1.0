"""
Data Compatibility Utils
This module provides functions to solve common data type compatibility issues in actual data processing
"""

import numpy as np
import pandas as pd
import re
from typing import Tuple, Union, List, Dict, Any
import logging
from collections import Counter
from app.utils.label_processor import EnhancedLabelProcessor

def safe_float_convert(val):
    """Safely convert value to float, return NaN if conversion fails"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan



# Set up logging
logger = logging.getLogger(__name__)

# Global variable to ensure pd is available in all functions - not needed as pd is already imported correctly

# Unified label standardization function - ensure globally consistent label handling
def normalize_label(val):
    """
    Standardize label values to ensure consistent string representation
    
    Processing rules:
    1. None values converted to "None" string
    2. Remove leading/trailing whitespace from strings
    3. Integer-like float values (1.0) converted to integer string ("1")
    4. Decimal values rounded to two decimal places
    5. Other types converted to strings
    
    Args:
        val: Label value of any type
        
    Returns:
        str: Standardized string representation
    """
    if val is None:
        return "None"
    
    # Remove leading/trailing whitespace
    if isinstance(val, str):
        val = val.strip()
    
    # Process numeric labels - ensure consistent integer representation (1.0 -> "1")
    try:
        num = float(val)
        if num.is_integer():
            return str(int(num))  # Ensure integer values are consistent, e.g., 1.0 -> "1"
        else:
            return f"{num:.2f}"  # Keep two decimal places, ensuring consistent float representation
    except (ValueError, TypeError):
        # Non-numeric, return string representation
        return str(val).strip()

def standardize_classification_labels(
    y_true: Union[np.ndarray, pd.Series, List], 
    y_pred: Union[np.ndarray, pd.Series, List]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Standardize both true labels and predictions for classification tasks.
    This function ensures both arrays use the same encoding, handles mixed types,
    and converts everything to a consistent format for evaluation.
    
    Args:
        y_true: True labels (can be any type)
        y_pred: Predicted labels (can be any type)
        
    Returns:
        Tuple of:
            y_true_encoded: Encoded true labels
            y_pred_encoded: Encoded predictions
            label_map: Mapping from original labels to encoded values
    """
    global pd
    
    # Convert pandas to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    logger.info("Starting classification label standardization")

    # Ensure input is numpy array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Ensure all values are string type first to avoid mixed type issues
    # This helps with float vs int comparisons and other mixed type issues
    if np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number):
        # If both are numeric types, try to convert them to the same numeric type
        try:
            # **CRITICAL FIX: Check if contains string labels, if so skip numeric conversion**
            # First check if original data contains string labels
            has_string_labels = False
            try:
                # Check first few samples for string labels
                for sample in list(y_true[:5]) + list(y_pred[:5]):
                    str_sample = str(sample).strip()
                    # If contains letters or special characters (non-numeric characters), it's considered a string label
                    if any(c.isalpha() for c in str_sample):
                        has_string_labels = True
                        break
            except:
                pass
            
            if has_string_labels:
                # Contains string labels, skip numeric conversion
                logger.info("Detected string labels, skipping numeric conversion check")
                raise ValueError("Contains string labels, skip numeric conversion")
            
            # **CRITICAL FIX: Check for string labels before pd.to_numeric**
            # Check y_true for string labels
            sample_true = [str(label) for label in y_true[:5]]
            has_string_true = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit() 
                                 for label in sample_true if label.strip())
            
            # Check y_pred for string labels  
            sample_pred = [str(label) for label in y_pred[:5]]
            has_string_pred = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit() 
                                 for label in sample_pred if label.strip())
            
            if has_string_true or has_string_pred:
                print(f"🔧 String labels detected - True: {sample_true[:3]}, Pred: {sample_pred[:3]}")
                print("🔧 Skipping numeric conversion, using string processing instead")
                raise ValueError(f"String labels detected - True: {sample_true[:3]}, Pred: {sample_pred[:3]}")
            
            # Try to preserve numeric properties while ensuring consistent types
            y_true = pd.to_numeric(y_true, errors='coerce')  # Use coerce to handle string labels
            y_pred = pd.to_numeric(y_pred, errors='coerce')
            
            # Check if conversion failed (indicates string labels)
            if pd.isna(y_true).any() or pd.isna(y_pred).any():
                raise ValueError("Cannot convert string labels to numeric for regression metrics. Use classification metrics instead.")
            
            # **CRITICAL FIX: Safe integer conversion that checks for string labels**
            try:
                # Only convert to integer if data is actually numeric
                if np.all(np.equal(np.mod(y_true, 1), 0)) and np.all(np.equal(np.mod(y_pred, 1), 0)):
                    y_true = y_true.astype(int)
                    y_pred = y_pred.astype(int)
            except (TypeError, ValueError) as e:
                # If conversion fails due to string labels, keep as is
                print(f"Integer conversion skipped due to string labels: {e}")
        except:
            # If conversion fails, fall back to string processing
            y_true = np.array([str(label) if label is not None else "None" for label in y_true])
            y_pred = np.array([str(label) if label is not None else "None" for label in y_pred])
    else:
        # For mixed types, standardize to string format
        y_true = np.array([str(label) if label is not None else "None" for label in y_true])
        y_pred = np.array([str(label) if label is not None else "None" for label in y_pred])
    
    # Use global normalization function to process all labels
    y_true_str = np.array([normalize_label(label) for label in y_true])
    y_pred_str = np.array([normalize_label(label) for label in y_pred])
    
    # Record label conversion process
    if len(y_true) > 0:
        before_after = []
        for i in range(min(5, len(y_true))):
            before_after.append(f"'{y_true[i]}' → '{y_true_str[i]}'")
        logger.debug(f"Label standardization examples: {', '.join(before_after)}")
    
    logger.debug(f"Label examples - True: {y_true_str[:5] if len(y_true_str) > 5 else y_true_str}")
    logger.debug(f"Label examples - Predicted: {y_pred_str[:5] if len(y_pred_str) > 5 else y_pred_str}")

    # Get all unique labels
    all_labels = np.unique(np.concatenate([y_true_str, y_pred_str]))
    label_count = len(all_labels)
    logger.info(f"Detected {label_count} unique labels")
    
    # Record unique labels (for debugging)
    if label_count <= 20:
        logger.debug(f"All unique labels: {list(all_labels)}")
    else:
        logger.debug(f"First 20 unique labels: {list(all_labels[:20])}...")
    
    # If too many labels, might be due to float precision issues or data errors
    if label_count > 100:
        logger.warning(f"Warning: Detected {label_count} labels, more than 100. Attempting to further merge approximate values...")
        
        # Try to further merge numeric labels
        try:
            # Extract all potentially numeric labels
            numeric_labels = []
            for label in all_labels:
                try:
                    numeric_labels.append(float(label))
                except ValueError:
                    pass
            
            if len(numeric_labels) > 50:  # If there are indeed many numeric labels
                # Use K-means clustering to group numeric labels
                from sklearn.cluster import KMeans
                # No need to re-import these as they're already imported at the module level
                
                # Convert numeric labels to 2D array
                numeric_array = np.array(numeric_labels).reshape(-1, 1)
                
                # Choose appropriate number of clusters, max 20 categories
                n_clusters = min(20, len(numeric_labels) // 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(numeric_array)
                
                # Create mapping from category to representative value
                cluster_centers = kmeans.cluster_centers_.flatten()
                cluster_mapping = {}
                for i, cluster_id in enumerate(clusters):
                    numeric_value = numeric_labels[i]
                    representative = cluster_centers[cluster_id]
                    cluster_mapping[str(numeric_value)] = str(round(representative, 2))
                
                # Apply clustering results to remap labels
                remapped_true = np.array([cluster_mapping.get(label, label) for label in y_true_str])
                remapped_pred = np.array([cluster_mapping.get(label, label) for label in y_pred_str])
                
                # Use remapped labels
                y_true_str = remapped_true
                y_pred_str = remapped_pred
                
                # Update unique labels set
                all_labels = np.unique(np.concatenate([y_true_str, y_pred_str]))
                logger.info(f"Labels reduced to {len(all_labels)} unique labels after clustering")
        except Exception as e:
            logger.error(f"Label merging failed: {e}")
    
    # Create label to index mapping
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    logger.debug(f"Label mapping: {label_map}")
    
    # Convert labels to indices
    y_true_idx = np.array([label_map[label] for label in y_true_str])
    y_pred_idx = np.array([label_map[label] for label in y_pred_str])

    logger.info("Label standardization completed")
    return y_true_idx, y_pred_idx, label_map

def standardize_regression_values(
    y_true: Union[np.ndarray, pd.Series, List], 
    y_pred: Union[np.ndarray, pd.Series, List] = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Standardize true values and predictions for regression tasks.
    This function converts both arrays to float type and handles missing values.
    
    Args:
        y_true: True values (can be numeric or string representation of numbers)
        y_pred: Predicted values (can be numeric or string representation of numbers)
        
    Returns:
        If y_pred is provided:
            y_true_float: True values as float array
            y_pred_float: Predicted values as float array
            y_mask: Boolean mask of valid entries (where neither is NaN)
        Otherwise:
            y_true_float: True values as float array
            y_mask: Boolean mask of valid entries (where not NaN)
    """
    global pd
    
    # Convert pandas to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if y_pred is not None and isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Convert to float
    y_true_float = np.array([extract_number(val) for val in y_true])
    
    # Single parameter mode
    if y_pred is None:
        # Create valid data mask
        mask = ~np.isnan(y_true_float)
        return y_true_float, mask
    
    # Dual parameter mode
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    elif not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Convert to float
    y_pred_float = np.array([extract_number(val) for val in y_pred])
    
    # Create valid data mask
    mask = ~np.isnan(y_true_float) & ~np.isnan(y_pred_float)
    
    return y_true_float, y_pred_float, mask

def extract_number(val: Any) -> float:
    """
    Extract number from any value
    
    Args:
        val: Any input value
        
    Returns:
        float: Extracted number, returns NaN if extraction fails
    """
    if val is None:
        return np.nan
    
    try:
        # Already a numeric type
        if isinstance(val, (int, float)):
            return float(val)
        
        # Try direct conversion
        return float(val)
    except (ValueError, TypeError):
        # Try to extract number from string
        try:
            match = re.search(r'[-+]?[0-9]*\.?[0-9]+', str(val))
            return float(match.group()) if match else np.nan
        except:
            return np.nan

def is_classification_task(y: Union[np.ndarray, pd.Series, List]) -> bool:
    """
    Smartly determine if a task is classification
    
    Args:
        y: Target variable
        
    Returns:
        bool: True if classification task, False if regression task
    """
    global pd
    
    if isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, pd.DataFrame):
        y = y.values.flatten()
    
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    n_samples = len(y)
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    unique_ratio = n_unique / n_samples
    
    logger.info(f"Task type detection - samples: {n_samples}, unique values: {n_unique}, ratio: {unique_ratio:.3f}")
    
    # Rule 1: If unique value count is very small, likely classification
    if n_unique <= 10:
        logger.info(f"Small unique value count ({n_unique}≤10), determined as classification task")
        return True
    
    # Rule 2: If unique value ratio is very low, might be classification
    if unique_ratio <= 0.1:  # 10% threshold
        logger.info(f"Low unique value ratio ({unique_ratio:.3f}≤0.1), determined as classification task")
        return True
    
    # Rule 3: Check data type
    try:
        # Safely check data type, avoid forced conversion of string labels
        def safe_numeric_check(labels):
            """Safely check if labels are numeric type"""
            try:
                # Check if first few samples are obvious string labels
                sample_labels = [str(label) for label in labels[:10]]
                string_label_count = sum(1 for label in sample_labels 
                                       if not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit())
                
                if string_label_count > len(sample_labels) * 0.5:
                    # More than half are obvious string labels, skip numeric conversion
                    logger.info("Detected string labels, skipping numeric conversion check")
                    return None, 1.0  # Return None and high NaN ratio to indicate non-numeric
                
                # Try numeric conversion
                y_numeric = pd.to_numeric(labels, errors='coerce')
                nan_ratio = np.isnan(y_numeric).sum() / len(y_numeric)
                return y_numeric, nan_ratio
            except Exception as e:
                logger.warning(f"Numeric check failed: {e}")
                return None, 1.0
        
        y_numeric, nan_ratio = safe_numeric_check(y)
        
        if nan_ratio > 0.5:
            # Over 50% cannot be converted to numeric, might be text labels
            logger.info(f"Most data cannot be converted to numeric ({nan_ratio:.3f}>0.5), determined as classification task")
            return True
        
        # Rule 4: Check if integer values (only if numeric conversion succeeded)
        if y_numeric is not None:
            valid_numeric = y_numeric[~np.isnan(y_numeric)]
            if len(valid_numeric) > 0 and np.all(valid_numeric == np.round(valid_numeric)):
                if n_unique <= 20:
                    logger.info("All values are integers with few categories, determined as classification task")
                    return True
        
        # Rule 5: If unique value ratio is too high, might be regression
        if unique_ratio > 0.5:  # 50% threshold
            logger.info(f"High unique value ratio ({unique_ratio:.3f}>0.5), determined as regression task")
            return False
            
    except Exception as e:
        logger.warning(f"Numeric check failed: {e}, defaulting to classification task")
        return True
    
    # Rule 6: Final judgment based on unique value count
    if n_unique > 50:
        logger.info(f"Large unique value count ({n_unique}>50), determined as regression task")
        return False
    else:
        logger.info(f"Based on comprehensive judgment, determined as classification task")
        return True

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by handling missing values, converting types,
    and preparing it for machine learning
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    global pd
    
    if df is None or df.empty:
        return df
        
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
            
        # Replace NaN with column median
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # **CRITICAL FIX: Only convert string columns that are clearly numeric**
    # Skip columns that might be labels (like 'Label', 'Class', etc.)
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]):
            try:
                # Skip if column name suggests it's a label column
                col_name_lower = str(col).lower()
                if any(keyword in col_name_lower for keyword in ['label', 'class', 'category', 'target', 'variety', 'type', 'group']):
                    logger.debug(f"Skipping label column in clean_dataframe: {col}")
                    continue
                
                # Check if column contains numeric strings
                numeric_values = df_clean[col].apply(lambda x: extract_number(x))
                if numeric_values.isna().mean() < 0.5:  # If most can be converted
                    df_clean[col] = numeric_values
            except Exception as e:
                logger.debug(f"Skipping column {col} conversion in clean_dataframe: {e}")
                pass
    
    return df_clean

def prepare_data_for_training(
    X: pd.DataFrame, 
    y: Union[pd.Series, np.ndarray]
) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
    """
    Prepare data for model training by handling missing values,
    converting types, and checking for data quality issues
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]: Processed X and y
    """
    global pd
    
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        try:
            # Convert to DataFrame
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        except Exception as e:
            print(f"Error converting features to DataFrame: {e}")
            raise
    
    # Clean feature matrix
    X_clean = clean_dataframe(X)
    
    # Handle target variable based on task type
    is_classification = is_classification_task(y)
    
    if is_classification:
        # For classification, ensure labels are appropriate format
        if isinstance(y, pd.Series):
            # Keep as Series but ensure categoricals are properly encoded
            if pd.api.types.is_categorical_dtype(y):
                y = y.cat.codes
        elif isinstance(y, np.ndarray):
            # If numpy array, check if string conversion needed
            if y.dtype.kind not in 'iuf':  # not int, uint or float
                y = np.array([str(val) if val is not None else "None" for val in y])
    else:
        # For regression, try to convert to numeric
        # **CRITICAL FIX: Only convert if it's actually a regression task**
        if isinstance(y, pd.Series):
            # Check if it's really numeric data before conversion
            sample_values = y.head(5).tolist()
            string_count = sum(1 for val in sample_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())
            if string_count == 0:  # Only convert if no obvious string labels
                y = y.apply(lambda x: extract_number(x))
            else:
                logger.warning(f"Detected string labels in regression task, keeping as strings: {sample_values}")
        elif isinstance(y, np.ndarray):
            # Check if it's really numeric data before conversion
            sample_values = y[:5].tolist()
            string_count = sum(1 for val in sample_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())
            if string_count == 0:  # Only convert if no obvious string labels
                y = np.array([extract_number(val) for val in y])
            else:
                logger.warning(f"Detected string labels in regression task, keeping as strings: {sample_values}")
    
    return X_clean, y

def prepare_labels_safely(
    y: Union[np.ndarray, pd.Series, List], 
    task_type: str = None
) -> Tuple[np.ndarray, str]:
    """
    Process labels in the safest way possible, standardizing regardless of input type
    
    This is a one-stop solution for handling various mixed-type label issues
    
    Args:
        y: Input label array (any type)
        task_type: Task type, 'classification' or 'regression', auto-detected if None
        
    Returns:
        Tuple[np.ndarray, str]: (Processed labels, actual task type used)
    """
    global pd
    
    # **CRITICAL FIX: IMMEDIATE FORCED TYPE STANDARDIZATION**
    # Before any other processing, force ALL values to string to eliminate mixed types
  
    
    # Convert to numpy array first
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y_raw = y.values
    elif not isinstance(y, np.ndarray):
        try:
            y_raw = np.array(y)
        except:
            y_raw = np.array([str(val) if val is not None else "None" for val in y])
    else:
        y_raw = y
    
    # FORCED STRING CONVERSION - NO EXCEPTIONS
  
  
    
    # Convert EVERY value to string, handling all edge cases
    y_strings = []
    for val in y_raw:
        if val is None:
            y_strings.append("None")
        elif isinstance(val, (int, np.integer)):
            y_strings.append(str(int(val)))
        elif isinstance(val, (float, np.floating)):
            if np.isnan(val):
                y_strings.append("NaN")
            elif val == int(val):  # Float value that is an integer
                y_strings.append(str(int(val)))
            else:
                y_strings.append(f"{val:.2f}")
        elif isinstance(val, bool):
            y_strings.append(str(val))
        else:
            y_strings.append(str(val).strip())
    
    # Convert to numpy array of strings
    y = np.array(y_strings, dtype=str)
    
  
  
  
    
    # Verify no mixed types remain
    sample_types = set(type(v).__name__ for v in y[:10])
    if len(sample_types) > 1:
  
        # Final emergency conversion
        y = np.array([str(v) for v in y], dtype=str)
  
    
  
    
    # Check and record original label types
    value_types = set()
    type_counts = {}
    for val in y:
        val_type = "None" if val is None else type(val).__name__
        value_types.add(val_type)
        type_counts[val_type] = type_counts.get(val_type, 0) + 1
    
    logger.debug(f"prepare_labels_safely - Input label types: {value_types}")
    logger.debug(f"prepare_labels_safely - Type counts: {type_counts}")
    
    # Show examples of each type
    type_examples = {}
    for val in y:
        val_type = "None" if val is None else type(val).__name__
        if val_type not in type_examples and val is not None:
            type_examples[val_type] = str(val)
    
    logger.debug(f"prepare_labels_safely - Type examples: {type_examples}")
    
    # Handle None value ratio
    none_count = sum(1 for val in y if val is None)
    none_ratio = none_count / len(y) if len(y) > 0 else 0
    
    # Special handling if None ratio is high
    if none_ratio > 0.05:  # If None values exceed 5%
        logger.warning(f"Labels contain many None values ({none_ratio:.1%}), may affect model performance")
        
    # Auto-detect task type if not specified
    if task_type is None:
        # After standardization, check for task type
        unique_labels = np.unique(y)
        unique_count = len(unique_labels)
        
        # Classification if few unique labels or obviously categorical
        if unique_count <= 50:
            task_type = 'classification'
        else:
            # Try to determine if labels are numeric after string conversion
            try:
                # **CRITICAL FIX: Safe conversion that checks for string labels first**
                def safe_sample_convert(label):
                    try:
                        str_label = str(label).strip()
                        # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                        if str_label and not str_label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                            raise ValueError(f"String label detected: {str_label}")
                        return float(label)
                    except (ValueError, TypeError):
                        raise ValueError(f"Cannot convert to float: {label}")
                
                # Try to convert a sample back to numeric
                sample_numeric = [safe_sample_convert(label) for label in unique_labels[:10]]
                # If all values are integers, likely classification
                if all(val == int(val) for val in sample_numeric):
                    task_type = 'classification'
                else:
                    task_type = 'regression'
            except ValueError:
                # Contains non-numeric strings, definitely classification
                task_type = 'classification'
    
    logger.debug(f"prepare_labels_safely - Detected task type: {task_type}")
    
    if task_type == 'classification':
        # For classification: further normalize string labels
        logger.debug(f"prepare_labels_safely - Starting classification label processing")
        
        # Use normalize_label for each value to ensure consistency
        y_processed = np.array([normalize_label(label) for label in y])
        
        # Get unique normalized labels
        unique_normalized = np.unique(y_processed)
        
        logger.debug(f"prepare_labels_safely - Unique normalized labels: {len(unique_normalized)}")
        
        # Show conversion examples
        conversion_samples = []
        for i in range(min(5, len(y))):
            if y[i] != y_processed[i]:
                conversion_samples.append(f"'{y[i]}' → '{y_processed[i]}'")
        
        if conversion_samples:
            logger.debug(f"prepare_labels_safely - Label standardization examples:\n  " + "\n  ".join(conversion_samples))
        
        # Check for potential data issues
        if len(unique_normalized) > 100:
            logger.warning(f"Large number of unique labels: {len(unique_normalized)}. Check for data issues.")
        
        # Check for numeric types, warn if risk exists
        has_numeric = any(t in value_types for t in ['float', 'float64', 'int', 'int64'])
        if has_numeric and len(value_types) > 1:
            logger.warning(f"Detected mixed type labels processed as classification task, contains: {value_types}")
        
        return y_processed, 'classification'
        
    else:
        # Regression task: try to convert to float
        logger.debug(f"prepare_labels_safely - Starting regression label processing")
        
        try:
            y_numeric = []
            conversion_failures = 0
            
            for label in y:
                try:
                    # **CRITICAL FIX: Safe conversion that checks for string labels first**
                    str_label = str(label).strip()
                    # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                    if str_label and not str_label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                        raise ValueError(f"String label detected in regression: {str_label}")
                    
                    # Try direct float conversion
                    numeric_val = float(label)
                    y_numeric.append(numeric_val)
                except (ValueError, TypeError):
                    # Handle special string cases
                    str_label = str(label).strip().lower()
                    if str_label in ['none', 'null', 'nan', '']:
                        y_numeric.append(np.nan)
                    else:
                        # Try to extract numeric from string
                        import re
                        numeric_match = re.search(r'[-+]?\d*\.?\d+', str_label)
                        if numeric_match:
                            y_numeric.append(float(numeric_match.group()))
                        else:
                            y_numeric.append(np.nan)
                            conversion_failures += 1
            
            y_processed = np.array(y_numeric, dtype=float)
            
            # Log conversion statistics
            nan_count = np.isnan(y_processed).sum()
            logger.debug(f"prepare_labels_safely - Regression conversion results:")
            logger.debug(f"  - Conversion failures: {conversion_failures}")
            logger.debug(f"  - NaN values: {nan_count} ({nan_count/len(y_processed):.1%})")
            logger.debug(f"  - Value range: {np.nanmin(y_processed):.4f} to {np.nanmax(y_processed):.4f}")
            
            if conversion_failures > len(y) * 0.1:  # More than 10% failed
                logger.warning(f"High conversion failure rate ({conversion_failures/len(y):.1%}), consider classification instead")
            
            return y_processed, 'regression'
            
        except Exception as e:
            logger.warning(f"Regression label conversion failed: {str(e)}, falling back to classification")
            # Fall back to classification
            y_processed = np.array([normalize_label(label) for label in y])
            return y_processed, 'classification'

def encode_classification_labels(
    labels: Union[np.ndarray, pd.Series, List], 
    return_mapping: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, Dict, Dict]]:
    """
    Encode classification labels as integers (0 to N-1) for classification tasks.
    This function should be called early in the data processing pipeline to ensure
    all subsequent steps use the same integer encoding.

    Args:
        labels: Original label array (can be any type)
        return_mapping: Whether to return label mapping dictionaries

    Returns:
        If return_mapping=True:
            labels_encoded: Encoded integer labels
            forward_map: Mapping from original labels to encoding {original: int_code}
            inverse_map: Mapping from encoding to original labels {int_code: original}
        Otherwise:
            labels_encoded: Encoded integer labels
    """
    global pd
    
    logger.info("Encoding classification labels to integers")

    # Ensure input is numpy array
    if isinstance(labels, pd.Series):
        labels = labels.values
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Ensure all labels can be safely converted to string to prevent type mixing issues
    original_labels = labels.copy()  # Save original labels for detailed logging
    
    # If numeric type, check if all are integers
    if np.issubdtype(labels.dtype, np.number):
        try:
            # Check if all are integers
            if np.all(np.equal(np.mod(labels, 1), 0)):
                # **CRITICAL FIX: Safe integer conversion that checks for string labels**
                try:
                    # Check if labels contain string data before conversion
                    if labels.dtype.kind in ['U', 'S', 'O']:
                        # Contains string data, skip integer conversion
                        raise ValueError("Contains string labels, cannot convert to int")
                    labels_int = labels.astype(int)
                except (ValueError, TypeError):
                    # If conversion fails, fall back to string processing
                    raise ValueError("Integer conversion failed, falling back to string processing")
                unique_labels = np.unique(labels_int)
                forward_map = {int(label): idx for idx, label in enumerate(unique_labels)}
                inverse_map = {idx: int(label) for idx, label in enumerate(unique_labels)}
                labels_encoded = np.array([forward_map[int(label)] for label in labels_int])
                
                logger.info(f"Direct integer encoding used - {len(unique_labels)} unique integer labels")
                
                if return_mapping:
                    return labels_encoded, forward_map, inverse_map
                else:
                    return labels_encoded
        except Exception as e:
            logger.debug(f"Integer direct encoding failed, falling back to string conversion: {e}")
            # Continue with standard string conversion process
    
    # Apply global normalization
    labels_str = np.array([normalize_label(label) for label in labels])
    
    # Record conversion examples
    if len(labels) > 0:
        logger.debug("Label conversion examples:")
        for i in range(min(5, len(labels))):
            logger.debug(f"  Original: '{original_labels[i]}' ({type(original_labels[i]).__name__}) -> Normalized: '{labels_str[i]}' ({type(labels_str[i]).__name__})")
    
    # Get unique labels and create mapping
    unique_labels = np.unique(labels_str)
    logger.info(f"Encoding classification labels - Detected {len(unique_labels)} unique classification labels")
    
    # Record all unique labels (for debugging)
    if len(unique_labels) <= 20:
        logger.debug(f"All unique labels: {list(unique_labels)}")
    else:
        logger.debug(f"First 20 unique labels: {list(unique_labels[:20])}...")
    
    # Create bidirectional mapping dictionaries
    forward_map = {label: idx for idx, label in enumerate(unique_labels)}
    inverse_map = {idx: label for idx, label in enumerate(unique_labels)}
    
    # Convert labels to integer encoding
    labels_encoded = np.array([forward_map[label] for label in labels_str])
    
    # Record conversion examples
    if len(labels) > 0:
        logger.debug("Label encoding examples:")
        for i in range(min(5, len(labels))):
            logger.debug(f"  Normalized: '{labels_str[i]}' -> Encoded: {labels_encoded[i]} (Mapping back: '{inverse_map[labels_encoded[i]]}')")
    
    # Calculate label counts
    label_counts = Counter(labels_encoded)
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"Label encoding counts (top 10): {sorted_counts[:10]}")
    
    # Log some information for debugging
    logger.debug(f"Label mapping dictionary: {forward_map}")
    logger.debug(f"Label examples - Original: {labels[:5] if len(labels) > 5 else labels}")
    logger.debug(f"Label examples - Encoded: {labels_encoded[:5] if len(labels_encoded) > 5 else labels_encoded}")
    
    if return_mapping:
        return labels_encoded, forward_map, inverse_map
    else:
        return labels_encoded

def predict_with_encoded_labels(
    model,
    X: np.ndarray, 
    inverse_map: Dict
) -> np.ndarray:
    """
    Use pre-built label mapping to ensure prediction results match original label types.

    Args:
        model: Trained model
        X: Feature matrix
        inverse_map: Mapping dictionary from integer encoding to original labels

    Returns:
        Predictions in original label type
    """
    # Get model predictions (integer encoded)
    predictions = model.predict(X)
    
    # **CRITICAL FIX: Safe conversion to integers (in case they're floats)**
    if hasattr(predictions, 'dtype') and np.issubdtype(predictions.dtype, np.floating):
        try:
            # Check if predictions contain string labels before conversion
            if predictions.dtype.kind in ['U', 'S', 'O']:
                # Contains string data, skip integer conversion
                pass
            else:
                predictions = np.round(predictions).astype(int)
        except (ValueError, TypeError):
            # If conversion fails due to string labels, keep as is
            pass
    
    # Use inverse mapping to convert back to original labels
    # **CRITICAL FIX: Safe conversion that handles string labels**
    def safe_int_convert_for_mapping(val):
        """Safely convert value to int for mapping lookup"""
        try:
            str_val = str(val).strip()
            # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
            if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                return None  # Return None for string labels
            return int(float(val))
        except (ValueError, TypeError):
            return None
    
    original_predictions = []
    for p in predictions:
        int_p = safe_int_convert_for_mapping(p)
        if int_p is not None:
            # Use mapping for integer indices
            original_predictions.append(inverse_map.get(int_p, "Unknown"))
        else:
            # For string labels, use as-is
            original_predictions.append(str(p))
    
    return np.array(original_predictions)

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate classification model performance, safely handling mixed types.

    Args:
        y_true: True labels (can be any type)
        y_pred: Predicted labels (can be any type)

    Returns:
        Dictionary with various evaluation metrics
    """
    global pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import sklearn
    
    logger.info(f"Evaluating classification performance - y_true type: {type(y_true)}, dtype: {getattr(y_true, 'dtype', 'N/A')}")
    logger.info(f"Evaluating classification performance - y_pred type: {type(y_pred)}, dtype: {getattr(y_pred, 'dtype', 'N/A')}")
    
    # Check for mixed type issues, resolve type inconsistency if exists
    try:
        if hasattr(y_true, 'dtype') and hasattr(y_pred, 'dtype') and y_true.dtype != y_pred.dtype:
            logger.warning(f"Label types inconsistent: y_true={y_true.dtype}, y_pred={y_pred.dtype}")
            
            # Check if string and numeric mixed
            if y_true.dtype.kind in 'iuf' and y_pred.dtype.kind == 'U':
                # Predictions are strings, true labels are numbers, try to convert predictions to numbers
                logger.info("Attempting to convert string predictions to numbers...")
                try:
                    # **CRITICAL FIX: Safe conversion that checks for string labels first**
                    def safe_convert_to_int(val):
                        try:
                            str_val = str(val).strip()
                            # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                            if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                return None  # Return None for string labels
                            # **FIXED: Safe float conversion**
                            return int(safe_float_convert(val))
                        except (ValueError, TypeError):
                            return None
                    
                    converted_pred = [safe_convert_to_int(x) for x in y_pred]
                    if None in converted_pred:
                        # Contains string labels, cannot convert to numbers
                        raise ValueError("Contains string labels that cannot be converted to numbers")
                    
                    y_pred = np.array(converted_pred)
                    logger.info(f"Successfully converted predictions to numbers: {y_pred.dtype}")
                except Exception as e:
                    logger.warning(f"Cannot convert predictions to numbers: {str(e)}")
                    # Convert true labels to strings instead
                    y_true = np.array([str(x) for x in y_true])
                    logger.info(f"Converting true labels to strings instead: {y_true.dtype}")
            
            elif y_pred.dtype.kind in 'iuf' and y_true.dtype.kind == 'U':
                # True labels are strings, predictions are numbers
                logger.info("Attempting to convert numeric predictions to strings...")
                try:
                    y_pred = np.array([str(x) for x in y_pred])
                    logger.info(f"Successfully converted predictions to strings: {y_pred.dtype}")
                except Exception as e:
                    logger.warning(f"Cannot convert predictions to strings: {str(e)}")
                    # Try to convert true labels to numbers
                    try:
                        # **CRITICAL FIX: Safe conversion that checks for string labels first**
                        def safe_convert_to_int(val):
                            try:
                                str_val = str(val).strip()
                                # Check if it's a string label like 'ClassA', 'ClassB', 'ClassC'
                                if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                    return None  # Return None for string labels
                                # **FIXED: Safe float conversion**
                                return int(safe_float_convert(val))
                            except (ValueError, TypeError):
                                return None
                        
                        converted_true = [safe_convert_to_int(x) for x in y_true]
                        if None in converted_true:
                            # Contains string labels, cannot convert to numbers
                            raise ValueError("Contains string labels that cannot be converted to numbers")
                        
                        y_true = np.array(converted_true)
                        logger.info(f"Converting true labels to numbers instead: {y_true.dtype}")
                    except Exception as e2:
                        logger.error(f"Both conversions failed, cannot resolve type inconsistency: {str(e2)}")
    except Exception as e:
        logger.warning(f"Error while resolving type inconsistency: {str(e)}")
    
    # First try to calculate metrics directly (if types are compatible)
    try:
        # If both are already integers, can calculate directly
        if np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer):
            logger.debug("Both arrays are integer type, using direct calculation")
            
            # Get number of unique classes
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            num_classes = len(unique_classes)
            
            # Calculate metrics directly
            metrics = {}
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            # Multi-class metrics
            if num_classes > 1:
                metrics["f1"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            else:
                # Single class case
                metrics["f1"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
                metrics["precision"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
                metrics["recall"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
                
            return metrics
    except Exception as e:
        logger.debug(f"Direct metric calculation failed, falling back to standardization: {e}")
    
    # If direct calculation fails, use standardization method
    # Convert labels to string to ensure consistency
    logger.debug("Converting labels to normalized format for evaluation")
    
    # Ensure safe conversion to string
    if not np.issubdtype(y_true.dtype, np.number):
        y_true_str = np.array([str(label) if label is not None else "None" for label in y_true])
    else:
        y_true_str = y_true
        
    if not np.issubdtype(y_pred.dtype, np.number):
        y_pred_str = np.array([str(label) if label is not None else "None" for label in y_pred])
    else:
        y_pred_str = y_pred
        
    # Get standardized integer indices
    y_true_idx, y_pred_idx, _ = standardize_classification_labels(y_true_str, y_pred_str)
    
    # Log the type conversion results for debugging
    logger.debug(f"Label types after standardization - True: {y_true_idx.dtype}, Pred: {y_pred_idx.dtype}")
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true_idx, y_pred_idx]))
    num_classes = len(unique_classes)
    
    # Calculate evaluation metrics
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true_idx, y_pred_idx))
    
    # Multi-class metrics
    if num_classes > 1:
        metrics["f1"] = float(f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
        metrics["precision"] = float(precision_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
        metrics["recall"] = float(recall_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0))
    else:
        # Single class case
        metrics["f1"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
        metrics["precision"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
        metrics["recall"] = 1.0 if metrics["accuracy"] == 1.0 else 0.0
    
    return metrics 