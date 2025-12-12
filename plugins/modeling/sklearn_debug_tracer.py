#!/usr/bin/env python3
"""
sklearn call tracer - for diagnosing "Mix of label input types" errors
Intercept and log detailed information before all sklearn key function calls
"""

import functools
import numpy as np
import pandas as pd
from typing import Any, Callable
import logging

  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trace_sklearn_call(func_name: str):
    """Decorator: trace sklearn function calls"""
    def decorator(original_func: Callable) -> Callable:
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            try:
                # analysis parameters
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 TRACING sklearn.{func_name} CALL")
                logger.info(f"{'='*60}")
                
                # Check positional arguments
                for i, arg in enumerate(args):
                    if hasattr(arg, '__len__') and not isinstance(arg, str):
                        try:
                            # Check data type
                            if hasattr(arg, 'dtype'):
                                logger.info(f"Arg[{i}] dtype: {arg.dtype}")
                            else:
                                logger.info(f"Arg[{i}] type: {type(arg).__name__}")
                            
                            # Check for sample values
                            if len(arg) > 0:
                                sample_values = list(arg[:5]) if hasattr(arg, '__iter__') else [arg]
                                sample_types = [type(v).__name__ for v in sample_values]
                                logger.info(f"Arg[{i}] sample values: {sample_values}")
                                logger.info(f"Arg[{i}] sample types: {sample_types}")
                                
                                # Check for mixed types
                                unique_types = set(sample_types)
                                if len(unique_types) > 1:
                                    logger.warning(f"⚠️ Mixed types detected in arg[{i}]: {unique_types}")
                        except Exception as e:
                            logger.info(f"parameters[{i}] analysis failed: {e}")
                
                # Check keyword arguments
                for key, value in kwargs.items():
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        try:
                            if hasattr(value, 'dtype'):
                                logger.info(f"Kwarg[{key}] dtype: {value.dtype}")
                            else:
                                logger.info(f"Kwarg[{key}] type: {type(value).__name__}")
                            
                            if len(value) > 0:
                                sample_values = list(value[:5]) if hasattr(value, '__iter__') else [value]
                                sample_types = [type(v).__name__ for v in sample_values]
                                logger.info(f"Kwarg[{key}] sample values: {sample_values}")
                                logger.info(f"Kwarg[{key}] sample types: {sample_types}")
                                
                                unique_types = set(sample_types)
                                if len(unique_types) > 1:
                                    logger.warning(f"⚠️ Mixed types detected in kwarg[{key}]: {unique_types}")
                        except Exception as e:
                            logger.info(f"parameter {key} analysis failed: {e}")
                
                # Execute function
                logger.info("📞 Calling original function...")
                result = original_func(*args, **kwargs)
                logger.info("✅ Function call completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"❌ Function call failed: {e}")
                logger.error(f"Function: sklearn.{func_name}")
                
                # Special handling for mixed type errors
                if "Mix of label input types" in str(e) or "string and number" in str(e):
                    logger.error("🔍 Mixed label types error detected!")
                    logger.error("This indicates inconsistent data types in labels")
                    import traceback
                    logger.error(traceback.format_exc())
                
                raise e
        return wrapper
    return decorator

def install_sklearn_tracer():
    """Install sklearn function tracer"""
    try:
        # Import necessary functions
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder, LabelBinarizer
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Store original functions
        cross_val_score_original = cross_val_score
        cross_val_predict_original = cross_val_predict
        accuracy_score_original = accuracy_score
        f1_score_original = f1_score
        precision_score_original = precision_score
        recall_score_original = recall_score
        
        # Get modules for patching
        import sklearn.model_selection
        import sklearn.metrics
        
        sklearn.model_selection.cross_val_score = trace_sklearn_call("cross_val_score")(cross_val_score_original)
        sklearn.model_selection.cross_val_predict = trace_sklearn_call("cross_val_predict")(cross_val_predict_original)
        sklearn.metrics.accuracy_score = trace_sklearn_call("accuracy_score")(accuracy_score_original)
        sklearn.metrics.f1_score = trace_sklearn_call("f1_score")(f1_score_original)
        sklearn.metrics.precision_score = trace_sklearn_call("precision_score")(precision_score_original)
        sklearn.metrics.recall_score = trace_sklearn_call("recall_score")(recall_score_original)
        
        logger.info("✅ sklearn tracer installed successfully")
        
    except Exception as e:
        logger.error(f"Failed to install sklearn tracer: {e}")

def detect_mixed_types(data, name="data"):
    """Detect mixed types in data"""
    if not hasattr(data, '__len__') or isinstance(data, str):
        return False
    
    try:
        if len(data) == 0:
            return False
            
        # Sample data for type checking
        sample_size = min(len(data), 10)
        sample_values = [data[i] for i in range(sample_size)]
        sample_types = [type(v).__name__ for v in sample_values]
        
        unique_types = set(sample_types)
        
        if len(unique_types) > 1:
            logger.warning(f"⚠️ Mixed types detected in {name}")
            logger.warning(f"Sample values: {sample_values}")
            logger.warning(f"Sample types: {sample_types}")
            logger.warning(f"Unique types: {unique_types}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Type detection failed for {name}: {e}")
        return False

if __name__ == "__main__":
    # Test installation
    install_sklearn_tracer()
    print("English text")