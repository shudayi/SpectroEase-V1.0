import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from typing import Dict, Any, List, Optional
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
)
from app.utils.data_compatibility import (
    standardize_classification_labels,
    prepare_labels_safely, 
    encode_classification_labels,
    predict_with_encoded_labels,
    normalize_label,
    evaluate_classification
)
from app.utils.label_processor import EnhancedLabelProcessor
import logging
import traceback
from collections import Counter

  
try:
    import sys
    sys.path.append('.')
    from plugins.modeling.sklearn_debug_tracer import install_sklearn_tracers, detect_mixed_types
    install_sklearn_tracers()
    # Debug tracer activated
except ImportError as e:
    print("English text")
    def detect_mixed_types(data, name="data"):
        return False

# Setup logging
logger = logging.getLogger(__name__)

class QualitativeAnalyzer:
    """Qualitative Analysis Class"""
    
    def __init__(self, method: str = 'lda'):
        """
        Initialize qualitative analyzer
        
        Args:
            method: Analysis method ('kmeans', 'hierarchical', 'lda', 'qda', 'svm', 'rf', 'knn', 'dt', 'nn', 'xgboost', 'lightgbm')
        """
        # Standardize method names - allow multiple spellings
        method_mapping = {
            'random_forest': 'rf',
            'randomforest': 'rf',
            'random forest': 'rf',
            'rf': 'rf',
            'knn': 'knn',
            'lda': 'lda',
            'qda': 'qda',
            'svm': 'svm',
            'kmeans': 'kmeans',
            'k-means': 'kmeans',
            'k_means': 'kmeans',
            'hierarchical': 'hierarchical',
            'dt': 'dt',
            'decision_tree': 'dt',
            'decision tree': 'dt',
            'decisiontree': 'dt',
            'nn': 'nn',
            'neural_network': 'nn',
            'neural network': 'nn',
            'neuralnetwork': 'nn',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
        }
        
        # Try to standardize method name
        self.method = method_mapping.get(method.lower(), method)
        logger.debug(f"Initializing QualitativeAnalyzer, method: '{method}' standardized to '{self.method}'")
        
        # Initialize attributes
        self.model = None
        self.scaler = StandardScaler()
        self.task_type = 'classification'  # Explicitly mark task type as classification
        
        # **CRITICAL FIX: Initialize enhanced label processor for consistent handling**
        self.label_processor = EnhancedLabelProcessor()
        print("🔧 QualitativeAnalyzer initialized with EnhancedLabelProcessor")
        
        # If method name is invalid, log warning but don't raise error immediately (delay until fit)
        valid_methods = [k for k in method_mapping.values()]
        if self.method not in valid_methods:
            logger.warning(f"Warning: '{self.method}' is not a valid method name. Valid methods are: {set(valid_methods)}")
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model
        
        Args:
            X: Input features
            y: Target labels
            **kwargs: Additional parameters for the model
        """
        global pd
        
        logger.debug(f"Fitting model with method {self.method}")
        
        # Check if input data is empty
        if X is None or len(X) == 0:
            logger.error("Input features cannot be empty")
            raise ValueError("Input features cannot be empty")
            
        if y is None or len(y) == 0:
            logger.error("Target labels cannot be empty")
            raise ValueError("Target labels cannot be empty")
            
        # Standardize and prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle NaN values
        if np.isnan(X_scaled).any():
            logger.warning("NaN values found in scaled data. Filling with column means.")
            col_means = np.nanmean(X_scaled, axis=0)
            inds = np.where(np.isnan(X_scaled))
            X_scaled[inds] = np.take(col_means, inds[1])
        
        # **CRITICAL FIX: Standardize label processing, remove spaces and other inconsistencies**

        
        # Ensure all labels are strings and remove spaces
        if isinstance(y, pd.Series):
            y_clean = y.astype(str).str.strip()
        elif isinstance(y, np.ndarray):
            y_clean = np.array([str(label).strip() for label in y])
        else:
            y_clean = np.array([str(label).strip() for label in y])
        

        
        # Use sklearn LabelEncoder for standardized encoding
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_clean)
        
        # Create bidirectional mapping for debugging and validation
        self.label_forward_map = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        self.label_inverse_map = {idx: label for idx, label in enumerate(self.label_encoder.classes_)}
        



        
        # Validate mapping consistency
        assert len(self.label_forward_map) == len(self.label_inverse_map), "Mapping table length inconsistent"
        assert len(self.label_encoder.classes_) == len(np.unique(y_encoded)), "Encoded category count inconsistent"
        
        self.task_type = 'classification'
        
        try:
            # Create model based on method
            self.model = self._create_model(y_encoded)
            
            # Fit the model
            self.model.fit(X_scaled, y_encoded)
            
            # Verify model classes match our encoding
            if hasattr(self.model, 'classes_'):
                print(f"✅ Model training completed:")
                print(f"   Model recognized classes: {len(self.model.classes_)}")
                print(f"   Model class range: {self.model.classes_.min()}-{self.model.classes_.max()}")
                
                # Verify model classes match our encoding
                expected_classes = np.arange(len(self.label_encoder.classes_))
                if not np.array_equal(sorted(self.model.classes_), sorted(expected_classes)):
                    logger.warning(f"Model classes do not match expected classes: model={sorted(self.model.classes_)}, expected={sorted(expected_classes)}")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted labels in original format
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        logger.debug("Making predictions on new data")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Handle NaN values
        if np.isnan(X_scaled).any():
            logger.warning("NaN values found in scaled data. Filling with column means.")
            col_means = np.nanmean(X_scaled, axis=0)
            inds = np.where(np.isnan(X_scaled))
            X_scaled[inds] = np.take(col_means, inds[1])
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # **CRITICAL FIX: Use standard LabelEncoder for inverse transformation**
        print(f"🔍 Raw prediction results:")
        print(f"   Type: {type(predictions)}, dtype: {predictions.dtype}")
        print(f"   Shape: {predictions.shape}")
        print(f"   Unique values: {np.unique(predictions)}")
        print(f"   Samples: {predictions[:10]}")
        
        # Use stored LabelEncoder for inverse transformation
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            print(f"🗺️ Found LabelEncoder, converting integer indices to original labels")
            
            try:
                # Convert integer prediction results back to original labels
                predictions_original = self.label_encoder.inverse_transform(predictions)
                
                print(f"✅ Converted prediction results:")
                print(f"   Type: {type(predictions_original)}, dtype: {predictions_original.dtype}")
                print(f"   Unique values: {np.unique(predictions_original)}")
                print(f"   Samples: {predictions_original[:10]}")
                
                logger.info(f"Final predictions type: {type(predictions_original)}, dtype: {predictions_original.dtype}")
                return predictions_original
                
            except Exception as e:
                print(f"❌ LabelEncoder inverse transform failed: {e}")
                print(f"   Prediction range: {predictions.min()} - {predictions.max()}")
                print(f"   Encoder range: 0 - {len(self.label_encoder.classes_) - 1}")
                
                # Try manual mapping as backup option
                if hasattr(self, 'label_inverse_map') and self.label_inverse_map:
                    print("🔄 Attempting manual mapping as fallback")
                    try:
                        converted_predictions = []
                        for pred in predictions:
                            pred_int = int(pred)
                            if pred_int in self.label_inverse_map:
                                converted_predictions.append(self.label_inverse_map[pred_int])
                            else:
                                # Find closest mapping
                                available_keys = list(self.label_inverse_map.keys())
                                closest_key = min(available_keys, key=lambda x: abs(x - pred_int))
                                print(f"⚠️ Using closest mapping for {pred_int}: {closest_key} -> {self.label_inverse_map[closest_key]}")
                                converted_predictions.append(self.label_inverse_map[closest_key])
                        
                        predictions_manual = np.array(converted_predictions, dtype='<U50')
                        print(f"✅ Manual mapping successful: {predictions_manual[:5]}")
                        return predictions_manual
                        
                    except Exception as manual_error:
                        print(f"❌ Manual mapping also failed: {manual_error}")
        else:
            print("❌ No LabelEncoder found, this will cause conversion errors!")
            print(f"   Available attributes: {[attr for attr in dir(self) if 'label' in attr.lower()]}")
        
        # Final fallback: return original prediction results
        print("⚠️ Returning raw predictions as fallback")
        logger.info(f"Final predictions type: {type(predictions)}, dtype: {predictions.dtype}")
        return predictions
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation, supporting classification and regression tasks
        
        parameters:
            X: feature matrix (n_samples, n_features)
            y: label vector (n_samples,)
            cv: cross-validation folds
        
        Returns:
            dictionary containing evaluation metrics
        """
        logger.info("Starting cross-validation with mixed label type fix")
        
        global pd
        
        logger.info(f"Starting cross-validation with {cv} folds")
        logger.info(f"Input data shape: X={X.shape}, y={y.shape if hasattr(y, 'shape') else len(y)}")
        
        try:
            # **STEP 1: SINGLE PASS LABEL PROCESSING - NO REDUNDANT CONVERSIONS**
            logger.info("=== Single-pass label processing started ===")
            
            # Use our improved prepare_labels_safely function ONCE
            y_processed, detected_task_type = prepare_labels_safely(y, task_type='classification')
            
            logger.info(f"Label processing result: type={detected_task_type}, shape={y_processed.shape}")
            logger.info(f"Processed label samples: {y_processed[:10]}")
            logger.info(f"Unique label count: {len(np.unique(y_processed))}")
            
            # Encode to integer indices for sklearn compatibility ONCE
            logger.info("=== Single-pass integer encoding ===")
            
            from app.utils.data_compatibility import encode_classification_labels
            safe_y, forward_map, inverse_map = encode_classification_labels(
                y_processed, return_mapping=True
            )
            
            logger.info(f"Final encoded labels type: {safe_y.dtype}")
            logger.info(f"Final encoded labels range: {np.min(safe_y)} to {np.max(safe_y)}")
            logger.info(f"Number of classes: {len(np.unique(safe_y))}")
            
            # Store the mapping for later use
            self.current_label_mapping = {
                'forward': forward_map,
                'inverse': inverse_map
            }
            
            # **STEP 2: VERIFY NO MIXED TYPES REMAIN**
            assert safe_y.dtype.kind in 'iuf', f"Labels must be numeric after encoding, got {safe_y.dtype}"
            assert np.all(np.equal(np.mod(safe_y, 1), 0)), "All encoded labels must be integers"
            
            logger.info("✅ Label type check passed: {sample_types}")
            
            if not hasattr(self, 'model'):
                logger.error("No model configured for cross-validation")
                raise ValueError("No model configured for cross-validation")
            
            if not hasattr(self, 'scaler'):
                logger.warning("No scaler configured, using default StandardScaler")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Set task type from processed results
            self.task_type = detected_task_type
            logger.info(f"Task type: {self.task_type}")
            
            # Scale input data
            X_scaled = self.scaler.fit_transform(X)
            
            # Handle NaN values in features
            if np.isnan(X_scaled).any():
                logger.warning("NaN values found in scaled data. Filling with column means.")
                col_means = np.nanmean(X_scaled, axis=0)
                inds = np.where(np.isnan(X_scaled))
                X_scaled[inds] = np.take(col_means, inds[1])
            
            # **STEP 3: DIRECT SKLEARN CROSS-VALIDATION - NO MORE PROCESSING**
            logger.info("=== Direct cross-validation with processed labels ===")
            
            # Final safety check - make sure no mixed types
            sample_types = set(type(val).__name__ for val in safe_y[:20])
            if len(sample_types) > 1:
                raise ValueError(f"Mixed label types detected after processing: {sample_types}")
            
            logger.info(f"Final safe_y dtype: {safe_y.dtype}, shape: {safe_y.shape}")
            logger.info(f"Safe_y unique values: {np.unique(safe_y)}")
            
            # Create cross-validation splitter
            from sklearn.model_selection import StratifiedKFold, KFold
            
            if self.task_type == 'classification':
                logger.info(f"Using stratified cross-validation (StratifiedKFold), folds: {cv}")
                kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                
                # **CRITICAL: Use cross_val_predict with processed labels**
                logger.info("Performing cross-validation prediction...")
                try:
                    # Ensure model exists and is properly configured
                    if self.model is None:
                        logger.info("Model not configured, creating new model for cross-validation...")
                        self.model = self._create_model()
                        logger.info(f"Created model: {type(self.model).__name__}")
                    
                    logger.info(f"  X_scaled: {type(X_scaled).__name__}, dtype: {X_scaled.dtype}, shape: {X_scaled.shape}")
                    logger.info(f"  safe_y: {type(safe_y).__name__}, dtype: {safe_y.dtype}, shape: {safe_y.shape}")
                    logger.info(f"  safe_ysamples: {safe_y[:10]}")
                    logger.info(f"  model: {type(self.model).__name__}")
                    
                    if not isinstance(safe_y, np.ndarray):
                        safe_y = np.array(safe_y)
                    
                    # **CRITICAL FIX: Always use LabelEncoder for string labels**
                    # Never try to directly convert strings to integers
                    if safe_y.dtype.kind not in 'iu':  # If not integer type
                        logger.info("Converting non-integer labels using LabelEncoder...")
                        from sklearn.preprocessing import LabelEncoder
                        le_final = LabelEncoder()
                        # Always convert to string first to handle mixed types
                        safe_y_str = np.array([str(val) for val in safe_y])
                        safe_y = le_final.fit_transform(safe_y_str)
                        logger.info(f"LabelEncoder applied - converted {len(le_final.classes_)} unique labels")
                        logger.info(f"Label mapping: {dict(zip(le_final.classes_, range(len(le_final.classes_))))}")
                        # Store the encoder for later use
                        self.current_label_encoder = le_final
                        self.current_label_mapping = {
                            'forward': dict(zip(le_final.classes_, range(len(le_final.classes_)))),
                            'inverse': dict(zip(range(len(le_final.classes_)), le_final.classes_))
                        }
                    
                    # **CRITICAL FIX: Use enhanced label processor for safe conversion**
                    from app.utils.label_processor import EnhancedLabelProcessor
                    label_processor = EnhancedLabelProcessor()
                    
                    # Detect task type and process labels safely
                    detected_task = label_processor.detect_task_type(safe_y)
                    logger.info(f"Cross-validation task type detection: {detected_task}")
                    
                    if detected_task == 'classification':
                        # For classification, use LabelEncoder to convert string labels to integers
                        logger.info("Using LabelEncoder for string labels in cross-validation")
                        processed_y, metadata = label_processor.process_labels_smart(safe_y, 'classification')
                        safe_y = processed_y
                        
                        # Store the mapping for later use
                        if 'label_to_index' in metadata:
                            self.current_label_mapping = {
                                'forward': metadata['label_to_index'],
                                'inverse': metadata['index_to_label']
                            }
                            logger.info(f"Label mapping created: {len(metadata['label_to_index'])} classes")
                    else:
                        # For regression, ensure numeric conversion
                        try:
                            if not isinstance(safe_y, np.ndarray):
                                safe_y = np.array(safe_y)
                            if safe_y.dtype.kind not in 'iuf':  # not integer, unsigned, or float
                                # **CRITICAL FIX: Check for string labels before forcing float conversion**
                                sample_labels = [str(label) for label in safe_y[:5]]
                                has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                                       for label in sample_labels if label.strip())
                                
                                if has_string_labels:
                                    raise ValueError(f"Regression task cannot process string labels like {sample_labels[:3]}. Use classification analysis instead.")
                                else:
                                    # **CRITICAL FIX: Safe conversion to float64 with error handling**
                                    try:
                                        safe_y = safe_y.astype(np.float64)
                                    except (ValueError, TypeError) as conversion_error:
                                        raise ValueError(f"Cannot convert labels to float64: {conversion_error}. Labels contain: {sample_labels[:3]}")
                        except Exception as e:
                            logger.error(f"❌ Numeric conversion failed for regression: {e}")
                            raise ValueError(f"Regression task requires numeric labels, conversion failed: {e}")
                    
                    logger.info(f"Final label processing complete - dtype: {safe_y.dtype}, unique values: {np.unique(safe_y)}")
                    
                    # Use cross_val_predict to get predictions
                    y_pred = cross_val_predict(self.model, X_scaled, safe_y, cv=kfold)
                    
                except Exception as cv_error:
                    # Check if this is the "Mix of label input types" error
                    if "Mix of label input types" in str(cv_error):
                        logger.error(f"safe_y dtype: {getattr(safe_y, 'dtype', 'N/A')}")
                        logger.error(f"safe_y samples: {safe_y[:10]}")
                        
                        raise Exception(f"label type consistency validation failed: {str(cv_error)}")
                    else:
                        raise cv_error
                
                # Check for NaN values in predictions (only for numeric types)
                has_nan = False
                try:
                    if y_pred.dtype.kind in 'fc':  # float or complex
                        has_nan = np.isnan(y_pred).any()
                except (TypeError, ValueError):
                    # For non-numeric types, check for None values
                    has_nan = any(val is None or (isinstance(val, str) and val.lower() == 'nan') for val in y_pred)
                
                if has_nan:
                    if y_pred.dtype.kind in 'fc':  # numeric type
                        nan_count = np.isnan(y_pred).sum()
                    else:  # non-numeric type
                        nan_count = sum(1 for val in y_pred if val is None or (isinstance(val, str) and val.lower() == 'nan'))
                    logger.warning(f"NaN values found in predictions: {nan_count} ({nan_count/len(y_pred):.2%}). Using fallback method.")
                    # Fallback to manual cross-validation
                    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                    y_pred = np.zeros_like(safe_y, dtype=float)
                    
                    for train_idx, test_idx in kf.split(X_scaled):
                        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                        y_train = safe_y[train_idx]
                        self.model.fit(X_train, y_train)
                        fold_preds = self.model.predict(X_test)
                        y_pred[test_idx] = fold_preds
                    
                    logger.info(f"Corrected prediction analysis:")
                    logger.info(f"  - Prediction type: {type(y_pred)} / {getattr(y_pred, 'dtype', 'N/A')}")
                    logger.info(f"  - Unique predictions: {np.unique(y_pred)}")
                
                # Calculate classification metrics using processed labels
                logger.info("Calculating classification metrics...")
                
                # Convert both to same type if needed
                if hasattr(safe_y, 'dtype') and hasattr(y_pred, 'dtype'):
                    if safe_y.dtype != y_pred.dtype:
                        logger.warning(f"Type mismatch: safe_y={safe_y.dtype}, y_pred={y_pred.dtype}")
                        # **CRITICAL FIX: Handle both string and numeric labels appropriately**
                        try:
                            # Check if we have string labels (like Verde)
                            safe_y_is_string = hasattr(safe_y, 'dtype') and safe_y.dtype.kind in 'UO'  # Unicode or Object
                            y_pred_is_string = hasattr(y_pred, 'dtype') and y_pred.dtype.kind in 'UO'  # Unicode or Object
                            
                            if safe_y_is_string or y_pred_is_string:
                                logger.info("Detected string labels (like Verde), keeping as strings for classification")
                                # For string-based classification, ensure both are string arrays
                                if not safe_y_is_string:
                                    safe_y = np.array([str(x) for x in safe_y])
                                if not y_pred_is_string:
                                    y_pred = np.array([str(x) for x in y_pred])
                                logger.info("Both arrays prepared as strings for string-based classification")
                            else:
                                # For numeric labels, convert to int32
                                safe_y = safe_y.astype(np.int32)
                                y_pred = y_pred.astype(np.int32)
                                logger.info("Both arrays converted to int32 for numeric classification")
                        except Exception as conversion_error:
                            logger.error(f"Label type conversion failed: {conversion_error}")
                            logger.info("Keeping original data types for classification")
                
                # **CRITICAL FIX: Remove integer type assertion for string labels**
                # For string-based classification (like Verde), we don't need integer types
                logger.info(f"Final type check - allowing both integer and string types")
                logger.info(f"  - safe_y: {safe_y.dtype}, shape: {safe_y.shape}")
                logger.info(f"  - y_pred: {y_pred.dtype}, shape: {y_pred.shape}")
                
                # Only check compatibility, not force integer types
                if hasattr(safe_y, 'dtype') and hasattr(y_pred, 'dtype'):
                    logger.info(f"Label types are compatible for classification metrics")
                
                # Import metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                from sklearn.metrics import classification_report, confusion_matrix
                
                # Calculate metrics with error handling
                try:
                    accuracy = accuracy_score(safe_y, y_pred)
                    
                    # Calculate average metrics (use 'weighted' for multi-class)
                    precision = precision_score(safe_y, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(safe_y, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(safe_y, y_pred, average='weighted', zero_division=0)
                    
                    logger.info(f"Classification metrics calculated successfully")
                    logger.info(f"  - Accuracy: {accuracy:.4f}")
                    logger.info(f"  - Precision: {precision:.4f}")
                    logger.info(f"  - Recall: {recall:.4f}")
                    logger.info(f"  - F1-score: {f1:.4f}")
                    
                    # **Added: Low accuracy automatic diagnosis**
                    diagnosis_info = None
                    if accuracy < 0.5:  # Trigger diagnosis when accuracy < 50%
                        logger.warning(f"⚠️ Low accuracy detected ({accuracy:.4f}), diagnosing issues...")
                        diagnosis_info = self.diagnose_low_accuracy(X_scaled, safe_y, accuracy_threshold=0.5)
                        
                        logger.warning("🔍 Accuracy diagnosis results:")
                        if diagnosis_info['issues']:
                            logger.warning("   Issues found:")
                            for issue in diagnosis_info['issues']:
                                logger.warning(f"   - {issue}")
                        
                        if diagnosis_info['model_issues']:
                            logger.warning("   Model-related issues:")
                            for issue in diagnosis_info['model_issues']:
                                logger.warning(f"   - {issue}")
                        
                        if diagnosis_info['suggestions']:
                            logger.warning("   Improvement suggestions:")
                            for suggestion in diagnosis_info['suggestions'][:5]:  # Show only first 5 suggestions
                                logger.warning(f"   - {suggestion}")
                    
                    # Generate classification report
                    try:
                        # Map back to original labels for display
                        if hasattr(self, 'current_label_mapping') and 'inverse' in self.current_label_mapping:
                            # Get original labels for report
                            original_labels = [self.current_label_mapping['inverse'].get(label, str(label)) for label in np.unique(safe_y)]
                            target_names = [str(label) for label in original_labels]
                        else:
                            target_names = [str(label) for label in np.unique(safe_y)]
                            
                        class_report = classification_report(safe_y, y_pred, target_names=target_names, zero_division=0)
                        logger.info(f"Classification Report:\n{class_report}")
                        
                        # Confusion matrix
                        conf_matrix = confusion_matrix(safe_y, y_pred)
                        logger.info(f"Confusion Matrix:\n{conf_matrix}")
                        
                    except Exception as report_error:
                        logger.warning(f"Could not generate detailed classification report: {str(report_error)}")
                        class_report = f"Basic metrics only - detailed report failed: {str(report_error)}"
                        conf_matrix = np.array([[0]])
                    
                    # Return results
                    return {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'classification_report': class_report,
                        'confusion_matrix': conf_matrix.tolist(),
                        'task_type': 'classification',
                        'cv_folds': cv,
                        'n_samples': len(safe_y),
                        'n_classes': len(np.unique(safe_y)),
                        'diagnosis': diagnosis_info
                    }
                    
                except Exception as metrics_error:
                    logger.error(f"Error calculating classification metrics: {str(metrics_error)}")
                    raise ValueError(f"Failed to calculate classification metrics: {str(metrics_error)}")
                    
            else:
                # Handle regression case
                logger.info(f"Using K-Fold cross-validation for regression, folds: {cv}")
                kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
                
                if self.model is None:
                    logger.info("Model not configured, creating new model for cross-validation...")
                    self.model = self._create_model()
                    logger.info(f"Created model: {type(self.model).__name__}")
                
                y_pred = cross_val_predict(self.model, X_scaled, safe_y, cv=kfold)
                
                # Import regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Calculate regression metrics
                mse = mean_squared_error(safe_y, y_pred)
                mae = mean_absolute_error(safe_y, y_pred)
                r2 = r2_score(safe_y, y_pred)
                rmse = np.sqrt(mse)
                
                logger.info(f"Regression metrics calculated:")
                logger.info(f"  - R²: {r2:.4f}")
                logger.info(f"  - RMSE: {rmse:.4f}")
                logger.info(f"  - MAE: {mae:.4f}")
                logger.info(f"  - MSE: {mse:.4f}")
                
                return {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mse': mse,
                    'task_type': 'regression',
                    'cv_folds': cv,
                    'n_samples': len(safe_y)
                }
                
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Cross-validation error: {str(e)}")

    def _create_model(self, y=None):
        """Create a model with optimized parameters for spectral data classification"""
        if self.method == 'kmeans':
            # Use silhouette analysis to determine optimal number of clusters if not specified
            # Default to 3 clusters if analysis fails or number is not determined
            try:
                if hasattr(self, 'n_clusters'):
                    return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=500)
                else:
                    return KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=500)
            except:
                return KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=500)
                
        elif self.method == 'hierarchical':
            try:
                if hasattr(self, 'n_clusters'):
                    return AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
                else:
                    return AgglomerativeClustering(n_clusters=3, linkage='ward')
            except:
                return AgglomerativeClustering(n_clusters=3, linkage='ward')
                
        elif self.method == 'lda':
            # **OPTIMIZED: Linear Discriminant Analysis for spectral data**
            return LinearDiscriminantAnalysis(
                solver='svd',  # more stable for high-dimensional data
                shrinkage=None,  # automatic shrinkage
                store_covariance=True,
                tol=1e-6  # improve precision
            )
            
        elif self.method == 'qda':
            # **OPTIMIZED: Quadratic Discriminant Analysis**
            return QuadraticDiscriminantAnalysis(
                reg_param=0.01,  # add regularization to prevent overfitting
                store_covariance=True,
                tol=1e-6
            )
            
        elif self.method == 'svm':
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced SVM parameters**
            return SVC(
                C=1.0,                     # moderate regularization parameter
                kernel='rbf',              # RBF kernel suitable for nonlinear relationships in spectral data
                gamma='scale',             # Automatically adjust gamma, adapt to data dimensions
                probability=True,          # Enable probability prediction
                random_state=42,
                class_weight='balanced',   # handle imbalanced data
                tol=1e-4,                 # improve convergence precision
                cache_size=500,           # increase cache, improve training speed
                max_iter=5000             # increase maximum iterations
            )
            
        elif self.method in ['rf', 'random_forest']:
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced Random Forest parameters**
            return RandomForestClassifier(
                n_estimators=500,          # increase number of trees for better accuracy
                max_depth=20,              # limit depth to prevent overfitting
                min_samples_split=5,       # prevent overfitting
                min_samples_leaf=2,        # prevent overfitting
                max_features='sqrt',       # suitable for high-dimensional spectral data
                bootstrap=True,
                random_state=42,
                class_weight='balanced',   # handle imbalanced data
                oob_score=True,           # enable out-of-bag scoring
                n_jobs=-1,                # parallel computing
                criterion='gini',         # Gini impurity
                warm_start=False
            )
            
        elif self.method == 'knn':
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced KNN parameters**
            return KNeighborsClassifier(
                n_neighbors=7,             # increase neighbors, improve stability
                weights='distance',        # distance weighting, suitable for spectral data
                algorithm='auto',          # automatically select optimal algorithm
                leaf_size=20,             # optimize tree structure
                p=2,                      # Euclidean distance
                metric='minkowski',       # Minkowski distance
                n_jobs=-1                 # parallel computing
            )
            
        elif self.method in ['dt', 'decision_tree']:
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced Decision Tree parameters**
            return DecisionTreeClassifier(
                max_depth=20,             # increase maximum depth, adapt to complex spectral patterns
                random_state=42,
                class_weight='balanced',  # handle imbalanced data
                criterion='gini',         # Gini impurity
                splitter='best',          # optimal split
                min_samples_split=5,      # prevent overfitting
                min_samples_leaf=2,       # prevent overfitting
                max_features='sqrt',      # feature subset selection, suitable for high-dimensional data
                max_leaf_nodes=None
            )
            
        elif self.method in ['nn', 'neural_network', 'mlp']:
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced Neural Network parameters**
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(300, 150, 75),  # increase network capacity
                activation='relu',                   
                solver='adam',                       
                alpha=0.0001,                       # reduce regularization, allow better fitting
                batch_size='auto',
                learning_rate='adaptive',           
                learning_rate_init=0.001,           # lower learning rate, more stable training
                max_iter=2000,                      # increase training rounds
                shuffle=True,
                random_state=42,
                tol=1e-6,                          
                verbose=False,
                warm_start=False,
                early_stopping=True,               
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                n_iter_no_change=50                # increase early stopping patience
            )
            
        elif self.method == 'xgboost':
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced XGBoost parameters**
            try:
                import xgboost as xgb
                
                # Auto-detect number of classes
                if hasattr(y, 'nunique'):
                    num_classes = y.nunique()
                else:
                    num_classes = len(np.unique(y))
                
                # Choose objective function based on number of classes
                if num_classes == 2:
                    objective = 'binary:logistic'
                    eval_metric = 'logloss'
                else:
                    objective = 'multi:softprob'
                    eval_metric = 'mlogloss'
                
                return xgb.XGBClassifier(
                    n_estimators=200,              # increase number of trees
                    max_depth=6,                   # moderate tree depth to avoid overfitting
                    learning_rate=0.05,            # lower learning rate for better accuracy
                    subsample=0.8,                 # subsample ratio
                    colsample_bytree=0.8,          # feature subsampling
                    random_state=42,
                    n_jobs=-1,                     # parallel computing
                    reg_alpha=0.1,                 # L1 regularization
                    reg_lambda=1.0,                # enhanced L2 regularization
                    scale_pos_weight=1,            # handle imbalanced data
                    objective=objective,           # dynamic objective function selection
                    eval_metric=eval_metric,       # dynamic evaluation metric selection
                    num_class=num_classes if num_classes > 2 else None  # specify number of classes for multiclass
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                return self._create_model_fallback('rf')
                
        elif self.method == 'lightgbm':
            # **OPTIMIZED FOR SPECTRAL DATA: Enhanced LightGBM parameters**
            try:
                import lightgbm as lgb
                
                # Auto-detect number of classes
                if hasattr(y, 'nunique'):
                    num_classes = y.nunique()
                else:
                    num_classes = len(np.unique(y))
                
                # Choose objective function based on number of classes
                if num_classes == 2:
                    objective = 'binary'
                    metric = 'binary_logloss'
                    # 🔧 V1.2.1修复: 二分类不需要num_class参数
                    return lgb.LGBMClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        class_weight='balanced',
                        objective=objective,
                        metric=metric,
                        verbose=-1,
                        min_child_samples=20,
                        min_split_gain=0.1
                    )
                else:
                    objective = 'multiclass'
                    metric = 'multi_logloss'
                    # 多分类需要num_class参数
                    return lgb.LGBMClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        class_weight='balanced',
                        objective=objective,
                        metric=metric,
                        num_class=num_classes,
                        verbose=-1,
                        min_child_samples=20,
                        min_split_gain=0.1
                    )
            except ImportError:
                logger.warning("LightGBM not available, falling back to Random Forest")
                return self._create_model_fallback('rf')
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _create_model_fallback(self, fallback_method='rf'):
        """Create fallback model when preferred method is not available"""
        logger.info(f"Creating fallback model: {fallback_method}")
        original_method = self.method
        self.method = fallback_method
        model = self._create_model()
        self.method = original_method  # Restore original method
        return model
    
    def plot_dendrogram(self, X: np.ndarray) -> None:
        """
        Plot hierarchical clustering dendrogram
        
        Args:
            X: Input features
        """
        if self.method != 'hierarchical':
            raise ValueError("Only hierarchical clustering method supports dendrograms")
            
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle NaN values in X_scaled if any
        if np.isnan(X_scaled).any():
            logger.warning("NaN values found in scaled data for dendrogram. Filling with column means.")
            # For each column with NaN values, replace with column mean
            col_means = np.nanmean(X_scaled, axis=0)
            inds = np.where(np.isnan(X_scaled))
            X_scaled[inds] = np.take(col_means, inds[1])
        
        linkage_matrix = linkage(X_scaled, method='ward')
        
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        if self.model is None:
            return {}
            
        # Need to handle different data types properly for parameters
        # Return only serializable parameters
        params = {}
        
        try:
            if self.method == 'kmeans':
                params = {
                    'n_clusters': self.model.n_clusters,
                }
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(self.model, 'cluster_centers_'):
                    centers = self.model.cluster_centers_
                    if isinstance(centers, np.ndarray):
                        params['cluster_centers_shape'] = centers.shape
                    else:
                        params['cluster_centers_'] = 'Not available as numpy array'
                        
                if hasattr(self.model, 'labels_'):
                    labels = self.model.labels_
                    if isinstance(labels, np.ndarray):
                        params['num_labels'] = len(labels)
                        params['unique_labels'] = len(np.unique(labels))
                    else:
                        params['labels_'] = 'Not available as numpy array'
                        
            elif self.method == 'hierarchical':
                params = {
                    'n_clusters': self.model.n_clusters,
                }
                if hasattr(self.model, 'labels_'):
                    labels = self.model.labels_
                    if isinstance(labels, np.ndarray):
                        params['num_labels'] = len(labels)
                        params['unique_labels'] = len(np.unique(labels))
                    else:
                        params['labels_'] = 'Not available as numpy array'
                        
            elif self.method == 'rf':
                params = {
                    'n_estimators': self.model.n_estimators,
                }
                if hasattr(self.model, 'max_depth'):
                    params['max_depth'] = self.model.max_depth
                    
                if hasattr(self.model, 'feature_importances_'):
                    # Convert feature importance array to list for safe serialization
                    feature_importances = self.model.feature_importances_
                    if isinstance(feature_importances, np.ndarray):
                        params['feature_importances_shape'] = feature_importances.shape
                        # Include top 5 feature importance values
                        if len(feature_importances) > 0:
                            top_indices = np.argsort(feature_importances)[-5:]
                            params['top5_feature_indices'] = top_indices.tolist()
                            params['top5_feature_importances'] = feature_importances[top_indices].tolist()
                else:
                    params['feature_importances_'] = 'Not available as numpy array'
            else:
                # Get basic parameters that should be available on most models
                basic_params = self.model.get_params()
                # Filter out complex objects that might not be JSON serializable
                for key, value in basic_params.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        params[key] = value
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, bool, type(None))) for x in value):
                        params[key] = list(value)
                    else:
                        params[key] = str(type(value))
        except Exception as e:
            logger.error(f"Error extracting model parameters: {str(e)}")
            params['error'] = str(e)
                
        return params 

    def diagnose_low_accuracy(self, X, y, accuracy_threshold=0.3):
        """
        Diagnose possible causes of low accuracy
        
        Args:
            X: Feature data
            y: Label data
            accuracy_threshold: Accuracy threshold for triggering diagnosis
            
        Returns:
            dict: Diagnosis results and suggestions
        """
        diagnosis = {
            'issues': [],
            'suggestions': [],
            'data_stats': {},
            'model_issues': []
        }
        
        try:
            # 1. Data quality checks
            diagnosis['data_stats']['n_samples'] = len(X)
            diagnosis['data_stats']['n_features'] = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            diagnosis['data_stats']['n_classes'] = len(np.unique(y))
            
            # Check if sample size is sufficient
            if len(X) < 100:
                diagnosis['issues'].append("Insufficient sample size")
                diagnosis['suggestions'].append("Increase training sample size, recommend at least 100 samples")
            
            # Check if feature dimensionality is too high
            if X.shape[1] > len(X):
                diagnosis['issues'].append("High feature dimensionality (curse of dimensionality)")
                diagnosis['suggestions'].append("Use feature selection or dimensionality reduction techniques")
            
            # Check class balance
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            diagnosis['data_stats']['class_distribution'] = class_distribution
            
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                diagnosis['issues'].append("Severe class imbalance")
                diagnosis['suggestions'].append("Use class weights, oversampling, or undersampling techniques")
            elif imbalance_ratio > 3:
                diagnosis['issues'].append("Mild class imbalance")
                diagnosis['suggestions'].append("Consider using class_weight='balanced' parameter")
            
            # Check if there are too many classes
            if len(unique) > 20:
                diagnosis['issues'].append("Too many classes")
                diagnosis['suggestions'].append("Consider merging similar classes or using hierarchical classification")
            elif len(unique) > len(X) / 10:
                diagnosis['issues'].append("Too many classes relative to sample size")
                diagnosis['suggestions'].append("Increase samples per class")
            
            # 2. Data preprocessing checks
            # Check data range
            if hasattr(X, 'std'):
                feature_stds = X.std(axis=0)
                if np.any(feature_stds == 0):
                    diagnosis['issues'].append("Constant features detected")
                    diagnosis['suggestions'].append("Remove zero-variance features")
                
                if np.any(feature_stds > 1000) or np.any(feature_stds < 0.001):
                    diagnosis['issues'].append("Large feature scale differences")
                    diagnosis['suggestions'].append("Use StandardScaler or MinMaxScaler for feature scaling")
            
            # Check missing values
            if hasattr(X, 'isnull'):
                if X.isnull().any().any():
                    diagnosis['issues'].append("Missing values detected")
                    diagnosis['suggestions'].append("Handle missing values: imputation, deletion, or interpolation")
            elif isinstance(X, np.ndarray):
                if np.isnan(X).any():
                    diagnosis['issues'].append("NaN values detected")
                    diagnosis['suggestions'].append("Handle NaN values: imputation, deletion, or interpolation")
            
            # 3. Model compatibility checks
            # Check if current model is suitable for data characteristics
            if self.method == 'lda' and len(unique) >= X.shape[1]:
                diagnosis['model_issues'].append("LDA not suitable when n_classes >= n_features")
                diagnosis['suggestions'].append("Use Random Forest or SVM instead")
            
            if self.method == 'knn' and X.shape[1] > 50:
                diagnosis['model_issues'].append("KNN performs poorly on high-dimensional data")
                diagnosis['suggestions'].append("Apply dimensionality reduction first or use Random Forest")
            
            if self.method == 'svm' and len(X) > 10000:
                diagnosis['model_issues'].append("SVM is slow on large datasets")
                diagnosis['suggestions'].append("Consider Random Forest or reduce sample size")
            
            # 4. Spectral data specific recommendations
            diagnosis['suggestions'].extend([
                "For spectral data, recommend the following preprocessing:",
                "- SNV (Standard Normal Variate) to remove scattering effects",
                "- Derivative preprocessing (1st or 2nd order) to enhance spectral features",
                "- Baseline correction to remove baseline drift",
                "Recommended model selection order: Random Forest > SVM > LDA > KNN"
            ])
            
            return diagnosis
            
        except Exception as e:
            diagnosis['issues'].append(f"Diagnosis process error: {str(e)}")
            return diagnosis 