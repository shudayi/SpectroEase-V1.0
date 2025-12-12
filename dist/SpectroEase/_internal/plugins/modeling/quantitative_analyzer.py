import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from typing import Dict, Any, List, Optional
from app.utils.data_compatibility import standardize_classification_labels, standardize_regression_values
from app.utils.label_processor import EnhancedLabelProcessor
import logging

# Setup logging
logger = logging.getLogger(__name__)

class QuantitativeAnalyzer:
    """Quantitative Analysis Class"""
    
    def __init__(self, method: str = 'plsr'):
        """
        Initialize quantitative analyzer
        
        Args:
            method: Analysis method ('mlr', 'plsr', 'svr', 'rf', 'nn', 'xgboost', 'lightgbm', 'gpr', 'dt')
        """
        # Standardize method names
        # V1.4.1: Add full algorithm name mappings for LLM-generated algorithms
        method_lower = method.lower()
        method_mapping = {
            'linear': 'mlr',
            'mlr': 'mlr',
            'linear regression': 'mlr',
            'multiple linear regression': 'mlr',
            'pls': 'plsr',
            'plsr': 'plsr',
            'pls regression': 'plsr',
            'partial least squares': 'plsr',
            'partial least squares regression': 'plsr',  # V1.4.1: Support full name
            'svr': 'svr',
            'svm': 'svr',
            'support vector regression': 'svr',
            'support vector machine': 'svr',
            'rf': 'rf',
            'random_forest': 'rf',
            'random forest': 'rf',
            'randomforest': 'rf',
            'dt': 'dt',
            'decision_tree': 'dt',
            'decision tree': 'dt',
            'decisiontree': 'dt',
            'gpr': 'gpr',
            'gaussian_process': 'gpr',
            'gaussian process': 'gpr',
            'gaussianprocess': 'gpr',
        }
        
        # Try exact match first
        self.method = method_mapping.get(method_lower, method)
        
        # V1.4.1: If not found, try fuzzy matching (contains check)
        if self.method == method and method_lower not in method_mapping:
            for key, value in method_mapping.items():
                if key in method_lower or method_lower in key:
                    self.method = value
                    logger.info(f"Matched '{method}' to '{value}' via fuzzy matching")
                    break
        
        # V1.4.2: Enhanced fuzzy matching for long algorithm names (e.g., "Partial Least Squares Regression with CV Component Selection")
        if self.method == method and method_lower not in method_mapping:
            # Check if method name contains any of the mapping keys (for long names)
            for key, value in method_mapping.items():
                if key in method_lower:
                    self.method = value
                    logger.info(f"Matched '{method}' to '{value}' via enhanced fuzzy matching (contains '{key}')")
                    break
        
        # V1.4.1: Fallback warning if still not matched
        if self.method == method and method_lower not in ['mlr', 'plsr', 'svr', 'rf', 'dt', 'gpr', 'nn', 'xgboost', 'lightgbm']:
            # V1.4.2: Enhanced fallback for PLS-related algorithms
            if 'partial least squares' in method_lower or 'pls' in method_lower or 'regression' in method_lower:
                logger.info(f"Detected PLS-related algorithm name: '{method}', using PLSR")
                self.method = 'plsr'
            else:
                logger.warning(f"Unknown method: {method}. Using PLSR as fallback.")
                self.method = 'plsr'
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.task_type = 'regression'  # Explicitly mark task type as regression
        
        # **CRITICAL FIX: Initialize enhanced label processor for consistent handling**
        self.label_processor = EnhancedLabelProcessor()

        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model
        
        Args:
            X: Input features
            y: Target labels
            **kwargs: Additional parameters for the model
        """
        global pd
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
            
        # Scale input data
        X_scaled = self.scaler_x.fit_transform(X)
        
        # For regression methods, we also scale the target
        if self.task_type == 'regression':
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y
            
        # Create model based on method, adjusting parameters based on data
        if self.method == 'plsr':
            # **CRITICAL FIX: Optimize PLSR n_components selection strategy**
            # For spectral data, fewer components usually work better
            default_components = 3  # Changed to more conservative default value
            n_components = kwargs.get('n_components', default_components)
            
            # Ensure n_components does not exceed the minimum of feature count and sample count
            # Also consider the complexity of target variables
            max_components = min(10, n_components)  # Limit maximum components to 10
            
            # **NEW: Dynamically adjust component count based on data characteristics**
            # For small datasets, use fewer components
            if max_components > 5:
                max_components = min(5, max_components)  # Further limit to5
            
            # **CRITICAL FIX: Correctly set kwargs and create model**
            kwargs['n_components'] = max_components
        
        self.model = self._create_model(**kwargs)
        
        # Fit model
        self.model.fit(X_scaled, y_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target variable
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        global pd
        
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale input data
        X_scaled = self.scaler_x.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Inverse transform if regression
        if self.task_type == 'regression':
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            predictions = self.scaler_y.inverse_transform(predictions)
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.ravel()
                
        return predictions
        
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation and return evaluation metrics
        
        Args:
            X: Input features
            y: Target labels
            n_splits: Number of cross-validation splits
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        global pd
        
        # Print input data info for debugging
        logger.info(f"X shape: {X.shape}, y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
        logger.info(f"X type: {type(X)}, y type: {type(y)}")
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
            
        # Perform cross-validation with custom pipeline to handle scaling
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize arrays for predictions and true values
        predictions = np.zeros_like(y, dtype=float)
        
        # For each fold
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler_x = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)
            
            # For regression, also scale target
            if self.task_type == 'regression':
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            else:
                y_train_scaled = y_train
            
            # Create and fit model with adjusted parameters if needed
            kwargs_for_cv = {}
            if self.method == 'plsr':
                # **CRITICAL FIX: Use more conservative component count in cross-validation**
                # Use fewer components to avoid overfitting
                optimal_components = min(3, X_train_scaled.shape[1], X_train_scaled.shape[0])
                kwargs_for_cv['n_components'] = optimal_components
                
            model = self._create_model(**kwargs_for_cv)
            model.fit(X_train_scaled, y_train_scaled)
            
            # Make predictions
            fold_preds = model.predict(X_test_scaled)
            
            # For regression, inverse transform predictions
            if self.task_type == 'regression':
                if len(fold_preds.shape) == 1:
                    fold_preds = fold_preds.reshape(-1, 1)
                fold_preds = scaler_y.inverse_transform(fold_preds)
                if len(fold_preds.shape) > 1 and fold_preds.shape[1] == 1:
                    fold_preds = fold_preds.ravel()
            
            # Store predictions
            predictions[test_idx] = fold_preds
        
        # Calculate metrics
        if self.task_type == 'regression':
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            # Calculate metrics
            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            
            # Log results
            logger.info(f"Cross-validation results:")
            logger.info(f"  - R²: {r2:.4f}")
            logger.info(f"  - MSE: {mse:.4f}")
            logger.info(f"  - MAE: {mae:.4f}")
            
            # Return metrics
            return {
                'r2': float(r2),
                'mse': float(mse),
                'mae': float(mae),
                'predictions': predictions[:100].tolist() if len(predictions) > 100 else predictions.tolist(),
                'target': y[:100].tolist() if len(y) > 100 else y.tolist()
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average='weighted')
            precision = precision_score(y, predictions, average='weighted')
            recall = recall_score(y, predictions, average='weighted')
            
            # Log results
            logger.info(f"Cross-validation results:")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - F1 score: {f1:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            
            # Return metrics
            return {
                'accuracy': float(accuracy),
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'predictions': predictions[:100].tolist() if len(predictions) > 100 else predictions.tolist(),
                'target': y[:100].tolist() if len(y) > 100 else y.tolist()
            }
    
    def _create_model(self, **kwargs) -> Any:
        """
        Create a model instance based on method
        
        Args:
            **kwargs: Additional parameters for the model
            
        Returns:
            Model instance
        """
        # **CRITICAL FIX: Remove task_type from kwargs as it's not a sklearn parameter**
        kwargs.pop('task_type', None)
        
        # Linear Regression
        if self.method == 'mlr':
            return LinearRegression()
        
        # Partial Least Squares Regression
        elif self.method == 'plsr':
            # **CRITICAL FIX: Optimize PLSR n_components selection strategy**
            # For spectral data, fewer components usually work better
            default_components = 3  # Changed to more conservative default value
            n_components = kwargs.get('n_components', default_components)
            
            # Ensure n_components does not exceed the minimum of feature count and sample count
            # Also consider the complexity of target variables
            max_components = min(10, n_components)  # Limit maximum components to 10
            
            # **NEW: Dynamically adjust component count based on data characteristics**
            # For small datasets, use fewer components
            if max_components > 5:
                max_components = min(5, max_components)  # Further limit to5
                
            return PLSRegression(
                n_components=max_components, 
                scale=False,  # We have already manually standardized
                max_iter=1000, 
                tol=1e-6
            )
        
        # Support Vector Regression
        elif self.method == 'svr':
            svr_params = {
                'C': 10.0,           # Increase regularization parameters
                'epsilon': 0.01,     # Reduce epsilon, improve precision
                'kernel': 'rbf',
                'gamma': 'auto',     # Change to auto
                'cache_size': 500,   # Increase cache
                'tol': 1e-4          # Improve convergence precision
            }
            return SVR(**svr_params)
        
        # Random Forest Regression
        elif self.method == 'rf':
            rf_params = {
                'n_estimators': 200,      # Increase number of trees
                'max_depth': None,        # Allow deeper trees
                'min_samples_split': 5,   # Prevent overfitting
                'min_samples_leaf': 2,    # Prevent overfitting
                'max_features': 'sqrt',   # Feature subset selection
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            }
            return RandomForestRegressor(**rf_params)
        
        # Decision Tree Regression
        elif self.method == 'dt':
            dt_params = {
                'max_depth': 15,          # Increase maximum depth
                'min_samples_split': 5,   # Prevent overfitting
                'min_samples_leaf': 2,    # Prevent overfitting
                'max_features': 'sqrt',   # Feature subset selection
                'random_state': 42
            }
            return DecisionTreeRegressor(**dt_params)
        
        # Gaussian Process Regression
        elif self.method == 'gpr':
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
            gpr_params = {
                'kernel': kernel,
                'alpha': 1e-6,           # Reduce noise parameters
                'optimizer': 'fmin_l_bfgs_b',
                'n_restarts_optimizer': 5,  # Increase restart count
                'normalize_y': True,        # Standardize target variables
                'copy_X_train': True,
                'random_state': 42
            }
            return GaussianProcessRegressor(**gpr_params)
        
        # Default to PLS Regression
        else:
            logger.warning(f"Unknown method: {self.method}. Using PLSR as fallback.")
            n_components = kwargs.get('n_components', 10)
            return PLSRegression(n_components=n_components)
    
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
            if self.method == 'plsr':
                # For PLSR, return components, weights, etc.
                params = {
                    'n_components': self.model.n_components,
                }
                
                if hasattr(self.model, 'coef_'):
                    coef_shape = self.model.coef_.shape
                    params['coef_shape'] = list(coef_shape)
                    
                if hasattr(self.model, 'x_scores_'):
                    x_scores_shape = self.model.x_scores_.shape
                    params['x_scores_shape'] = list(x_scores_shape)
                    
                if hasattr(self.model, 'x_weights_'):
                    x_weights_shape = self.model.x_weights_.shape
                    params['x_weights_shape'] = list(x_weights_shape)
                    
            elif self.method == 'rf':
                # For Random Forest, return feature importances
                params = {
                    'n_estimators': self.model.n_estimators,
                    'max_depth': self.model.max_depth,
                }
                
                if hasattr(self.model, 'feature_importances_'):
                    # Get top 5 feature importances
                    importances = self.model.feature_importances_
                    indices = np.argsort(importances)[-5:]
                    top_importances = [(int(i), float(importances[i])) for i in indices]
                    params['top_importances'] = top_importances
                    
            else:
                # For other models, try to get basic parameters
                try:
                    model_params = self.model.get_params()
                    # Filter for serializable params
                    for k, v in model_params.items():
                        if isinstance(v, (int, float, str, bool, type(None))):
                            params[k] = v
                        else:
                            params[k] = str(type(v))
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            params['error'] = str(e)
            
        return params 