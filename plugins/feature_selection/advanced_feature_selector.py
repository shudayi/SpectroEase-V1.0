# plugins/feature_selection/advanced_feature_selector.py

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectFromModel, VarianceThreshold, SelectPercentile, 
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA, FastICA
from typing import Dict, Any, Tuple
from app.utils.label_processor import EnhancedLabelProcessor

class AdvancedFeatureSelector(FeatureSelectionAlgorithm):
    """Advanced feature selection plugin"""
    
    def __init__(self):
        # **CRITICAL FIX: Initialize enhanced label processor for consistent task type detection**
        self.label_processor = EnhancedLabelProcessor()
    
    def get_name(self) -> str:
        return "Advanced Feature Selector"
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'method': {
                'type': 'str',
                'default': 'variance_threshold',
                'options': ['variance_threshold', 'correlation_filter', 'lasso_selection', 
                           'tree_importance', 'ica_selection', 'percentile_selection'],
                'description': 'Feature selection method'
            },
            'threshold': {
                'type': 'float',
                'default': 0.01,
                'description': 'Threshold for variance or correlation filtering'
            },
            'n_features': {
                'type': 'int',
                'default': 100,
                'description': 'Number of features to select'
            },
            'percentile': {
                'type': 'int',
                'default': 50,
                'description': 'Percentile of features to keep'
            },
            'alpha': {
                'type': 'float',
                'default': 1.0,
                'description': 'Regularization strength for Lasso'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random state for reproducibility'
            }
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Select features"""
        method = params.get('method', 'variance_threshold')
        
        if method == 'variance_threshold':
            return self._variance_threshold(X, y, params)
        elif method == 'correlation_filter':
            return self._correlation_filter(X, y, params)
        elif method == 'lasso_selection':
            return self._lasso_selection(X, y, params)
        elif method == 'tree_importance':
            return self._tree_importance(X, y, params)
        elif method == 'ica_selection':
            return self._ica_selection(X, y, params)
        elif method == 'percentile_selection':
            return self._percentile_selection(X, y, params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _variance_threshold(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Variance threshold feature selection"""
        threshold = params.get('threshold', 0.01)
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = selector.get_support()
        
        # Keep DataFrame format
        selected_columns = X.columns[selected_features]
        X_result = pd.DataFrame(X_selected, index=X.index, columns=selected_columns)
        
        return X_result, selected_features
    
    def _correlation_filter(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Correlation filtering feature selection"""
        threshold = params.get('threshold', 0.95)
        
        # Calculate correlation matrix between features
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Select features to keep
        selected_features = ~X.columns.isin(to_drop)
        X_result = X.loc[:, selected_features]
        
        return X_result, selected_features.values
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Lasso regularization feature selection"""
        alpha = params.get('alpha', 1.0)
        random_state = params.get('random_state', 42)
        
        # Determine task type
        if self._is_classification_task(y):
            # For classification tasks, use LassoCV with logistic regression
            from sklearn.linear_model import LogisticRegressionCV
            lasso = LogisticRegressionCV(
                penalty='l1', 
                solver='liblinear',
                Cs=[alpha],
                random_state=random_state
            )
        else:
            # For regression tasks, use LassoCV
            lasso = LassoCV(alphas=[alpha], random_state=random_state)
        
        selector = SelectFromModel(lasso)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support()
        
        selected_columns = X.columns[selected_features]
        X_result = pd.DataFrame(X_selected, index=X.index, columns=selected_columns)
        
        return X_result, selected_features
    
    def _tree_importance(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Tree model importance-based feature selection"""
        n_features = params.get('n_features', 100)
        random_state = params.get('random_state', 42)
        
        # Determine task typeselect model
        if self._is_classification_task(y):
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        selector = SelectFromModel(model, max_features=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support()
        
        selected_columns = X.columns[selected_features]
        X_result = pd.DataFrame(X_selected, index=X.index, columns=selected_columns)
        
        return X_result, selected_features
    
    def _ica_selection(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Independent component analysis feature selection"""
        n_features = min(params.get('n_features', 50), X.shape[1])
        random_state = params.get('random_state', 42)
        
        # Apply ICA
        ica = FastICA(n_components=n_features, random_state=random_state)
        X_selected = ica.fit_transform(X)
        
        # Create new column names
        selected_columns = [f'ICA_{i+1}' for i in range(n_features)]
        X_result = pd.DataFrame(X_selected, index=X.index, columns=selected_columns)
        
        # For ICA, all components are selected
        selected_features = np.ones(X.shape[1], dtype=bool)
        selected_features[n_features:] = False
        
        return X_result, selected_features
    
    def _percentile_selection(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Percentile feature selection"""
        percentile = params.get('percentile', 50)
        
        # Determine task typeselect scoring function
        if self._is_classification_task(y):
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support()
        
        selected_columns = X.columns[selected_features]
        X_result = pd.DataFrame(X_selected, index=X.index, columns=selected_columns)
        
        return X_result, selected_features
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """
        DEPRECATED: Use enhanced label processor instead
        This method is kept for backward compatibility
        """
        # **CRITICAL FIX: Use enhanced label processor for consistent detection**
        task_type = self.label_processor.detect_task_type(y)
        return task_type == 'classification' 