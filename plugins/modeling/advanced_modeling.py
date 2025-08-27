# plugins/modeling/advanced_modeling.py

from interfaces.modeling_algorithm import ModelingAlgorithm
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Tuple, Optional
from app.utils.label_processor import EnhancedLabelProcessor

class AdvancedModeling(ModelingAlgorithm):
    """Advanced modeling plugin"""
    
    def __init__(self):

        self.label_processor = EnhancedLabelProcessor()
    
    def get_name(self) -> str:
        return "Advanced Modeling"
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'method': {
                'type': 'str',
                'default': 'ensemble_voting',
                'options': ['ensemble_voting', 'neural_network', 'gaussian_process', 
                           'bagging', 'adaboost', 'extra_trees'],
                'description': 'Advanced modeling method'
            },
            'task_type': {
                'type': 'str',
                'default': 'auto',
                'options': ['auto', 'classification', 'regression'],
                'description': 'Task type (auto-detect if not specified)'
            },
            'n_estimators': {
                'type': 'int',
                'default': 100,
                'description': 'Number of estimators for ensemble methods'
            },
            'hidden_layer_sizes': {
                'type': 'tuple',
                'default': (100, 50),
                'description': 'Hidden layer sizes for neural network'
            },
            'max_iter': {
                'type': 'int',
                'default': 1000,
                'description': 'Maximum iterations for iterative algorithms'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random state for reproducibility'
            },
            'voting': {
                'type': 'str',
                'default': 'soft',
                'options': ['hard', 'soft'],
                'description': 'Voting method for ensemble (classification only)'
            }
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Any:
        """Train model"""
        method = params.get('method', 'ensemble_voting')
        task_type = params.get('task_type', 'auto')
        

        if task_type == 'auto':
            task_type = self._detect_task_type(y)
        
        if method == 'ensemble_voting':
            return self._train_ensemble_voting(X, y, params, task_type)
        elif method == 'neural_network':
            return self._train_neural_network(X, y, params, task_type)
        elif method == 'gaussian_process':
            return self._train_gaussian_process(X, y, params, task_type)
        elif method == 'bagging':
            return self._train_bagging(X, y, params, task_type)
        elif method == 'adaboost':
            return self._train_adaboost(X, y, params, task_type)
        elif method == 'extra_trees':
            return self._train_extra_trees(X, y, params, task_type)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Predict"""
        return model.predict(X)
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict probability (only for classification tasks)"""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Enhanced task type detection using the label processor
        
        Args:
            y: Target variable
            
        Returns:
            str: Task type ('classification' or 'regression')
        """

        return self.label_processor.detect_task_type(y)
    
    def _train_ensemble_voting(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train ensemble voting model"""
        random_state = params.get('random_state', 42)
        voting = params.get('voting', 'soft')
        
        if task_type == 'classification':
       
            estimators = [
                ('lr', LogisticRegression(random_state=random_state, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state)),
                ('nb', GaussianNB()),
                ('dt', DecisionTreeClassifier(random_state=random_state))
            ]
            
            model = VotingClassifier(estimators=estimators, voting=voting)
        else:
       
            estimators = [
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=50, random_state=random_state)),
                ('ridge', Ridge(random_state=random_state)),
                ('dt', DecisionTreeRegressor(random_state=random_state))
            ]
            
            model = VotingRegressor(estimators=estimators)
        
        model.fit(X, y)
        return model
    
    def _train_neural_network(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train neural network model"""
        hidden_layer_sizes = params.get('hidden_layer_sizes', (100, 50))
        max_iter = params.get('max_iter', 1000)
        random_state = params.get('random_state', 42)
        
        if task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        model.fit(X, y)
        return model
    
    def _train_gaussian_process(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train Gaussian process model"""
        random_state = params.get('random_state', 42)
        
        if task_type == 'classification':
            model = GaussianProcessClassifier(random_state=random_state)
        else:
            model = GaussianProcessRegressor(random_state=random_state)
        
        model.fit(X, y)
        return model
    
    def _train_bagging(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train Bagging model"""
        n_estimators = params.get('n_estimators', 100)
        random_state = params.get('random_state', 42)
        
        if task_type == 'classification':
            base_estimator = DecisionTreeClassifier(random_state=random_state)
            model = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                random_state=random_state
            )
        else:
            base_estimator = DecisionTreeRegressor(random_state=random_state)
            model = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                random_state=random_state
            )
        
        model.fit(X, y)
        return model
    
    def _train_adaboost(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train AdaBoost model"""
        n_estimators = params.get('n_estimators', 100)
        random_state = params.get('random_state', 42)
        
        if task_type == 'classification':
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        else:
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                random_state=random_state
            )
        
        model.fit(X, y)
        return model
    
    def _train_extra_trees(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], task_type: str) -> Any:
        """Train Extra Trees model"""
        n_estimators = params.get('n_estimators', 100)
        random_state = params.get('random_state', 42)
        
        if task_type == 'classification':
            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=random_state
            )
        
        model.fit(X, y)
        return model 