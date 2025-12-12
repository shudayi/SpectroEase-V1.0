# plugins/data_partitioning/stratified_splitter.py

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Any, Tuple

# Add project root to Python path for interface import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm

class StratifiedSplitter(DataPartitioningAlgorithm):
    """
    A data partitioning plugin that performs stratified splitting, suitable for
    quantitative (regression) tasks by binning the continuous target variable.
    """
    
    def get_name(self) -> str:
        return "Stratified Split"
    
    def get_params_info(self) -> Dict[str, Any]:
        """返回参数信息（接口要求的方法名）"""
        return {
            'test_size': {
                'type': 'float',
                'default': 0.2,
                'description': 'Proportion of the dataset to include in the test split.'
            },
            'n_bins': {
                'type': 'int',
                'default': 5,
                'description': 'Number of bins to create from the continuous target variable for stratification.'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Seed for the random number generator for reproducibility.'
            }
        }
    
    def partition(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """数据分割（接口要求的方法名）- 返回单次分割结果"""
        result = self._apply_internal(data, params)
        return result['X_train'], result['X_test'], result['y_train'], result['y_test']
    
    def _apply_internal(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply stratified splitting based on a binned continuous target variable (internal).
        """
        test_size = params.get('test_size', 0.2)
        n_bins = params.get('n_bins', 5)
        random_state = params.get('random_state', 42)
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Stratified Split requires a numeric target variable for binning.")

        # Create bins from the continuous target variable
        # Using qcut to ensure each bin has a similar number of samples
        try:
            y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        except ValueError as e:
            raise ValueError(f"Could not create {n_bins} bins. Try reducing the number of bins. Original error: {e}")

        # Perform stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        
        try:
            train_idx, test_idx = next(sss.split(X, y_binned))
        except ValueError:
             # Fallback for cases where stratification is not possible (e.g., a bin has only one member)
             # This is a simple random split, but it's better than crashing.
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'details': {
                    'method': self.get_name(),
                    'warning': 'Could not perform stratified split, fell back to random split.',
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'total_samples': len(data),
                    'training_ratio': len(X_train) / len(data),
                    'test_ratio': len(X_test) / len(data),
                }
            }

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'details': {
                'method': self.get_name(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(data),
                'training_ratio': len(X_train) / len(data),
                'test_ratio': len(X_test) / len(data),
                'stratification_bins': n_bins
            }
        }