# plugins/data_partitioning/advanced_splitter.py

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupShuffleSplit
from typing import Dict, Any, Tuple, List

class AdvancedSplitter(DataPartitioningAlgorithm):
    """Advanced data splitting plugin"""
    
    def get_name(self) -> str:
        return "Advanced Splitter"
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'method': {
                'type': 'str',
                'default': 'time_series',
                'options': ['time_series', 'group_shuffle', 'blocked'],
                'description': 'Advanced splitting method'
            },
            'n_splits': {
                'type': 'int',
                'default': 5,
                'description': 'Number of splits for cross-validation'
            },
            'test_size': {
                'type': 'float',
                'default': 0.2,
                'description': 'Test set size ratio'
            },
            'gap': {
                'type': 'int',
                'default': 0,
                'description': 'Gap between train and test for time series'
            },
            'max_train_size': {
                'type': 'int',
                'default': None,
                'description': 'Maximum training set size'
            }
        }
    
    def apply(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced data splitting algorithm"""
        method = params.get('method', 'time_series')
        n_splits = params.get('n_splits', 5)
        test_size = params.get('test_size', 0.2)
        gap = params.get('gap', 0)
        max_train_size = params.get('max_train_size', None)
        
        X = data.iloc[:, :-1]  # features
        y = data.iloc[:, -1]   # target variable
        
        if method == 'time_series':
            return self._time_series_split(X, y, n_splits, gap, max_train_size)
        elif method == 'group_shuffle':
            return self._group_shuffle_split(X, y, n_splits, test_size)
        elif method == 'blocked':
            return self._blocked_split(X, y, n_splits, test_size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                          n_splits: int, gap: int, max_train_size: int) -> Dict[str, Any]:
        """Time series splitting"""
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size
        )
        
        splits = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            splits.append({
                'fold': i + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            })
        
        return {
            'method': 'Time Series Split',
            'n_splits': n_splits,
            'splits': splits,
            'total_samples': len(X)
        }
    
    def _group_shuffle_split(self, X: pd.DataFrame, y: pd.Series,
                           n_splits: int, test_size: float) -> Dict[str, Any]:
        """Grouped random splitting"""
        # Assume first column is group identifier
        groups = X.iloc[:, 0] if len(X.columns) > 0 else np.arange(len(X))
        
        gss = GroupShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=42
        )
        
        splits = []
        for i, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            splits.append({
                'fold': i + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_groups': len(np.unique(groups[train_idx])),
                'test_groups': len(np.unique(groups[test_idx]))
            })
        
        return {
            'method': 'Group Shuffle Split',
            'n_splits': n_splits,
            'splits': splits,
            'total_groups': len(np.unique(groups))
        }
    
    def _blocked_split(self, X: pd.DataFrame, y: pd.Series,
                      n_splits: int, test_size: float) -> Dict[str, Any]:
        """Block splitting (suitable for continuous samples in spectral data)"""
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        # Calculate size of each block
        block_size = n_samples // n_splits
        
        splits = []
        for i in range(n_splits):
            # Calculate start and end positions of test set
            test_start = i * block_size
            test_end = min(test_start + n_test, n_samples)
            
            # Test set indices
            test_idx = np.arange(test_start, test_end)
            
            # Training set indices (excluding test set)
            train_idx = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n_samples)
            ])
            
            # If training set is too large, random sampling
            if len(train_idx) > n_train:
                train_idx = np.random.choice(train_idx, n_train, replace=False)
            
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            splits.append({
                'fold': i + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'block_start': test_start,
                'block_end': test_end
            })
        
        return {
            'method': 'Blocked Split',
            'n_splits': n_splits,
            'splits': splits,
            'block_size': block_size
        }

    def split_data(self, data: pd.DataFrame, params: dict) -> dict:
        """Split the data into training and test sets according to the selected method."""
        method = params.get('method', 'time_series')
        n_splits = params.get('n_splits', 5)
        test_size = params.get('test_size', 0.2)
        gap = params.get('gap', 0)
        max_train_size = params.get('max_train_size', None)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Only return first split (consistent with interface requirements)
        if method == 'time_series':
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, max_train_size=max_train_size)
            for train_idx, test_idx in tscv.split(X):
                return {
                    'X_train': X.iloc[train_idx],
                    'X_test': X.iloc[test_idx],
                    'y_train': y.iloc[train_idx],
                    'y_test': y.iloc[test_idx]
                }
        elif method == 'group_shuffle':
            groups = X.iloc[:, 0] if len(X.columns) > 0 else np.arange(len(X))
            gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
            for train_idx, test_idx in gss.split(X, y, groups):
                return {
                    'X_train': X.iloc[train_idx],
                    'X_test': X.iloc[test_idx],
                    'y_train': y.iloc[train_idx],
                    'y_test': y.iloc[test_idx]
                }
        elif method == 'blocked':
            n_samples = len(X)
            n_test = int(n_samples * test_size)
            n_train = n_samples - n_test
            block_size = n_samples // n_splits
            for i in range(n_splits):
                test_start = i * block_size
                test_end = min(test_start + n_test, n_samples)
                test_idx = np.arange(test_start, test_end)
                train_idx = np.concatenate([
                    np.arange(0, test_start),
                    np.arange(test_end, n_samples)
                ])
                if len(train_idx) > n_train:
                    train_idx = np.random.choice(train_idx, n_train, replace=False)
                return {
                    'X_train': X.iloc[train_idx],
                    'X_test': X.iloc[test_idx],
                    'y_train': y.iloc[train_idx],
                    'y_test': y.iloc[test_idx]
                }
        else:
            raise ValueError(f"Unknown method: {method}") 