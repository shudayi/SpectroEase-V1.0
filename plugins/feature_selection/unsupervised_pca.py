# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm

class UnsupervisedPCAPlugin(FeatureSelectionAlgorithm):
    """
    A refactored PCA plugin that correctly implements the FeatureSelectionAlgorithm
    interface and can be used for both exploratory and modeling purposes.
    """
    def get_name(self) -> str:
        return "PCA"

    def get_params_info(self) -> Dict[str, Any]:
        """Correctly implements the interface method."""
        return {
            'n_components': {
                'type': 'int',
                'default': 10,
                'description': 'Number of principal components to keep.'
            }
        }

    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """
        Core PCA execution logic, adapted to the select_features interface method.
        The presence of 'X_test' in kwargs determines the mode.
        """
        n_components = kwargs.get('n_components', 10)
        X_test = kwargs.get('X_test')

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_train_transformed = pca.fit_transform(X_train_scaled)

        results = {
            'model': pca,
            'scaler': scaler,
            'X_train_transformed': pd.DataFrame(X_train_transformed, index=X.index, columns=[f'PC{i+1}' for i in range(n_components)])
        }

        if X_test is not None:
            # Modeling mode
            X_test_scaled = scaler.transform(X_test)
            X_test_transformed = pca.transform(X_test_scaled)
            results['X_test_transformed'] = pd.DataFrame(X_test_transformed, index=X_test.index, columns=[f'PC{i+1}' for i in range(n_components)])
        else:
            # Exploratory mode
            results['explained_variance_ratio'] = pca.explained_variance_ratio_
            results['cumulative_explained_variance'] = np.cumsum(pca.explained_variance_ratio_)
            results['loadings'] = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=X.columns)
            results['scores'] = results['X_train_transformed']

        return results

    def get_recommendations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Provides recommendations for the number of components in a modeling context.
        """
        n_classes = len(np.unique(y))
        recommended_k = min(X.shape[1], n_classes * 3)
        return {'n_components': recommended_k}