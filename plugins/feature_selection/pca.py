# plugins/feature_selection/pca.py

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple

class PCA(FeatureSelectionAlgorithm):
    """Principal component analysis feature selection plugin"""
    
    def get_name(self) -> str:
        return "PCA Feature Selection"
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'n_components': {
                'type': 'int',
                'default': None,
                'description': 'Number of components to keep (None for auto)'
            },
            'variance_ratio': {
                'type': 'float',
                'default': 0.95,
                'description': 'Cumulative variance ratio to retain'
            },
            'whiten': {
                'type': 'bool',
                'default': False,
                'description': 'Whether to whiten the components'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random state for reproducibility'
            }
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Use PCA for feature selection/dimensionality reduction
        
        Note: PCA is unsupervised and doesn't require y, but we keep the interface consistent
        """
        n_components = params.get('n_components', None)
        variance_ratio = params.get('variance_ratio', 0.95)
        whiten = params.get('whiten', False)
        random_state = params.get('random_state', 42)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # If no component count specified, automatically determine based on variance ratio
        if n_components is None:
            # First perform PCA with all components
            pca_temp = SklearnPCA()
            pca_temp.fit(X_scaled)
            
            # Calculate cumulative variance ratio
            cumsum_ratio = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_ratio >= variance_ratio) + 1
            
            print(f"🔧 PCA auto-selected {n_components} components for {variance_ratio*100}% variance")
        
        # Perform PCA
        pca = SklearnPCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )
        
        X_transformed = pca.fit_transform(X_scaled)
        
        # Create new column names
        component_names = [f'PC{i+1}' for i in range(n_components)]
        X_result = pd.DataFrame(X_transformed, index=X.index, columns=component_names)
        
        # For PCA, all principal components are "selected"
        selected_features = np.ones(n_components, dtype=bool)
        
        print(f"✅ PCA completed: {X.shape[1]} → {n_components} components")
        print(f"📊 Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_result, selected_features