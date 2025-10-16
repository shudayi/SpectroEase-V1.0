import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
import pandas as pd

class SpectrumPreprocessor(PreprocessingAlgorithm):
    """Spectrum data preprocessor"""
    
    def get_name(self) -> str:
        """Return algorithm name"""
        return "Spectrum Preprocessor"
        
    def get_params_info(self) -> Dict[str, any]:
        """Return parameter information"""
        return {
            'baseline_correction': {
                'name': 'Baseline Correction',
                'type': 'bool',
                'default': True,
                'description': 'Whether to perform baseline correction'
            },
            'poly_order': {
                'name': 'Polynomial Order',
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 5,
                'description': 'Polynomial order for baseline correction'
            },
            'smoothing': {
                'name': 'Smoothing',
                'type': 'bool',
                'default': False,
                'description': 'Whether to perform smoothing processing'
            },
            'window_length': {
                'name': 'Window Length',
                'type': 'int',
                'default': 5,
                'min': 3,
                'max': 21,
                'description': 'Smoothing window length'
            },
            'normalization': {
                'name': 'Normalization',
                'type': 'str',
                'default': 'none',
                'options': ['none', 'minmax', 'l2'],
                'description': 'Normalization method'
            },
            'outlier_detection': {
                'name': 'Outlier Detection',
                'type': 'bool',
                'default': False,
                'description': 'Whether to perform outlier detection'
            },
            'threshold': {
                'name': 'Threshold',
                'type': 'float',
                'default': 2.0,
                'min': 1.0,
                'max': 5.0,
                'description': 'Standard deviation threshold for outlier detection'
            }
        }
    
    def apply(self, data: pd.DataFrame, params: Dict[str, any]) -> pd.DataFrame:
        """
        Apply preprocessing algorithm
        
        Args:
            data: Input data
            params: Parameter dictionary
            
        Returns:
            pd.DataFrame: Processed data
        """
        
        processed_data = data.copy()
        
        # Baseline correction
        if params.get('baseline_correction', True):
            processed_data = self._baseline_correction(processed_data, params.get('poly_order', 2))
        
        # Smoothing
        if params.get('smoothing', False):
            processed_data = self._smooth_data(processed_data, params.get('window_length', 5))
        
        # Normalization
        norm_method = params.get('normalization', 'none')
        if norm_method != 'none':
            processed_data = self._normalize_data(processed_data, norm_method)
        
        # Outlier detection and removal
        if params.get('outlier_detection', False):
            processed_data = self._remove_outliers(processed_data, params.get('threshold', 2.0))
        
        return processed_data
    
    def _baseline_correction(self, data, poly_order):
        """
        Polynomial baseline correction
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        processed_data = data.copy()
        wavelengths = data.columns.astype(float)
        
        for idx in data.index:
            spectrum = data.loc[idx].values
            
            # Asymmetric least squares baseline correction
            baseline = self._als_baseline(spectrum, lam=1e6, p=0.01)
            processed_data.loc[idx] = spectrum - baseline
        
        return processed_data
    
    def _smooth_data(self, data, window_length):
        """
        Savitzky-Golay smoothing
        """
        processed_data = data.copy()
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window_length does not exceed data length
        window_length = min(window_length, data.shape[1])
        if window_length < 3:
            window_length = 3
            
        for idx in data.index:
            spectrum = data.loc[idx].values
            smoothed_spectrum = savgol_filter(spectrum, window_length, 3)
            processed_data.loc[idx] = smoothed_spectrum
            
        return processed_data
    
    def _normalize_data(self, data, method):
        """
        Normalize spectrum data using row-wise operations suitable for spectra.
        """
        processed_data = data.copy()
        data_values = data.values

        if method == 'minmax':
            # Perform row-wise Min-Max scaling
            min_vals = np.min(data_values, axis=1, keepdims=True)
            max_vals = np.max(data_values, axis=1, keepdims=True)
            # Avoid division by zero for flat spectra
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            normalized_values = (data_values - min_vals) / range_vals
        elif method == 'standard':
            # Perform row-wise Z-Score (Standard) scaling
            mean_vals = np.mean(data_values, axis=1, keepdims=True)
            std_vals = np.std(data_values, axis=1, keepdims=True)
            # Avoid division by zero for flat spectra
            std_vals[std_vals == 0] = 1
            normalized_values = (data_values - mean_vals) / std_vals
        elif method == 'l2':
            # Vector normalization (L2 norm)
            norm_vals = np.linalg.norm(data_values, axis=1, keepdims=True)
            norm_vals[norm_vals == 0] = 1 # Avoid division by zero
            normalized_values = data_values / norm_vals
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        # Put standardized data back into DataFrame
        processed_data.iloc[:, :] = normalized_values
        return processed_data
    
    def _remove_outliers(self, data, threshold):
        """
        Outlier detection and removal
        """
        processed_data = data.copy()
        data_values = data.values
        
        mean = np.mean(data_values, axis=1, keepdims=True)
        std = np.std(data_values, axis=1, keepdims=True)
        
        mask = np.abs(data_values - mean) > threshold * std
        
        # Replace outliers with mean
        processed_values = data_values.copy()
        processed_values[mask] = np.broadcast_to(mean, data_values.shape)[mask]
        
        # Put processed data back into DataFrame
        processed_data.iloc[:, :] = processed_values
        return processed_data
    
    def _als_baseline(self, y, lam=1e6, p=0.01, niter=10):
        """
        Asymmetric Least Squares baseline correction
        
        Args:
            y: Input spectrum
            lam: Smoothness parameter (larger values make baseline smoother)
            p: Asymmetry parameter (0 < p < 1, smaller values give more asymmetry)
            niter: Number of iterations
            
        Returns:
            np.ndarray: Baseline
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        L = len(y)
        # Construct difference matrix
        D = sparse.diags(diagonals=[1, -2, 1], offsets=[0, 1, 2], shape=(L-2, L))
        
        # Initialize weights
        w = np.ones(L)
        z = np.zeros_like(y)
        
        for i in range(niter):
            # Construct weight matrix
            W = sparse.diags(w, 0, shape=(L, L))
            
            # Solve the system: (W + lam * D'D) * z = W * y
            Z = W + lam * D.T.dot(D)
            z = spsolve(Z, w * y)
            
            # Update weights
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z