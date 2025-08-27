#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SNV (Standard Normal Variate) preprocessing algorithm
For scatter correction of spectrum data
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
import logging

logger = logging.getLogger(__name__)

class SNVProcessor(PreprocessingAlgorithm):
    """Standard Normal Variate (SNV) Preprocessor
    
    SNV is a commonly used scatter correction method in spectrum analysis that standardizes
    each spectrum sample to eliminate the effects of multiplicative scattering and baseline drift.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "SNV Processor"
        self.description = "Eliminate multiplicative scattering effects and baseline drift"
        self.parameters = {
            'center': True,    # Whether to center data
            'scale': True,     # Whether to scale data
            'min_std': 1e-6,   # Minimum standard deviation to avoid division by zero
            'copy': True       # Whether to copy data
        }
        self.fitted = False
        
    def get_name(self) -> str:
        """Return algorithm name"""
        return self.name
        
    def set_parameters(self, **params):
        """
        Set algorithm parameters
        
        Args:
            **params: Parameter dictionary, including center, scale, min_std, copy parameters
        """
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
                
        # Validate parameters
        if self.parameters['min_std'] <= 0:
            raise ValueError("min_std must be positive")
            
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter settings
        
        Returns:
            Dict: Current parameter dictionary
        """
        return self.parameters.copy()
        
    def get_parameter_info(self) -> Dict:
        """Return parameter information."""
        return {
            'center': {
                'name': 'Center Data',
                'type': 'bool',
                'default': True,
                'description': 'Subtract the mean of each spectrum (centering).'
            },
            'scale': {
                'name': 'Scale Data',
                'type': 'bool', 
                'default': True,
                'description': 'Divide by the standard deviation of each spectrum (scaling).'
            },
            'min_std': {
                'name': 'Min Std Threshold',
                'type': 'float',
                'default': 1e-6,
                'min': 1e-10,
                'max': 1e-3,
                'description': 'Safety threshold to avoid division by zero.'
            },
            'copy': {
                'name': 'Copy Data',
                'type': 'bool',
                'default': True,
                'description': 'Create a copy of the input data to avoid modifying the original.'
            }
        }
    
    def apply(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        applying SNV transformation
        
        Args:
            data: input spectral data (samples × wavelengths)
            params: algorithm parameters（if None, use stored parameters）
            
        Returns:
            pd.DataFrame: data after SNV transformation
        """
        # Use provided params or fall back to stored parameters
        if params is None:
            params = self.parameters
        else:
            # Merge with stored parameters, giving priority to provided params
            merged_params = self.parameters.copy()
            merged_params.update(params)
            params = merged_params
        
        center = params.get('center', True)
        scale = params.get('scale', True)
        min_std = params.get('min_std', 1e-6)
        copy_data = params.get('copy', True)
        
        logger.info(f"Applying SNV transform: center={center}, scale={scale}, min_std={min_std}")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Available parameters keys: {list(params.keys())}")
        logger.info(f"All params values: {params}")
        
        if copy_data:
            data = data.copy()
        
        logger.info("Starting SNV transformation...")
        result = self._snv_transform(data.values, center=center, scale=scale, min_std=min_std)
        result_df = pd.DataFrame(result, index=data.index, columns=data.columns)
        
        logger.info(f"SNV transformation completed. Output shape: {result_df.shape}")
        
        return result_df
    
    def _snv_transform(self, spectra: np.ndarray, center: bool = True, 
                      scale: bool = True, min_std: float = 1e-6) -> np.ndarray:
        """
        core SNV transformation algorithm
        
        Args:
            spectra: spectral data array (n_samples, n_wavelengths)
            center: whether to center
            scale: whether to standardize
            min_std: minimum standard deviation threshold
            
        Returns:
            np.ndarray: spectral data after SNV transformation
        """
        snv_spectra = spectra.copy()
        n_samples, n_wavelengths = spectra.shape
        
        logger.debug(f"Processing {n_samples} spectra, each with {n_wavelengths} wavelength points.")
        
        for i in range(n_samples):
            spectrum = spectra[i, :]
            
            if center:
                mean_val = np.mean(spectrum)
                spectrum_centered = spectrum - mean_val
            else:
                spectrum_centered = spectrum
                
            if scale:
                std_val = np.std(spectrum_centered)
                
                if std_val < min_std:
                    logger.warning(f"Standard deviation of sample {i} ({std_val:.2e}) is below threshold ({min_std:.2e}). Using threshold value.")
                    std_val = min_std
                    
                snv_spectra[i, :] = spectrum_centered / std_val
            else:
                snv_spectra[i, :] = spectrum_centered
                
        return snv_spectra
    
    def validate_parameters(self, params: Dict) -> Tuple[bool, str]:
        """
        validate parameter validity
        
        Args:
            params: parameters dictionary
            
        Returns:
            Tuple[bool, str]: (whether valid, error message)
        """
        try:
            min_std = params.get('min_std', 1e-6)
            if not isinstance(min_std, (int, float)):
                return False, "Minimum standard deviation must be a number."
            
            # If min_std is 0, automatically set it to a small positive value
            if min_std <= 0:
                params['min_std'] = 1e-6
                logger.warning(f"min_std was {min_std}, automatically set to 1e-6")
                
            if min_std > 1e-3:
                return False, "Minimum standard deviation should not exceed 1e-3."
                
            return True, ""
            
        except Exception as e:
            return False, f"Parameter validation failed: {str(e)}"
    
    def get_description(self) -> str:
        """Return detailed algorithm description"""
        return """
        Standard Normal Variate (SNV) is a key scatter correction method in spectral preprocessing.
        
        How it works:
        1. Standardizes each spectrum sample individually.
        2. Subtracts the mean of that sample (centering).
        3. Divides by the standard deviation of that sample (scaling).
        
        Effects:
        - Eliminates multiplicative scatter effects.
        - Reduces baseline drift.
        - Emphasizes spectral shape features.
        - Improves the accuracy of spectral comparisons.
        
        Use cases:
        - Diffuse reflectance spectra of solid samples.
        - Near-infrared (NIR) spectral analysis.
        - Raman spectroscopy preprocessing.
        - Other spectral techniques with scatter interference.
        """
    
    def get_example_usage(self) -> str:
        """Return usage examples"""
        return """
        # Basic usage
        snv_processor = SNVProcessor()
        processed_data = snv_processor.apply(spectra_data, {
            'center': True,
            'scale': True,
            'min_std': 1e-6
        })
        
        # Centering only, no scaling
        processed_data = snv_processor.apply(spectra_data, {
            'center': True,
            'scale': False
        })
        
        # Custom minimum standard deviation threshold
        processed_data = snv_processor.apply(spectra_data, {
            'center': True,
            'scale': True,
            'min_std': 1e-5
        })
        """

def test_snv_processor():
    """Test the SNV processor"""
    print("=" * 50)
    print("Testing SNV Processor")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    n_samples, n_wavelengths = 10, 100
    
    wavelengths = np.linspace(400, 2500, n_wavelengths)
    base_spectrum = np.exp(-((wavelengths - 1000) / 200) ** 2)  # Gaussian peak
    
    spectra = []
    for i in range(n_samples):
        baseline = 0.1 + 0.2 * np.random.random()
        multiplier = 0.8 + 0.4 * np.random.random()
        noise = 0.01 * np.random.normal(size=n_wavelengths)
        
        spectrum = baseline + multiplier * base_spectrum + noise
        spectra.append(spectrum)
    
    spectra = np.array(spectra)
    data = pd.DataFrame(spectra, columns=[f"λ{w:.0f}" for w in wavelengths])
    
    print(f"Original data statistics:")
    print(f"Shape: {data.shape}")
    mean_values = data.mean(axis=1)
    std_values = data.std(axis=1)
    print(f"Mean range: [{float(mean_values.min()):.3f}, {float(mean_values.max()):.3f}]")
    print(f"Standard deviation range: [{float(std_values.min()):.3f}, {float(std_values.max()):.3f}]")
    
    # Create SNV processor and apply
    snv_processor = SNVProcessor()
    
    # Test default parameters
    processed_data = snv_processor.apply(data, {})
    
    print(f"\nSNV processed statistics:")
    print(f"Shape: {processed_data.shape}")
    print(f"Mean range: [{processed_data.mean(axis=1).min():.6f}, {processed_data.mean(axis=1).max():.6f}]")
    print(f"Standard deviation range: [{processed_data.std(axis=1).min():.6f}, {processed_data.std(axis=1).max():.6f}]")
    
    # Validate SNV effect
    means = processed_data.mean(axis=1)
    stds = processed_data.std(axis=1)
    
    print(f"\nVerifying SNV effect:")
    print(f"All sample means are close to 0: {np.allclose(means, 0, atol=1e-10)}")
    print(f"All sample standard deviations are close to 1: {np.allclose(stds, 1, atol=1e-10)}")
    
    print("\nSNV processor test complete!")

if __name__ == "__main__":
    test_snv_processor() 