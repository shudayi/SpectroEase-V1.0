# app/algorithms/baseline_correction.py
"""
Advanced baseline correction algorithms
Supplementing existing Polynomial, ALS, airPLS
"""

import numpy as np
import pandas as pd
from typing import Optional


class SNIPBaseline:
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP)
    
    SNIP is a baseline correction algorithm designed for multi-peak spectra, 
    particularly suitable for complex Raman and XRF spectra.
    
    Advantages:
    - Preserves peak shape
    - Suitable for multi-peak overlapping spectra
    - No parameter tuning required
    - Fast computation
    
    Applications:
    - Raman spectra (multi-peak)
    - XRF/XRD spectra
    - Complex biological sample spectra
    
    References:
    Ryan, C. G., Clayton, E., Griffin, W. L., et al. (1988).
    SNIP, a statistics-sensitive background treatment for the 
    quantitative analysis of PIXE spectra in geoscience applications.
    Nuclear Instruments and Methods in Physics Research B, 34(3), 396-402.
    
    Morháč, M., Kliman, J., Matoušek, V., et al. (1997).
    Background elimination methods for multidimensional coincidence 
    γ-ray spectra. Nuclear Instruments and Methods in Physics Research A, 
    401(1), 113-132.
    """
    
    def __init__(self, 
                 max_half_width: int = 40,
                 decreasing: bool = True,
                 smooth_half_width: Optional[int] = None):
        """
        Initialize SNIP algorithm
        
        Parameters:
        -----------
        max_half_width : int
            Maximum half-width, controls baseline smoothness
            Default 40, recommended range: 20-100
            - Small value: Baseline fits more tightly, suitable for narrow peaks
            - Large value: Baseline is smoother, suitable for wide peaks
        decreasing : bool
            Whether to use decreasing window (recommended True)
        smooth_half_width : int, optional
            Additional smoothing parameter, None means no smoothing
        """
        self.max_half_width = max_half_width
        self.decreasing = decreasing
        self.smooth_half_width = smooth_half_width
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SNIP baseline correction
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Original spectra
            
        Returns:
        --------
        X_corrected : ndarray
            Baseline-corrected spectra
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            is_dataframe = True
            df_info = (X.columns, X.index)
        else:
            X_values = X.copy()
            is_dataframe = False
        
        n_samples, n_points = X_values.shape
        X_corrected = np.zeros_like(X_values)
        
        print(f"🔧 SNIP baseline correction: Processing {n_samples} spectra")
        print(f"  Maximum half-width: {self.max_half_width}")
        
        for i in range(n_samples):
            spectrum = X_values[i, :]
            baseline = self._snip_baseline(spectrum)
            X_corrected[i, :] = spectrum - baseline
            
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  Processed {i+1}/{n_samples} spectra")
        
        print(f"✅ SNIP baseline correction completed")
        
        if is_dataframe:
            return pd.DataFrame(X_corrected, columns=df_info[0], index=df_info[1])
        return X_corrected
    
    def _snip_baseline(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Calculate SNIP baseline for a single spectrum
        
        Parameters:
        -----------
        spectrum : ndarray
            Single spectrum data
            
        Returns:
        --------
        baseline : ndarray
            SNIP baseline
        """
        n = len(spectrum)
        
        # Transform to logarithmic space (SNIP works in logarithmic space)
        # Add small constant to avoid log(0)
        eps = np.abs(np.min(spectrum)) + 1e-6 if np.min(spectrum) < 0 else 1e-6
        spectrum_log = np.log(np.log(np.sqrt(spectrum + eps + 1) + 1) + 1)
        
        # Iterative peak clipping
        baseline_log = spectrum_log.copy()
        
        if self.decreasing:
            # Decreasing window
            for width in range(self.max_half_width, 0, -1):
                for i in range(width, n - width):
                    # SNIP algorithm core
                    left_val = baseline_log[i - width]
                    right_val = baseline_log[i + width]
                    average = (left_val + right_val) / 2.0
                    
                    # Peak clipping: if current point is above average of neighbors, reduce it
                    if baseline_log[i] > average:
                        baseline_log[i] = average
        else:
            # Increasing window
            for width in range(1, self.max_half_width + 1):
                for i in range(width, n - width):
                    left_val = baseline_log[i - width]
                    right_val = baseline_log[i + width]
                    average = (left_val + right_val) / 2.0
                    
                    if baseline_log[i] > average:
                        baseline_log[i] = average
        
        # Additional smoothing (optional)
        if self.smooth_half_width is not None and self.smooth_half_width > 0:
            for width in range(1, self.smooth_half_width + 1):
                baseline_smoothed = baseline_log.copy()
                for i in range(width, n - width):
                    baseline_smoothed[i] = (baseline_log[i - width] + baseline_log[i + width]) / 2.0
                baseline_log = baseline_smoothed
        
        # Transform back from logarithmic space to original space
        baseline = (np.exp(np.exp(baseline_log) - 1) - 1) ** 2 - eps
        
        # Ensure baseline does not exceed original spectrum
        baseline = np.minimum(baseline, spectrum)
        
        return baseline


def snip_baseline_correction(X: np.ndarray, 
                             max_half_width: int = 40) -> np.ndarray:
    """
    SNIP baseline correction convenience function
    
    Parameters:
    -----------
    X : ndarray
        Original spectrum data
    max_half_width : int
        Maximum half-width
        
    Returns:
    --------
    X_corrected : ndarray
        Baseline-corrected spectra
    """
    snip = SNIPBaseline(max_half_width=max_half_width)
    return snip.fit_transform(X)

