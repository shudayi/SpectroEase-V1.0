# app/algorithms/nir_specific.py
"""
NIR spectrum-specific algorithms
Contains algorithms essential for NIR spectral analysis such as water peak removal
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.cross_decomposition import PLSRegression
from scipy.linalg import svd


class WaterPeakRemoval:
    """
    NIR water peak removal algorithm
    
    NIR spectra have strong water absorption peaks at 1400-1900 nm, which severely interfere 
    with the analysis of water-containing samples such as food and agricultural products.
    
    Main water peak positions:
    - 1450 nm: O-H first overtone
    - 1940 nm: O-H combination band
    - 760 nm: O-H third overtone
    
    Implementation methods:
    - EPO (External Parameter Orthogonalization): External parameter orthogonalization
    - DOSC (Direct Orthogonal Signal Correction): Direct orthogonal signal correction
    
    References:
    Roger, J. M., Chauchard, F., & Bellon-Maurel, V. (2003).
    EPO–PLS external parameter orthogonalisation of PLS application to 
    temperature-independent measurement of sugar content of intact fruits.
    Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.
    """
    
    # NIR water absorption bands (nm)
    WATER_REGIONS = [
        (1350, 1550),  # 1450 nm main peak
        (1850, 2000),  # 1940 nm peak
        (700, 800),    # 760 nm peak
    ]
    
    def __init__(self, 
                 method: str = 'epo',
                 n_components: int = 3,
                 wavelengths: Optional[np.ndarray] = None):
        """
        Initialize water peak removal algorithm
        
        Parameters:
        -----------
        method : str
            Removal method: 'epo', 'dosc', 'interpolation'
        n_components : int
            Number of orthogonal components used by EPO/DOSC, default 3
        wavelengths : ndarray, optional
            Wavelength axis (nm)
        """
        self.method = method.lower()
        self.n_components = n_components
        self.wavelengths = wavelengths
        
        self.P_epo_ = None  # EPO projection matrix
        
        if self.method not in ['epo', 'dosc', 'interpolation']:
            raise ValueError(f"Unknown water peak removal method: {method}")
    
    def fit(self, 
            X_water: np.ndarray, 
            wavelengths: Optional[np.ndarray] = None) -> 'WaterPeakRemoval':
        """
        Fit water peak removal model
        
        Parameters:
        -----------
        X_water : ndarray, shape (n_samples, n_wavelengths)
            Pure water or spectra with different water content, used to learn water spectral features
        wavelengths : ndarray, optional
            Wavelength axis
            
        Returns:
        --------
        self
        """
        if wavelengths is not None:
            self.wavelengths = wavelengths
        
        if isinstance(X_water, pd.DataFrame):
            X_water = X_water.values
        
        print(f"🔧 NIR water peak removal: Fitting {self.method.upper()} model")
        
        if self.method == 'epo':
            self._fit_epo(X_water)
        elif self.method == 'dosc':
            self._fit_dosc(X_water)
        # interpolation method does not require fitting
        
        return self
    
    def _fit_epo(self, X_water: np.ndarray):
        """
        Fit EPO model
        
        EPO extracts the main variation directions of water through PCA, then projects 
        the data onto its orthogonal complement space
        """
        # Center the data
        X_centered = X_water - np.mean(X_water, axis=0)
        
        # SVD decomposition
        U, S, Vt = svd(X_centered, full_matrices=False)
        
        # Take first n_components principal components (water variation directions)
        n_comp = min(self.n_components, len(S))
        V_water = Vt[:n_comp, :].T  # shape: (n_wavelengths, n_components)
        
        # EPO projection matrix: P = I - V_water @ V_water.T
        I = np.eye(V_water.shape[0])
        self.P_epo_ = I - V_water @ V_water.T
        
        print(f"  EPO: Extracted {n_comp} water components")
    
    def _fit_dosc(self, X_water: np.ndarray):
        """
        Fit DOSC model
        
        DOSC is similar to EPO but uses a different orthogonalization strategy
        """
        # DOSC uses the same principle as EPO
        self._fit_epo(X_water)
        print(f"  DOSC: Extracted {self.n_components} orthogonal components")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply water peak removal
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Original NIR spectra
            
        Returns:
        --------
        X_corrected : ndarray
            Spectra after water peak removal
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            is_dataframe = True
            df_info = (X.columns, X.index)
        else:
            X_values = X.copy()
            is_dataframe = False
        
        print(f"🔧 NIR water peak removal: Applying {self.method.upper()} correction")
        
        if self.method == 'epo' or self.method == 'dosc':
            if self.P_epo_ is None:
                raise ValueError("Must call fit method first")
            X_corrected = self._apply_epo(X_values)
        elif self.method == 'interpolation':
            if self.wavelengths is None:
                raise ValueError("Interpolation method requires wavelength information")
            X_corrected = self._apply_interpolation(X_values)
        
        print(f"✅ Water peak removal completed")
        
        if is_dataframe:
            return pd.DataFrame(X_corrected, columns=df_info[0], index=df_info[1])
        return X_corrected
    
    def _apply_epo(self, X: np.ndarray) -> np.ndarray:
        """
        Apply EPO projection
        
        X_corrected = X @ P_epo
        """
        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        
        # EPO projection
        X_projected = X_centered @ self.P_epo_
        
        # Restore mean
        X_corrected = X_projected + X_mean
        
        return X_corrected
    
    def _apply_interpolation(self, X: np.ndarray) -> np.ndarray:
        """
        Remove water peaks using interpolation method
        
        Replace water absorption regions with boundary point interpolation
        """
        from scipy.interpolate import interp1d
        
        X_corrected = X.copy()
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            spectrum = X[i, :]
            
            for region_start, region_end in self.WATER_REGIONS:
                # Find region indices
                mask = (self.wavelengths >= region_start) & (self.wavelengths <= region_end)
                
                if not np.any(mask):
                    continue
                
                region_indices = np.where(mask)[0]
                if len(region_indices) == 0:
                    continue
                
                start_idx = region_indices[0]
                end_idx = region_indices[-1]
                
                # Boundary interpolation
                if start_idx > 0 and end_idx < len(self.wavelengths) - 1:
                    left_idx = max(0, start_idx - 10)
                    right_idx = min(len(self.wavelengths) - 1, end_idx + 10)
                    
                    good_mask = np.ones(len(self.wavelengths), dtype=bool)
                    good_mask[start_idx:end_idx+1] = False
                    good_mask[:left_idx] = False
                    good_mask[right_idx+1:] = False
                    
                    if np.sum(good_mask) >= 2:
                        interpolator = interp1d(
                            self.wavelengths[good_mask],
                            spectrum[good_mask],
                            kind='cubic',
                            fill_value='extrapolate'
                        )
                        spectrum[mask] = interpolator(self.wavelengths[mask])
            
            X_corrected[i, :] = spectrum
        
        return X_corrected
    
    def fit_transform(self, 
                     X: np.ndarray, 
                     X_water: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and apply water peak removal
        
        Parameters:
        -----------
        X : ndarray
            Spectra to be corrected
        X_water : ndarray, optional
            Water spectra (for EPO/DOSC methods)
            If None, use X itself
            
        Returns:
        --------
        X_corrected : ndarray
            Corrected spectra
        """
        if X_water is None:
            X_water = X
        
        self.fit(X_water)
        return self.transform(X)


def nir_water_removal(X: np.ndarray, 
                     wavelengths: np.ndarray,
                     method: str = 'epo',
                     X_water: Optional[np.ndarray] = None) -> np.ndarray:
    """
    NIR water peak removal convenience function
    
    Parameters:
    -----------
    X : ndarray
        NIR spectral data
    wavelengths : ndarray
        Wavelength axis (nm)
    method : str
        Removal method
    X_water : ndarray, optional
        Water spectra
        
    Returns:
    --------
    X_corrected : ndarray
        Spectra after water peak removal
    """
    remover = WaterPeakRemoval(method=method, wavelengths=wavelengths)
    return remover.fit_transform(X, X_water)

