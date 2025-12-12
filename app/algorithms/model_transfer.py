# app/algorithms/model_transfer.py
"""
Model transfer algorithms
Allow sharing calibration models between different instruments, a key technology for industrial applications
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.linalg import lstsq
from sklearn.decomposition import PCA


class PDS:
    """
    Piecewise Direct Standardization (PDS)
    
    PDS is the most commonly used model transfer algorithm. It establishes a linear transfer function 
    from slave instrument to master instrument, allowing spectra measured by different instruments 
    to use the same calibration model.
    
    Advantages:
    - Does not require large number of standard samples
    - High transfer accuracy
    - High computational efficiency
    - Applicable to various spectral types
    
    Applications:
    - Model sharing between different brands/models of instruments
    - Model updates after instrument aging
    - Collaborative analysis with multiple instruments
    - Integration of portable and laboratory instruments
    
    References:
    Wang, Y., Veltkamp, D. J., & Kowalski, B. R. (1991).
    Multivariate instrument standardization.
    Analytical Chemistry, 63(23), 2750-2756.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize PDS algorithm
        
        Parameters:
        -----------
        window_size : int
            Transfer window size, each wavelength uses window_size wavelengths before and after 
            to establish transfer relationship
            Default 5, recommended range 3-11 (odd number)
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd number
        self.window_size = window_size
        
        self.B_ = None  # Transfer matrix
        self.master_mean_ = None
        self.slave_mean_ = None
        
    def fit(self, 
            X_master: np.ndarray, 
            X_slave: np.ndarray) -> 'PDS':
        """
        Build transfer model
        
        Parameters:
        -----------
        X_master : ndarray, shape (n_samples, n_wavelengths)
            Master instrument spectra (reference instrument)
        X_slave : ndarray, shape (n_samples, n_wavelengths)
            Slave instrument spectra (instrument to be standardized)
            
        Note: X_master and X_slave must be measurements of the same set of samples on different instruments
        
        Returns:
        --------
        self
        """
        if isinstance(X_master, pd.DataFrame):
            X_master = X_master.values
        if isinstance(X_slave, pd.DataFrame):
            X_slave = X_slave.values
        
        if X_master.shape != X_slave.shape:
            raise ValueError("Master and slave instrument spectra must have the same shape")
        
        n_samples, n_wavelengths = X_master.shape
        
        if n_samples < 10:
            print(f"⚠ Warning: Few standard samples ({n_samples}), recommend at least 20 samples for better transfer effect")
        
        print(f"🔧 PDS model transfer: Building transfer model")
        print(f"  Standard samples: {n_samples}")
        print(f"  Wavelengths: {n_wavelengths}")
        print(f"  Window size: {self.window_size}")
        
        # Center the data
        self.master_mean_ = np.mean(X_master, axis=0)
        self.slave_mean_ = np.mean(X_slave, axis=0)
        
        X_master_centered = X_master - self.master_mean_
        X_slave_centered = X_slave - self.slave_mean_
        
        # Build transfer relationship for each wavelength
        half_window = self.window_size // 2
        self.B_ = np.zeros((n_wavelengths, n_wavelengths))
        
        for i in range(n_wavelengths):
            # Determine window range
            start = max(0, i - half_window)
            end = min(n_wavelengths, i + half_window + 1)
            
            # Slave instrument spectra within window
            X_window = X_slave_centered[:, start:end]
            
            # Master instrument value at current wavelength
            y_target = X_master_centered[:, i]
            
            # Solve transfer coefficients using least squares
            if X_window.shape[1] > 0:
                coeffs, _, _, _ = lstsq(X_window, y_target)
                self.B_[i, start:end] = coeffs
        
        print(f"✅ PDS transfer model building completed")
        
        return self
    
    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        """
        Apply transfer model to standardize slave instrument spectra to master instrument
        
        Parameters:
        -----------
        X_slave : ndarray
            New spectra measured by slave instrument
            
        Returns:
        --------
        X_standardized : ndarray
            Standardized spectra (equivalent to master instrument measurements)
        """
        if self.B_ is None:
            raise ValueError("Must call fit method first to build transfer model")
        
        if isinstance(X_slave, pd.DataFrame):
            X_values = X_slave.values
            is_dataframe = True
            df_info = (X_slave.columns, X_slave.index)
        else:
            X_values = X_slave.copy()
            is_dataframe = False
        
        # Center the data
        X_centered = X_values - self.slave_mean_
        
        # Apply transfer matrix
        X_standardized_centered = X_centered @ self.B_.T
        
        # Restore master instrument mean
        X_standardized = X_standardized_centered + self.master_mean_
        
        print(f"✅ PDS transfer completed: {X_values.shape[0]} spectra standardized")
        
        if is_dataframe:
            return pd.DataFrame(X_standardized, columns=df_info[0], index=df_info[1])
        return X_standardized
    
    def fit_transform(self, X_master: np.ndarray, X_slave: np.ndarray) -> np.ndarray:
        """Build model and standardize"""
        self.fit(X_master, X_slave)
        return self.transform(X_slave)


class SBC:
    """
    Slope and Bias Correction (SBC)
    
    SBC is a simple but effective model transfer method that eliminates systematic differences 
    between instruments through linear correction.
    
    Correction model:
    X_standardized = slope * X_slave + bias
    
    Advantages:
    - Simple and fast computation
    - Requires fewer standard samples
    - Clear physical meaning
    - Suitable for preliminary transfer
    
    Limitations:
    - Can only correct linear differences
    - Less accurate than PDS
    
    References:
    Bouveresse, E., & Massart, D. L. (1996).
    Improvement of the piecewise direct standardisation procedure 
    for the transfer of NIR spectra for multivariate calibration.
    Chemometrics and Intelligent Laboratory Systems, 32(2), 201-213.
    """
    
    def __init__(self):
        """Initialize SBC algorithm"""
        self.slope_ = None
        self.bias_ = None
        
    def fit(self, X_master: np.ndarray, X_slave: np.ndarray) -> 'SBC':
        """
        Build slope and bias correction model
        
        Parameters:
        -----------
        X_master : ndarray, shape (n_samples, n_wavelengths)
            Master instrument spectra
        X_slave : ndarray, shape (n_samples, n_wavelengths)
            Slave instrument spectra
            
        Returns:
        --------
        self
        """
        if isinstance(X_master, pd.DataFrame):
            X_master = X_master.values
        if isinstance(X_slave, pd.DataFrame):
            X_slave = X_slave.values
        
        if X_master.shape != X_slave.shape:
            raise ValueError("Master and slave instrument spectra must have the same shape")
        
        n_samples, n_wavelengths = X_master.shape
        
        print(f"🔧 SBC model transfer: Building correction model")
        print(f"  Standard samples: {n_samples}")
        print(f"  Wavelengths: {n_wavelengths}")
        
        # Calculate slope and bias for each wavelength
        self.slope_ = np.zeros(n_wavelengths)
        self.bias_ = np.zeros(n_wavelengths)
        
        for i in range(n_wavelengths):
            y = X_master[:, i]
            x = X_slave[:, i]
            
            # Linear regression: y = slope * x + bias
            # Using least squares method
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator > 1e-10:
                self.slope_[i] = numerator / denominator
                self.bias_[i] = y_mean - self.slope_[i] * x_mean
            else:
                self.slope_[i] = 1.0
                self.bias_[i] = 0.0
        
        print(f"✅ SBC correction model building completed")
        print(f"  Average slope: {np.mean(self.slope_):.4f}")
        print(f"  Average bias: {np.mean(self.bias_):.4f}")
        
        return self
    
    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        """
        Apply slope and bias correction
        
        Parameters:
        -----------
        X_slave : ndarray
            New spectra measured by slave instrument
            
        Returns:
        --------
        X_standardized : ndarray
            Standardized spectra
        """
        if self.slope_ is None or self.bias_ is None:
            raise ValueError("Must call fit method first to build correction model")
        
        if isinstance(X_slave, pd.DataFrame):
            X_values = X_slave.values
            is_dataframe = True
            df_info = (X_slave.columns, X_slave.index)
        else:
            X_values = X_slave.copy()
            is_dataframe = False
        
        # Apply correction: X_standardized = slope * X + bias
        X_standardized = X_values * self.slope_ + self.bias_
        
        print(f"✅ SBC correction completed: {X_values.shape[0]} spectra standardized")
        
        if is_dataframe:
            return pd.DataFrame(X_standardized, columns=df_info[0], index=df_info[1])
        return X_standardized
    
    def fit_transform(self, X_master: np.ndarray, X_slave: np.ndarray) -> np.ndarray:
        """Build model and apply correction"""
        self.fit(X_master, X_slave)
        return self.transform(X_slave)


# Convenience functions
def pds_transfer(X_master: np.ndarray, 
                 X_slave: np.ndarray,
                 window_size: int = 5) -> Tuple[np.ndarray, PDS]:
    """
    PDS model transfer convenience function
    
    Parameters:
    -----------
    X_master : ndarray
        Master instrument spectra
    X_slave : ndarray
        Slave instrument spectra
    window_size : int
        Window size
        
    Returns:
    --------
    X_standardized : ndarray
        Standardized spectra
    pds_model : PDS
        Trained PDS model
    """
    pds = PDS(window_size=window_size)
    X_standardized = pds.fit_transform(X_master, X_slave)
    return X_standardized, pds


def sbc_transfer(X_master: np.ndarray, 
                 X_slave: np.ndarray) -> Tuple[np.ndarray, SBC]:
    """
    SBC model transfer convenience function
    
    Parameters:
    -----------
    X_master : ndarray
        Master instrument spectra
    X_slave : ndarray
        Slave instrument spectra
        
    Returns:
    --------
    X_standardized : ndarray
        Standardized spectra
    sbc_model : SBC
        Trained SBC model
    """
    sbc = SBC()
    X_standardized = sbc.fit_transform(X_master, X_slave)
    return X_standardized, sbc

