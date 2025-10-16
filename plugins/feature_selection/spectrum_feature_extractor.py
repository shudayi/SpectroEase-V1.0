import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List

class SpectrumFeatureExtractor:
    """Spectrum feature extraction class"""
    
    @staticmethod
    def pca(spectra: np.ndarray, 
            n_components: Optional[int] = None,
            explained_variance_ratio_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Principal Component Analysis feature extraction
        
        Args:
            spectra: Spectral data array
            n_components: Number of components, if None auto-select based on variance ratio
            explained_variance_ratio_threshold: Variance ratio threshold
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (principal component scores, loadings)
        """
  
        scaler = StandardScaler()
        spectra_scaled = scaler.fit_transform(spectra)
        
        # PCAdimensionality reduction
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(spectra_scaled)
        
  
        if n_components is None:
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance_ratio >= explained_variance_ratio_threshold) + 1
            scores = scores[:, :n_components]
            
        return scores, pca.components_[:n_components]
    
    @staticmethod
    def plsr(spectra: np.ndarray,
             y: np.ndarray,
             n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partial least squares regression feature extraction
        
        Args:
            spectra: spectral data array
            y: target variable
            n_components: number of latent variables
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (latent variable scores, latent variable loadings)
        """
        # **CRITICAL FIX: Enhanced handling for both numeric and string labels**
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        

        
        # Check if y contains string labels (like Verde, ClassA, etc.)
        has_string_labels = False
        try:
            # Check if any labels are non-numeric strings
            for val in y[:10]:  # Check first 10 samples
                str_val = str(val).strip()
                if str_val and not str_val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                    has_string_labels = True
                    break
        except:
            has_string_labels = True
            
        if has_string_labels:
            label_encoder = LabelEncoder()
            y_numeric = label_encoder.fit_transform(y.astype(str))
        else:
            if isinstance(y, pd.Series):
                y_numeric = pd.to_numeric(y, errors='coerce')
                if pd.isna(y_numeric).any():
                    raise ValueError("PLSR: Some labels cannot be converted to numeric")
            else:
                try:
                    y_numeric = np.array([float(val) for val in y])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"PLSR: Cannot convert labels to numeric: {e}")

        # Standardize features and targets
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        spectra_scaled = scaler_x.fit_transform(spectra)
        y_scaled = scaler_y.fit_transform(y_numeric.reshape(-1, 1))
        
        # PLSR dimensionality reduction
        plsr = PLSRegression(n_components=n_components)
        plsr.fit(spectra_scaled, y_scaled)
        

        return plsr.x_scores_, plsr.x_loadings_
    
    @staticmethod
    def peak_detection(spectra: np.ndarray,
                      wavelengths: np.ndarray,
                      prominence: float = 0.1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        peak detection
        
        Args:
            spectra: spectral data array
            wavelengths: wavelength array
            prominence: peak prominence threshold
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (peak position list, peak intensity list)
        """
        from scipy.signal import find_peaks
        
        if len(spectra.shape) == 1:
            spectra = spectra.reshape(1, -1)
            
        all_peaks = []
        all_intensities = []
        
        for i in range(spectra.shape[0]):
  
            peaks, properties = find_peaks(spectra[i], prominence=prominence)
            all_peaks.append(wavelengths[peaks])
            all_intensities.append(spectra[i][peaks])
            
  
        return all_peaks, all_intensities
    
    @staticmethod
    def wavelet_transform(spectra: np.ndarray,
                         wavelet: str = 'db4',
                         level: int = 4) -> np.ndarray:
        """
        wavelet transform feature extraction
        
        Args:
            spectra: spectral data array
            wavelet: wavelet basis function
            level: decomposition levels
            
        Returns:
            np.ndarray: wavelet coefficients
        """
        import pywt
        
        if len(spectra.shape) == 1:
            spectra = spectra.reshape(1, -1)
            
        wavelet_coeffs = []
        
        for i in range(spectra.shape[0]):
  
            coeffs = pywt.wavedec(spectra[i], wavelet, level=level)
  
            flat_coeffs = np.concatenate([c for c in coeffs])
            wavelet_coeffs.append(flat_coeffs)
            
        return np.array(wavelet_coeffs) 