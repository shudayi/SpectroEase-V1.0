# app/algorithms/raman_specific.py
"""
Raman spectrum-specific algorithms
Contains algorithms essential for Raman spectral analysis such as fluorescence background removal and wavenumber calibration
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline


class ModPolyBaseline:
    """
    Modified Polynomial (ModPoly) Baseline Correction
    
    An improved polynomial baseline correction algorithm designed for Raman spectrum fluorescence background removal.
    Compared to standard polynomial fitting, ModPoly avoids overfitting peaks through iterative weighting.
    
    Advantages:
    - Specifically handles Raman fluorescence background
    - Preserves true peak shape
    - Automatically adjusts fitting weights
    - Suitable for strong fluorescence interference
    
    References:
    Lieber, C. A., & Mahadevan-Jansen, A. (2003).
    Automated method for subtraction of fluorescence from biological Raman spectra.
    Applied Spectroscopy, 57(11), 1363-1367.
    """
    
    def __init__(self, 
                 polynomial_order: int = 5,
                 max_iterations: int = 100,
                 tolerance: float = 0.001,
                 gradient: float = 0.001):
        """
        Initialize ModPoly algorithm
        
        Parameters:
        -----------
        polynomial_order : int
            Polynomial order, default 5 (Raman typically uses 5-7 order)
        max_iterations : int
            Maximum number of iterations, default 100
        tolerance : float
            Convergence threshold, default 0.001
        gradient : float
            Gradient threshold for distinguishing peaks and background, default 0.001
        """
        self.polynomial_order = polynomial_order
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.gradient = gradient
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply ModPoly baseline correction
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Original Raman spectra
            
        Returns:
        --------
        X_corrected : ndarray
            Corrected spectra
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X.copy()
        
        n_samples, n_points = X_values.shape
        X_corrected = np.zeros_like(X_values)
        
        print(f"🔧 ModPoly baseline correction: Processing {n_samples} spectra")
        
        for i in range(n_samples):
            spectrum = X_values[i, :]
            baseline = self._fit_baseline(spectrum)
            X_corrected[i, :] = spectrum - baseline
            
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  Processed {i+1}/{n_samples} spectra")
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_corrected, columns=X.columns, index=X.index)
        return X_corrected
    
    def _fit_baseline(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Fit ModPoly baseline for a single spectrum
        
        Parameters:
        -----------
        spectrum : ndarray
            Single spectrum data
            
        Returns:
        --------
        baseline : ndarray
            Fitted baseline
        """
        n_points = len(spectrum)
        x = np.arange(n_points)
        
        # Initialize weights (all points have equal weight)
        weights = np.ones(n_points)
        
        baseline_prev = np.zeros(n_points)
        
        for iteration in range(self.max_iterations):
            # Weighted polynomial fitting
            coeffs = np.polyfit(x, spectrum, self.polynomial_order, w=weights)
            baseline = np.polyval(coeffs, x)
            
            # Check convergence
            if np.linalg.norm(baseline - baseline_prev) < self.tolerance:
                break
            
            baseline_prev = baseline.copy()
            
            # Update weights
            # If data point is above baseline (peak), reduce weight
            # If data point is below or near baseline (background), keep high weight
            deviation = spectrum - baseline
            
            # Modified weight strategy
            weights = np.where(deviation > self.gradient, 0.0, 1.0)
            
            # Smooth weight transition
            weights = np.where((deviation > 0) & (deviation <= self.gradient), 
                             np.exp(-deviation / self.gradient), weights)
        
        return baseline


class RamanFluorescenceRemoval:
    """
    Comprehensive Raman fluorescence background removal method
    
    Integrates multiple fluorescence removal algorithms:
    - ModPoly: Modified polynomial
    - VRA: Variable Ratio Algorithm
    - AFBS: Adaptive Fluorescence Background Subtraction
    
    Fluorescence is the largest interference source in Raman spectra and must be effectively removed for accurate analysis.
    """
    
    def __init__(self, method: str = 'modpoly', **params):
        """
        Initialize fluorescence removal algorithm
        
        Parameters:
        -----------
        method : str
            Algorithm selection: 'modpoly', 'vra', 'afbs'
        **params : dict
            Algorithm-specific parameters
        """
        self.method = method.lower()
        self.params = params
        
        if self.method == 'modpoly':
            self.baseline_fitter = ModPolyBaseline(**params)
        elif self.method == 'vra':
            self.polynomial_order = params.get('polynomial_order', 4)
        elif self.method == 'afbs':
            self.lam = params.get('lambda', 1e5)
            self.p = params.get('p', 0.01)
        else:
            raise ValueError(f"Unknown fluorescence removal method: {method}")
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fluorescence removal"""
        if self.method == 'modpoly':
            return self.baseline_fitter.fit_transform(X)
        elif self.method == 'vra':
            return self._vra_fluorescence_removal(X)
        elif self.method == 'afbs':
            return self._afbs_fluorescence_removal(X)
    
    def _vra_fluorescence_removal(self, X: np.ndarray) -> np.ndarray:
        """
        Variable Ratio Algorithm (VRA) fluorescence removal
        
        Iteratively adjusts baseline fitting through variable ratio
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X.copy()
        
        n_samples, n_points = X_values.shape
        X_corrected = np.zeros_like(X_values)
        
        print(f"🔧 VRA fluorescence removal: Processing {n_samples} spectra")
        
        for i in range(n_samples):
            spectrum = X_values[i, :]
            
            # VRA algorithm: Use minimum points to fit baseline
            x = np.arange(n_points)
            
            # Find minimum points in segments
            n_segments = 10
            segment_size = n_points // n_segments
            min_points_x = []
            min_points_y = []
            
            for seg in range(n_segments):
                start = seg * segment_size
                end = start + segment_size if seg < n_segments - 1 else n_points
                segment = spectrum[start:end]
                min_idx = np.argmin(segment) + start
                min_points_x.append(min_idx)
                min_points_y.append(spectrum[min_idx])
            
            # Polynomial fitting of minimum points
            if len(min_points_x) > self.polynomial_order:
                coeffs = np.polyfit(min_points_x, min_points_y, self.polynomial_order)
                baseline = np.polyval(coeffs, x)
                X_corrected[i, :] = spectrum - baseline
            else:
                X_corrected[i, :] = spectrum
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_corrected, columns=X.columns, index=X.index)
        return X_corrected
    
    def _afbs_fluorescence_removal(self, X: np.ndarray) -> np.ndarray:
        """
        Adaptive Fluorescence Background Subtraction (AFBS)
        
        Uses WhittakerSmooth to adaptively fit fluorescence background
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X.copy()
        
        n_samples, n_points = X_values.shape
        X_corrected = np.zeros_like(X_values)
        
        print(f"🔧 AFBS fluorescence removal: Processing {n_samples} spectra")
        
        for i in range(n_samples):
            spectrum = X_values[i, :]
            baseline = self._whittaker_smooth(spectrum, self.lam, self.p)
            X_corrected[i, :] = spectrum - baseline
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_corrected, columns=X.columns, index=X.index)
        return X_corrected
    
    @staticmethod
    def _whittaker_smooth(y: np.ndarray, lam: float, p: float) -> np.ndarray:
        """
        Whittaker smoothing algorithm for baseline estimation
        
        Parameters:
        -----------
        y : ndarray
            Input signal
        lam : float
            Smoothing parameter, larger value means smoother
        p : float
            Asymmetry parameter, 0.001-0.1
        """
        n = len(y)
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
        w = np.ones(n)
        
        for _ in range(10):  # Iterate 10 times
            W = sparse.diags(w, 0, shape=(n, n))
            Z = W + lam * D.T @ D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z


class RamanShiftCalibration:
    """
    Raman wavenumber calibration algorithm
    
    Uses standard materials (e.g., silicon 520 cm⁻¹ peak) to calibrate the wavenumber axis of Raman spectra.
    Wavenumber calibration is crucial for quantitative analysis and data comparison between different instruments.
    
    Standard materials:
    - Silicon (Si): 520.7 cm⁻¹
    - Cyclohexane: 801.3, 1028.3, 1157.6, 1266.4, 1444.4 cm⁻¹
    - Polystyrene: 621, 1001, 1031, 1155, 1583, 1602 cm⁻¹
    
    References:
    McCreery, R. L. (2000).
    Raman Spectroscopy for Chemical Analysis.
    John Wiley & Sons.
    """
    
    def __init__(self, 
                 reference_peaks: Optional[list] = None,
                 calibration_method: str = 'silicon'):
        """
        Initialize Raman Shift calibration
        
        Parameters:
        -----------
        reference_peaks : list of float, optional
            Reference peak positions (cm⁻¹)
        calibration_method : str
            Calibration method: 'silicon' (single peak), 'multipoint' (multiple peaks)
        """
        self.calibration_method = calibration_method
        
        # Default reference peaks
        if reference_peaks is None:
            if calibration_method == 'silicon':
                self.reference_peaks = [520.7]  # Silicon standard peak
            elif calibration_method == 'cyclohexane':
                self.reference_peaks = [801.3, 1028.3, 1266.4]
            elif calibration_method == 'polystyrene':
                self.reference_peaks = [1001, 1031, 1583, 1602]
            else:
                self.reference_peaks = [520.7]
        else:
            self.reference_peaks = reference_peaks
        
        self.shift_correction_ = None
        self.scale_correction_ = None
    
    def calibrate(self, 
                  wavenumbers: np.ndarray, 
                  standard_spectrum: np.ndarray) -> Tuple[float, float]:
        """
        Calibrate using standard material spectrum
        
        Parameters:
        -----------
        wavenumbers : ndarray
            Current wavenumber axis
        standard_spectrum : ndarray
            Standard material spectrum
            
        Returns:
        --------
        shift : float
            Wavenumber shift (cm⁻¹)
        scale : float
            Wavenumber scale factor
        """
        print(f"🔧 Raman Shift calibration: Using {self.calibration_method} method")
        
        # Detect peak positions
        detected_peaks = []
        
        for ref_peak in self.reference_peaks:
            # Search near reference peak
            search_window = 50  # ±50 cm⁻¹
            mask = (wavenumbers >= ref_peak - search_window) & \
                   (wavenumbers <= ref_peak + search_window)
            
            if not np.any(mask):
                continue
            
            window_wn = wavenumbers[mask]
            window_intensity = standard_spectrum[mask]
            
            # Find peaks
            peaks, properties = find_peaks(window_intensity, prominence=np.std(window_intensity))
            
            if len(peaks) > 0:
                # Select strongest peak
                max_peak_idx = peaks[np.argmax(window_intensity[peaks])]
                detected_peak = window_wn[max_peak_idx]
                detected_peaks.append((ref_peak, detected_peak))
                print(f"  Detected peak: {detected_peak:.2f} cm⁻¹ (reference: {ref_peak} cm⁻¹)")
        
        if len(detected_peaks) == 0:
            print("  ⚠ No reference peaks detected, cannot calibrate")
            return 0.0, 1.0
        
        # Calculate correction parameters
        if len(detected_peaks) == 1:
            # Single-point calibration: only calculate shift
            ref, det = detected_peaks[0]
            self.shift_correction_ = ref - det
            self.scale_correction_ = 1.0
            print(f"✅ Single-point calibration completed: Shift = {self.shift_correction_:.2f} cm⁻¹")
        else:
            # Multi-point calibration: linear fitting
            refs = np.array([p[0] for p in detected_peaks])
            dets = np.array([p[1] for p in detected_peaks])
            
            # Linear regression: ref = scale * det + shift
            coeffs = np.polyfit(dets, refs, 1)
            self.scale_correction_ = coeffs[0]
            self.shift_correction_ = coeffs[1]
            
            print(f"✅ Multi-point calibration completed: Scale = {self.scale_correction_:.6f}, Shift = {self.shift_correction_:.2f} cm⁻¹")
        
        return self.shift_correction_, self.scale_correction_
    
    def apply_calibration(self, wavenumbers: np.ndarray) -> np.ndarray:
        """
        Apply calibration to wavenumber axis
        
        Parameters:
        -----------
        wavenumbers : ndarray
            Original wavenumbers
            
        Returns:
        --------
        calibrated_wavenumbers : ndarray
            Calibrated wavenumbers
        """
        if self.shift_correction_ is None:
            raise ValueError("Must call calibrate method first")
        
        calibrated = wavenumbers * self.scale_correction_ + self.shift_correction_
        return calibrated
    
    def transform_spectrum(self, 
                          wavenumbers: np.ndarray, 
                          spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate wavenumber axis of spectrum
        
        Parameters:
        -----------
        wavenumbers : ndarray
            Original wavenumbers
        spectrum : ndarray
            Spectral intensity
            
        Returns:
        --------
        calibrated_wavenumbers : ndarray
            Calibrated wavenumbers
        spectrum : ndarray
            Spectral intensity (unchanged)
        """
        calibrated_wn = self.apply_calibration(wavenumbers)
        return calibrated_wn, spectrum

