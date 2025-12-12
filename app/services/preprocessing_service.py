
# app/services/preprocessing_service.py

from app.utils.exceptions import PreprocessingError
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Any, Optional
# from sklearn.preprocessing import StandardScaler, MinMaxScaler # No longer used directly
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm

# Import new specialized algorithms
from app.algorithms.raman_specific import ModPolyBaseline, RamanFluorescenceRemoval, RamanShiftCalibration
from app.algorithms.mir_specific import AtmosphericCompensation
from app.algorithms.nir_specific import WaterPeakRemoval
from app.algorithms.model_transfer import PDS, SBC
from app.algorithms.baseline_correction import SNIPBaseline

# Import data flow tracker
from app.utils.data_flow_tracker import data_flow_tracker

class PreprocessingService:
    def __init__(self, plugins: Optional[Dict[str, PreprocessingAlgorithm]] = None):
        self.plugins = plugins or {}
        self.tracker = data_flow_tracker  # Use global tracker
        self.enable_tracking = True  # Whether to enable tracking (configurable)

    def apply_method(self, data: pd.DataFrame, method: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a preprocessing method with full data flow tracking
        
        🔍 V1.2.1: Wrapper method, add data flow tracking
        """
        # Execute preprocessing and track
        result = self._apply_method_internal(data, method, params)
        
        # 🔍 Data flow tracking - record output snapshot and verify
        if self.enable_tracking:
            output_stage = f"after_{method.replace(' ', '_').lower()}"
            input_stage = f"before_{method.replace(' ', '_').lower()}"
            
            self.tracker.snapshot(result, output_stage, metadata={'method': method, 'params': params})
            
            # Automatically compare and verify
            comparison = self.tracker.compare(input_stage, output_stage, algorithm_name=method)
            self.tracker.print_comparison(comparison, verbose=False)
            print("═" * 60)
        
        return result
    
    def _apply_method_internal(self, data: pd.DataFrame, method: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply a preprocessing method to the spectral data (internal)"""
        print("🔧 Executing preprocessing algorithm: {0}".format(method))
        
        # 🔍 V1.2.1: Data flow tracking - record input snapshot
        input_stage = f"before_{method.replace(' ', '_').lower()}"
        
        if self.enable_tracking:
            self.tracker.snapshot(data, input_stage, metadata={'method': method, 'params': params})
            print(f"   Input snapshot: {input_stage}")
            print(f"   Input shape: {data.shape}")
            print(f"   Input range: [{data.values.min():.2f}, {data.values.max():.2f}]")
        
        # Execute preprocessing (maintain original logic)
        # V1.4.0: Check both service plugins and view plugins (for dynamically added algorithms)
        algorithm = None
        if method in self.plugins:
            algorithm = self.plugins[method]
            print(f"   Found algorithm in service plugins: {method}")
        elif hasattr(self, 'view') and self.view and hasattr(self.view, 'plugins') and method in self.view.plugins:
            algorithm = self.view.plugins[method]
            print(f"   Found algorithm in view plugins: {method}")
        else:
            print(f"   ⚠️  Algorithm not found: {method}")
            print(f"   Available in service plugins: {list(self.plugins.keys())[:5]}...")
            if hasattr(self, 'view') and self.view and hasattr(self.view, 'plugins'):
                print(f"   Available in view plugins: {list(self.view.plugins.keys())}")
        
        if algorithm is not None:
            # Use plugin algorithm
            result = algorithm.apply(data, params)
            
            # 🔍 V1.4.1: Verify that preprocessing actually changed the data
            if self.enable_tracking:
                data_diff = np.abs(result.values - data.values)
                mean_diff = np.mean(data_diff)
                max_diff = np.max(data_diff)
                pct_changed = np.sum(data_diff > 1e-10) / data_diff.size * 100
                
                print(f"   Preprocessing verification:")
                print(f"      Mean difference: {mean_diff:.6f}")
                print(f"      Max difference: {max_diff:.6f}")
                print(f"      Change percentage: {pct_changed:.2f}%")
                
                if mean_diff < 1e-6:
                    print(f"      ⚠️  WARNING: Preprocessing algorithm '{method}' returned data with minimal changes!")
                    print(f"      Possible reasons:")
                    print(f"      1. Algorithm implementation may be incorrect")
                    print(f"      2. Parameters may be inappropriate")
                    print(f"      3. Algorithm may not be suitable for this data")
            
            return result
        else:
            if method == "Baseline Correction":
                return self.baseline_correction(data, **params)
            elif method == "Savitzky-Golay Filter":
                return self.savgol_filter(data, **params)
            elif method == "Moving Average":
                return self.moving_average(data, **params)
            elif method == "Median Filter":
                return self.median_filter(data, **params)
            elif method == "Gaussian Smooth":
                return self.gaussian_smooth(data, **params)
            elif method == "Wavelet Smooth":
                return self.wavelet_smooth(data, **params)
            elif method == "Standard Normal Variate (SNV)":
                return self.standard_normal_variate(data, **params)
            elif method == "Multiplicative Scatter Correction (MSC)":
                return self.msc(data, **params)
            elif method == "Extended Multiplicative Scatter Correction (EMSC)":
                return self.emsc(data, **params)
            elif method == "Robust Normal Variate (RNV)":
                return self.rnv(data, **params)
            elif method == "Orthogonal Signal Correction (OSC)":
                return self.osc(data, **params)
            elif method == "Normalize":
                return self.normalize(data, **params)
            elif method == "Standard Scale":
                return self.standard_scale(data)
            elif method == "Min-Max Scale":
                return self.min_max_scale(data)
            elif method == "Vector Normalization":
                return self.vector_normalization(data, **params)
            elif method == "Area Normalization":
                return self.area_normalization(data, **params)
            elif method == "Maximum Normalization":
                return self.maximum_normalization(data, **params)
            elif method == "First Derivative":
                return self.first_derivative(data)
            elif method == "Second Derivative":
                return self.second_derivative(data)
            elif method == "Savitzky-Golay Derivative":
                return self.savgol_derivative(data, **params)
            elif method == "Finite Difference":
                return self.finite_difference(data, **params)
            elif method == "Gap-Segment Derivative":
                return self.gap_segment_derivative(data, **params)
            elif method == "Despiking":
                return self.despiking(data, **params)
            elif method == "Denoising":
                return self.denoising(data, **params)
            elif method == "Peak Alignment":
                return self.peak_alignment(data, **params)
            elif method == "Outlier Detection":
                return self.outlier_detection(data, **params)
            # New: Raman-specific algorithms
            elif method == "Raman Fluorescence Removal":
                return self.raman_fluorescence_removal(data, **params)
            elif method == "Raman Shift Calibration":
                # Requires wavenumber information and standard spectrum
                if 'wavenumbers' not in params or 'standard_spectrum' not in params:
                    print("⚠️  Raman Shift calibration requires wavenumber information and standard spectrum, skipping")
                    return data
                return self.raman_shift_calibration(data, params['wavenumbers'], params['standard_spectrum'], **params)
            # New: MIR-specific algorithms
            elif method == "Atmospheric Compensation":
                # 🔧 P0 fix: Check and extract wavenumber information
                if 'wavelengths' not in params or params['wavelengths'] is None:
                    print("⚠️  MIR atmospheric compensation requires wavenumber information, skipping this step")
                    return data
                wavenumbers = params.get('wavelengths')  # Get wavenumber information (don't delete key)
                return self.atmospheric_compensation(data, wavenumbers, **params)
            # New: NIR-specific algorithms
            elif method == "Water Peak Removal":
                # 🔧 P0 fix: Check and extract wavelength information
                if 'wavelengths' not in params or params['wavelengths'] is None:
                    print("⚠️  NIR water peak removal requires wavelength information, skipping this step")
                    return data
                wavelengths = params.get('wavelengths')  # Get wavelength information (don't delete key)
                return self.water_peak_removal(data, wavelengths, **params)
            # New: Model transfer algorithms
            elif method == "Model Transfer":
                # Requires master and slave instrument data
                if 'data_master' not in params or 'data_slave' not in params:
                    print("⚠️  Model transfer requires master and slave instrument data, skipping")
                    return data
                return self.model_transfer(params['data_master'], params['data_slave'], method=params.get('method', 'pds'), **params)
            else:
                error_msg = f"Unsupported preprocessing method: {method}"
                print(f"❌ {error_msg}")
    
                raise PreprocessingError(error_msg)

    def baseline_correction(self, data, method='Polynomial', polynomial_order=3, **kwargs):
        """Apply baseline correction using various methods"""
        try:
            print(f"📊 Baseline correction: {method}")
            
            # Handle new algorithms
            if 'ModPoly' in method:
                print("  Using ModPoly baseline correction (Raman-specific)")
                modpoly = ModPolyBaseline(
                    polynomial_order=polynomial_order,
                    max_iterations=kwargs.get('max_iterations', 100),
                    tolerance=kwargs.get('tolerance', 0.001),
                    gradient=kwargs.get('gradient', 0.001)
                )
                return modpoly.fit_transform(data)
            
            elif 'SNIP' in method:
                print("  Using SNIP baseline correction (multi-peak spectra)")
                snip = SNIPBaseline(
                    max_half_width=polynomial_order if polynomial_order > 10 else 40,
                    decreasing=True
                )
                return snip.fit_transform(data)
            
            # Handle parameter extraction for traditional methods
            if isinstance(polynomial_order, dict):
                polynomial_order = polynomial_order.get('polynomial_order', 3)
            
            # Ensure polynomial_order is an integer
            try:
                polynomial_order = int(polynomial_order)
            except (ValueError, TypeError):
                polynomial_order = 3
            
            corrected_data = data.copy()
            data_array = corrected_data.values
            
            if method.lower() == 'polynomial':
                # Polynomial baseline correction
                for i in range(data_array.shape[0]):
                    spectrum = data_array[i, :]
                    x = np.arange(len(spectrum))
                      
                    coeffs = np.polyfit(x, spectrum, polynomial_order)
                    baseline = np.polyval(coeffs, x)
                    
                    data_array[i, :] = spectrum - baseline
                    
            elif method.lower() == 'als':
                # ALS (Asymmetric Least Squares) baseline correction - simplified implementation
                for i in range(data_array.shape[0]):
                    spectrum = data_array[i, :]
                    # Simplified ALS: use polynomial as approximation
                    x = np.arange(len(spectrum))
                    coeffs = np.polyfit(x, spectrum, max(2, polynomial_order-1))
                    baseline = np.polyval(coeffs, x)
                    data_array[i, :] = spectrum - baseline
                    
            elif method.lower() == 'airpls':
                # airPLS baseline correction - simplified implementation  
                for i in range(data_array.shape[0]):
                    spectrum = data_array[i, :]
                    # Simplified airPLS: use polynomial as approximation
                    x = np.arange(len(spectrum))
                    coeffs = np.polyfit(x, spectrum, max(1, polynomial_order-2))
                    baseline = np.polyval(coeffs, x)
                    data_array[i, :] = spectrum - baseline
            else:
                # Default to polynomial
                for i in range(data_array.shape[0]):
                    spectrum = data_array[i, :]
                    x = np.arange(len(spectrum))
                    coeffs = np.polyfit(x, spectrum, polynomial_order)
                    baseline = np.polyval(coeffs, x)
                    data_array[i, :] = spectrum - baseline
            
            corrected_data.iloc[:, :] = data_array

            return corrected_data
            
        except Exception as e:
            raise PreprocessingError(f"Baseline Correction failed: {e}")

    def savitzky_golay_filter(self, data, window_length=15, polyorder=2, **kwargs):
        return self.savgol_filter(data, window_length, polyorder, **kwargs)

    def savgol_filter(self, data, window_length=15, polyorder=2, **kwargs):
        """Apply Savitzky-Golay smoothing filter"""
        try:

            
              
            if window_length % 2 == 0:
                window_length += 1
    
            
              
            window_length = min(window_length, data.shape[1])
            if window_length <= polyorder:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
    
            
            filtered_data = savgol_filter(data.values, window_length=window_length, 
                                        polyorder=polyorder, axis=1)
            result = pd.DataFrame(filtered_data, index=data.index, columns=data.columns)

            return result
            
        except Exception as e:
            raise PreprocessingError(f"Savitzky-Golay Filter failed: {e}")

    def moving_average(self, data, window_size=5, **kwargs):
        """Apply moving average smoothing"""
        try:

            result = data.rolling(window=window_size, min_periods=1, axis=1).mean()

            return result
        except Exception as e:
            raise PreprocessingError(f"Moving Average failed: {e}")

    def median_filter(self, data, kernel_size=3, **kwargs):
        """Apply median filter for noise reduction"""
        try:

            result = data.rolling(window=kernel_size, min_periods=1, center=True, axis=1).median()

            return result
        except Exception as e:
            raise PreprocessingError(f"Median Filter failed: {e}")

    def gaussian_smooth(self, data, sigma=1.0, **kwargs):
        """Apply Gaussian smoothing filter"""
        try:

            from scipy.ndimage import gaussian_filter
            smoothed_data = gaussian_filter(data.values, sigma=sigma, axes=1)
            result = pd.DataFrame(smoothed_data, index=data.index, columns=data.columns)

            return result
        except Exception as e:
            raise PreprocessingError(f"Gaussian Smooth failed: {e}")

    def standard_normal_variate_snv(self, data, center=True, scale=True, min_std=1e-6, **kwargs):
        return self.standard_normal_variate(data, center, scale, min_std, **kwargs)

    def standard_normal_variate(self, data, center=True, scale=True, min_std=1e-6, **kwargs):
        """Apply Standard Normal Variate (SNV) normalization"""
        try:

            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                  
                if center:
                    mean_val = np.mean(spectrum)
                    spectrum = spectrum - mean_val
                
                  
                if scale:
                    std_val = np.std(spectrum)
                    if std_val < min_std:
                        std_val = min_std
                    spectrum = spectrum / std_val
                
                data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)

            return result
            
        except Exception as e:
            raise PreprocessingError(f"Standard Normal Variate (SNV) failed: {e}")

    def multiplicative_scatter_correction_msc(self, data, **kwargs):
        return self.msc(data, **kwargs)

    def msc(self, data, **kwargs):
        """Apply Multiplicative Scatter Correction (MSC)"""
        try:

            mean_spectrum = data.mean(axis=0)
            msc_data = data.copy()
            
            for i in range(data.shape[0]):
                spectrum = data.iloc[i, :]
                  
                coeffs = np.polyfit(mean_spectrum, spectrum, 1)
                  
                msc_data.iloc[i, :] = (spectrum - coeffs[1]) / coeffs[0]
            

            return msc_data
        except Exception as e:
            raise PreprocessingError(f"Multiplicative Scatter Correction (MSC) failed: {e}")

    def normalize(self, data, norm='l2', **kwargs):
        """Apply L2 or L1 normalization to spectral data"""
        try:

            from sklearn.preprocessing import normalize
            normalized_data = normalize(data.values, norm=norm, axis=1)
            result = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

            return result
        except Exception as e:
            raise PreprocessingError(f"Normalize failed: {e}")

    def z_score_standardization(self, data, **kwargs):
        return self.standard_scale(data, **kwargs)

    def standard_scale(self, data, **kwargs):
        """Apply row-wise standard scaling (z-score normalization) for each spectrum."""
        try:
            print("📊 Applying row-wise Z-Score normalization (Standard Scale)")
            print("   ⚠️  Note: Standard Scale will compress spectral data range to approximately -3 to +3")
            print("   This will make spectra appear flat in visualization, but does not affect modeling performance")
            
            data_values = data.values
            
            # Calculate mean and std for each row (spectrum)
            mean_vals = np.mean(data_values, axis=1, keepdims=True)
            std_vals = np.std(data_values, axis=1, keepdims=True)
            
            # Avoid division by zero for flat spectra
            std_vals[std_vals == 0] = 1
            
            scaled_data = (data_values - mean_vals) / std_vals
            result = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
            
            # Output data range information
            print(f"   Data range before processing: [{np.min(data_values):.2f}, {np.max(data_values):.2f}]")
            print(f"   Data range after processing: [{np.min(scaled_data):.2f}, {np.max(scaled_data):.2f}]")

            return result
        except Exception as e:
            raise PreprocessingError(f"Standard Scale failed: {e}")

    def min_max_normalization(self, data, **kwargs):
        return self.min_max_scale(data, **kwargs)

    def min_max_scale(self, data, **kwargs):
        """Apply row-wise min-max scaling (0-1 normalization) for each spectrum."""
        try:
            print("📊 Applying row-wise Min-Max normalization")
            data_values = data.values

            # Calculate min and max for each row (spectrum)
            min_vals = np.min(data_values, axis=1, keepdims=True)
            max_vals = np.max(data_values, axis=1, keepdims=True)

            # Avoid division by zero for flat spectra
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1

            scaled_data = (data_values - min_vals) / range_vals
            result = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

            return result
        except Exception as e:
            raise PreprocessingError(f"Min-Max Scale failed: {e}")

    def first_derivative(self, data, **kwargs):
        """Calculate first derivative of spectral data"""
        try:

            derivative = data.diff(axis=1).fillna(0)

            return derivative
        except Exception as e:
            raise PreprocessingError(f"First Derivative failed: {e}")

    def second_derivative(self, data, **kwargs):
        """Calculate second derivative of spectral data"""
        try:

            derivative = data.diff(axis=1).diff(axis=1).fillna(0)

            return derivative
        except Exception as e:
            raise PreprocessingError(f"Second Derivative failed: {e}")
    
    def despiking(self, data, method='mad', window=11, threshold=5, **kwargs):
        """Apply despiking (spike removal) to spectral data - Raman spike removal processing
        
        Args:
            data: Input spectral data (DataFrame)
            method: Despiking method ('mad' or 'localz')
            window: Window size for spike detection (3-21, must be odd)
            threshold: Threshold for spike detection (1-10)
            
        Returns:
            DataFrame: Processed spectral data with spike statistics
        """
        try:
            print("🔧 Removing spikes from Raman spectra")
            
            # Validate parameters with expanded ranges
            window = max(3, min(21, int(window)))  # Expand to 3-21
            if window % 2 == 0:
                window += 1  # Ensure odd window size
            threshold = max(1.0, min(10.0, float(threshold)))  # Expand to 1-10
            
            logging.info(f"Despiking parameters: method={method}, window={window}, threshold={threshold}")
            
            data_array = data.values.copy()
            spike_stats = []  # Store spike statistics for each spectrum
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                spike_count = 0
                
                if method.lower() in ['mad', 'MAD'.lower()]:
                    # Method 1: Median Absolute Deviation (MAD) based
                    for j in range(window//2, len(spectrum) - window//2):
                        # Extract window around current point
                        window_data = spectrum[j-window//2:j+window//2+1]
                        
                        # Calculate median and MAD
                        median_val = np.median(window_data)
                        mad = np.median(np.abs(window_data - median_val))
                        
                        # Detect spike if MAD is not zero
                        if mad > 0:
                            deviation = abs(spectrum[j] - median_val) / mad
                            if deviation > threshold:
                                # Replace spike with median
                                data_array[i, j] = median_val
                                spike_count += 1
                                
                elif method.lower() in ['localz', 'local z-score']:
                    # Method 2: Local Z-score based
                    for j in range(window//2, len(spectrum) - window//2):
                        # Extract window around current point
                        window_data = spectrum[j-window//2:j+window//2+1]
                        
                        # Calculate local mean and std
                        local_mean = np.mean(window_data)
                        local_std = np.std(window_data)
                        
                        # Detect spike if std is not zero
                        if local_std > 0:
                            z_score = abs(spectrum[j] - local_mean) / local_std
                            if z_score > threshold:
                                # Replace spike with local mean
                                data_array[i, j] = local_mean
                                spike_count += 1
                
                # Store statistics
                spike_ratio = spike_count / len(spectrum)
                spike_stats.append({
                    'spectrum_index': i,
                    'spike_count': spike_count,
                    'spike_ratio': spike_ratio
                })
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            
            # Add spike statistics as metadata
            result._spike_stats = spike_stats
            result._despiking_params = {
                'method': method,
                'window': window,
                'threshold': threshold
            }
            
            # Print summary statistics
            total_spikes = sum(stat['spike_count'] for stat in spike_stats)
            avg_spike_ratio = np.mean([stat['spike_ratio'] for stat in spike_stats])
            print(f"✅ Despiking completed: {total_spikes} spikes removed ({avg_spike_ratio:.2%} of data)")
            
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Despiking failed: {e}")

    def outlier_detection(self, data, method='z-score', threshold=3.0, **kwargs):
        """Detect and remove outliers using various methods"""
        try:
            print(f"📊 Applying {method.upper()} outlier detection")
            
            if method.lower() == 'iqr':
                return self.iqr_outlier_detection(data, **kwargs)
            elif method.lower() == 'z-score':
                return self.zscore_outlier_detection(data, threshold, **kwargs)
            elif method.lower() == 'iso' or method.lower() == 'isolation':
                return self.isolation_forest_outlier_detection(data, **kwargs)
            elif method.lower() == 'lof':
                return self.lof_outlier_detection(data, **kwargs)
            else:
                # Default to Z-score
                return self.zscore_outlier_detection(data, threshold, **kwargs)
                
        except Exception as e:
            raise PreprocessingError(f"Outlier Detection failed: {e}")
    
    def zscore_outlier_detection(self, data, threshold=3.0, **kwargs):
        """Z-score based outlier detection"""
        try:
            # Calculate Z-scores for each spectrum
            data_array = data.values
            mean_vals = np.mean(data_array, axis=1, keepdims=True)
            std_vals = np.std(data_array, axis=1, keepdims=True)
            
            # Avoid division by zero
            std_vals[std_vals < 1e-8] = 1e-8
            
            z_scores = np.abs((data_array - mean_vals) / std_vals)
            
            # Mark outliers
            outlier_mask = z_scores > threshold
            
            # Replace outliers with interpolated values
            cleaned_data = data_array.copy()
            for i in range(cleaned_data.shape[0]):
                if np.any(outlier_mask[i]):
                    # Simple interpolation for outliers
                    outlier_indices = np.where(outlier_mask[i])[0]
                    for idx in outlier_indices:
                        if idx > 0 and idx < cleaned_data.shape[1] - 1:
                            cleaned_data[i, idx] = (cleaned_data[i, idx-1] + cleaned_data[i, idx+1]) / 2
                        elif idx == 0:
                            cleaned_data[i, idx] = cleaned_data[i, 1]
                        else:
                            cleaned_data[i, idx] = cleaned_data[i, -2]
            
            result = pd.DataFrame(cleaned_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Z-score outlier detection failed: {e}")
    
    def iqr_outlier_detection(self, data, **kwargs):
        """IQR-based outlier detection"""
        try:
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Calculate IQR
                q1 = np.percentile(spectrum, 25)
                q3 = np.percentile(spectrum, 75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Replace outliers
                outlier_mask = (spectrum < lower_bound) | (spectrum > upper_bound)
                spectrum[outlier_mask] = np.median(spectrum)
                
                data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"IQR outlier detection failed: {e}")
    
    def isolation_forest_outlier_detection(self, data, contamination=0.1, **kwargs):
        """Isolation Forest outlier detection"""
        try:
            from sklearn.ensemble import IsolationForest
            
            data_array = data.values.copy()
            
            # Apply Isolation Forest to each spectrum
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :].reshape(-1, 1)
                
                # Detect outliers
                outlier_labels = iso_forest.fit_predict(spectrum)
                outlier_mask = outlier_labels == -1
                
                # Replace outliers with median
                if np.any(outlier_mask):
                    median_val = np.median(data_array[i, :])
                    data_array[i, outlier_mask.flatten()] = median_val
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except ImportError:
            # Fallback to Z-score if sklearn not available
            return self.zscore_outlier_detection(data, **kwargs)
        except Exception as e:
            raise PreprocessingError(f"Isolation Forest outlier detection failed: {e}")
    
    def lof_outlier_detection(self, data, n_neighbors=20, contamination=0.1, **kwargs):
        """Local Outlier Factor detection"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            data_array = data.values.copy()
            
            # Apply LOF to each spectrum
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :].reshape(-1, 1)
                
                # Detect outliers
                outlier_labels = lof.fit_predict(spectrum)
                outlier_mask = outlier_labels == -1
                
                # Replace outliers with local median
                if np.any(outlier_mask):
                    # Use local median for replacement
                    window_size = min(5, len(spectrum) // 4)
                    for idx in np.where(outlier_mask.flatten())[0]:
                        start = max(0, idx - window_size)
                        end = min(len(spectrum), idx + window_size + 1)
                        local_median = np.median(data_array[i, start:end])
                        data_array[i, idx] = local_median
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except ImportError:
            # Fallback to IQR if sklearn not available
            return self.iqr_outlier_detection(data, **kwargs)
        except Exception as e:
            raise PreprocessingError(f"LOF outlier detection failed: {e}")
    
    # ==================== SCATTER CORRECTION METHODS ====================
    
    def emsc(self, data, polynomial_order=2, reference_spectrum=None, **kwargs):
        """Apply Extended Multiplicative Scatter Correction (EMSC)"""
        try:
            print("📊 Applying EMSC correction")
            
            data_array = data.values.copy()
            n_samples, n_wavelengths = data_array.shape
            
            # Use mean spectrum as reference if not provided
            if reference_spectrum is None:
                reference_spectrum = np.mean(data_array, axis=0)
            
            corrected_data = data_array.copy()
            
            for i in range(n_samples):
                spectrum = data_array[i, :]
                
                # Build design matrix for polynomial baseline + multiplicative term
                X = np.ones((n_wavelengths, polynomial_order + 2))
                
                # Polynomial terms for baseline
                x_norm = np.linspace(-1, 1, n_wavelengths)
                for j in range(1, polynomial_order + 1):
                    X[:, j] = x_norm ** j
                
                # Multiplicative scatter term
                X[:, -1] = reference_spectrum
                
                # Solve for coefficients
                try:
                    coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]
                    
                    # Remove polynomial baseline and apply multiplicative correction
                    baseline = X[:, :-1] @ coeffs[:-1]
                    multiplicative_coeff = coeffs[-1]
                    
                    if abs(multiplicative_coeff) > 1e-8:
                        corrected_data[i, :] = (spectrum - baseline) / multiplicative_coeff
                    else:
                        corrected_data[i, :] = spectrum - baseline
                        
                except np.linalg.LinAlgError:
                    # Fallback to simple MSC if EMSC fails
                    coeffs = np.polyfit(reference_spectrum, spectrum, 1)
                    corrected_data[i, :] = (spectrum - coeffs[1]) / coeffs[0]
            
            result = pd.DataFrame(corrected_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"EMSC failed: {e}")
    
    def rnv(self, data, **kwargs):
        """Apply Robust Normal Variate (RNV) normalization"""
        try:
            print("📊 Applying RNV normalization")
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Use median and MAD instead of mean and std for robustness
                median_val = np.median(spectrum)
                mad = np.median(np.abs(spectrum - median_val))
                
                # Apply RNV transformation
                if mad > 1e-8:
                    data_array[i, :] = (spectrum - median_val) / mad
                else:
                    data_array[i, :] = spectrum - median_val
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"RNV failed: {e}")
    
    def osc(self, data, n_components=1, **kwargs):
        """Apply Orthogonal Signal Correction (OSC)"""
        try:
            print("📊 Applying OSC correction")
            
            from sklearn.decomposition import PCA
            
            data_array = data.values.copy()
            
            # Apply PCA to find orthogonal components
            pca = PCA(n_components=min(n_components, data_array.shape[1]//4))
            
            # Fit PCA and get orthogonal components
            pca.fit(data_array.T)  # Transpose for wavelength-wise PCA
            orthogonal_components = pca.components_.T
            
            # Remove orthogonal variation
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Project onto orthogonal space and remove
                for j in range(orthogonal_components.shape[1]):
                    component = orthogonal_components[:, j]
                    projection = np.dot(spectrum, component) * component
                    spectrum = spectrum - projection
                
                data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"OSC failed: {e}")
    
    # ==================== SMOOTHING METHODS ====================
    
    def wavelet_smooth(self, data, wavelet='db4', mode='soft', **kwargs):
        """Apply wavelet smoothing"""
        try:
            print("📊 Applying wavelet smoothing")
            
            import pywt
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(spectrum, wavelet, mode='periodization')
                
                # Soft thresholding for denoising
                threshold = 0.1 * np.max(np.abs(coeffs[0]))
                coeffs_thresh = [pywt.threshold(c, threshold, mode) for c in coeffs]
                
                # Reconstruct signal
                data_array[i, :] = pywt.waverec(coeffs_thresh, wavelet, mode='periodization')
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except ImportError:
            # Fallback to Gaussian smoothing if PyWavelets not available
            print("⚠️ PyWavelets not available, using Gaussian smoothing")
            return self.gaussian_smooth(data, sigma=1.0)
        except Exception as e:
            raise PreprocessingError(f"Wavelet smoothing failed: {e}")
    
    # ==================== NORMALIZATION METHODS ====================
    
    def vector_normalization(self, data, **kwargs):
        """Apply vector (unit) normalization"""
        try:
            print("📊 Applying vector normalization")
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                norm = np.linalg.norm(spectrum)
                
                if norm > 1e-8:
                    data_array[i, :] = spectrum / norm
                else:
                    data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Vector normalization failed: {e}")
    
    def area_normalization(self, data, **kwargs):
        """Apply area normalization"""
        try:
            print("📊 Applying area normalization")
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                area = np.trapz(np.abs(spectrum))
                
                if area > 1e-8:
                    data_array[i, :] = spectrum / area
                else:
                    data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Area normalization failed: {e}")
    
    def maximum_normalization(self, data, **kwargs):
        """Apply maximum normalization"""
        try:
            print("📊 Applying maximum normalization")
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                max_val = np.max(np.abs(spectrum))
                
                if max_val > 1e-8:
                    data_array[i, :] = spectrum / max_val
                else:
                    data_array[i, :] = spectrum
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Maximum normalization failed: {e}")
    
    # ==================== DERIVATIVE METHODS ====================
    
    def savgol_derivative(self, data, window_length=11, polyorder=2, deriv=1, **kwargs):
        """Apply Savitzky-Golay derivative"""
        try:
            print(f"📊 Applying Savitzky-Golay {deriv} derivative")
            
            from scipy.signal import savgol_filter
            
            # Ensure odd window length
            if window_length % 2 == 0:
                window_length += 1
            
            # Ensure valid parameters
            window_length = min(window_length, data.shape[1])
            if window_length <= polyorder:
                polyorder = max(1, window_length - 1)
                if polyorder % 2 == 0:
                    polyorder -= 1
            
            derivative_data = savgol_filter(data.values, window_length=window_length, 
                                          polyorder=polyorder, deriv=deriv, axis=1)
            result = pd.DataFrame(derivative_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Savitzky-Golay derivative failed: {e}")
    
    def finite_difference(self, data, order=1, **kwargs):
        """Apply finite difference derivative"""
        try:
            print(f"📊 Applying finite difference (order {order})")
            
            data_array = data.values.copy()
            
            for _ in range(order):
                # Apply finite difference
                diff_data = np.diff(data_array, axis=1)
                # Pad to maintain original shape
                data_array = np.pad(diff_data, ((0, 0), (0, 1)), mode='edge')
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Finite difference failed: {e}")
    
    def gap_segment_derivative(self, data, gap=1, **kwargs):
        """Apply gap-segment derivative"""
        try:
            print(f"📊 Applying gap-segment derivative (gap={gap})")
            
            data_array = data.values.copy()
            derivative_data = np.zeros_like(data_array)
            
            # Calculate gap-segment derivative
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                for j in range(gap, len(spectrum) - gap):
                    derivative_data[i, j] = (spectrum[j + gap] - spectrum[j - gap]) / (2 * gap)
                
                # Handle boundaries
                derivative_data[i, :gap] = derivative_data[i, gap]
                derivative_data[i, -gap:] = derivative_data[i, -gap-1]
            
            result = pd.DataFrame(derivative_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Gap-segment derivative failed: {e}")
    
    # ==================== SIGNAL ENHANCEMENT METHODS ====================
    
    def denoising(self, data, method='wavelet', **kwargs):
        """Apply denoising methods"""
        try:
            print(f"📊 Applying {method} denoising")
            
            if method.lower() == 'wavelet':
                return self.wavelet_denoising(data, **kwargs)
            elif method.lower() == 'fft':
                return self.fft_denoising(data, **kwargs)
            elif method.lower() == 'wiener':
                return self.wiener_denoising(data, **kwargs)
            elif method.lower() == 'nlm':
                return self.nlm_denoising(data, **kwargs)
            else:
                # Default to wavelet
                return self.wavelet_denoising(data, **kwargs)
                
        except Exception as e:
            raise PreprocessingError(f"Denoising failed: {e}")
    
    def wavelet_denoising(self, data, wavelet='db4', threshold_mode='soft', **kwargs):
        """Wavelet denoising"""
        try:
            import pywt
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(spectrum, wavelet)
                
                # Estimate noise level and set threshold
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
                
                # Apply thresholding
                coeffs_thresh = [pywt.threshold(c, threshold, threshold_mode) for c in coeffs]
                
                # Reconstruct
                data_array[i, :] = pywt.waverec(coeffs_thresh, wavelet)
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except ImportError:
            print("⚠️ PyWavelets not available, using Gaussian smoothing")
            return self.gaussian_smooth(data, sigma=1.0)
        except Exception as e:
            raise PreprocessingError(f"Wavelet denoising failed: {e}")
    
    def fft_denoising(self, data, cutoff_freq=0.1, **kwargs):
        """FFT-based denoising"""
        try:
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # FFT
                fft_spectrum = np.fft.fft(spectrum)
                freqs = np.fft.fftfreq(len(spectrum))
                
                # Low-pass filter
                fft_spectrum[np.abs(freqs) > cutoff_freq] = 0
                
                # Inverse FFT
                data_array[i, :] = np.real(np.fft.ifft(fft_spectrum))
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"FFT denoising failed: {e}")
    
    def wiener_denoising(self, data, noise_power=0.1, **kwargs):
        """Wiener filter denoising"""
        try:
            from scipy.signal import wiener
            
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                # Apply Wiener filter
                data_array[i, :] = wiener(spectrum, noise=noise_power)
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            # Fallback to moving average
            return self.moving_average(data, window_size=5)
    
    def nlm_denoising(self, data, **kwargs):
        """Non-local means denoising (simplified version)"""
        try:
            # Simplified implementation using local averaging
            data_array = data.values.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                denoised = np.zeros_like(spectrum)
                
                window_size = 5
                for j in range(len(spectrum)):
                    start = max(0, j - window_size)
                    end = min(len(spectrum), j + window_size + 1)
                    
                    # Simple local averaging
                    denoised[j] = np.mean(spectrum[start:end])
                
                data_array[i, :] = denoised
            
            result = pd.DataFrame(data_array, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"NLM denoising failed: {e}")
    
    def peak_alignment(self, data, method='dtw', reference_spectrum=None, **kwargs):
        """Apply peak alignment methods"""
        try:
            print(f"📊 Applying {method.upper()} peak alignment")
            
            if method.lower() == 'dtw':
                return self.dtw_alignment(data, reference_spectrum, **kwargs)
            elif method.lower() == 'cow':
                return self.cow_alignment(data, reference_spectrum, **kwargs)
            elif method.lower() == 'ics':
                return self.ics_alignment(data, reference_spectrum, **kwargs)
            elif method.lower() == 'pafft':
                return self.pafft_alignment(data, reference_spectrum, **kwargs)
            else:
                # Default to simple correlation-based alignment
                return self.simple_alignment(data, reference_spectrum, **kwargs)
                
        except Exception as e:
            raise PreprocessingError(f"Peak alignment failed: {e}")
    
    def dtw_alignment(self, data, reference_spectrum=None, **kwargs):
        """Dynamic Time Warping alignment (simplified)"""
        try:
            data_array = data.values.copy()
            
            if reference_spectrum is None:
                reference_spectrum = np.mean(data_array, axis=0)
            
            # Simplified DTW - use interpolation for alignment
            aligned_data = data_array.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # Find correlation-based shift
                correlation = np.correlate(spectrum, reference_spectrum, mode='full')
                shift = np.argmax(correlation) - len(reference_spectrum) + 1
                
                # Apply shift using interpolation
                if shift != 0:
                    x_old = np.arange(len(spectrum))
                    x_new = x_old - shift
                    
                    # Interpolate to align
                    aligned_spectrum = np.interp(x_old, x_new, spectrum, 
                                               left=spectrum[0], right=spectrum[-1])
                    aligned_data[i, :] = aligned_spectrum
            
            result = pd.DataFrame(aligned_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"DTW alignment failed: {e}")
    
    def cow_alignment(self, data, reference_spectrum=None, **kwargs):
        """Correlation Optimized Warping alignment (simplified)"""
        try:
            # Use DTW as approximation for COW
            return self.dtw_alignment(data, reference_spectrum, **kwargs)
            
        except Exception as e:
            raise PreprocessingError(f"COW alignment failed: {e}")
    
    def ics_alignment(self, data, reference_spectrum=None, **kwargs):
        """Interval Correlation Shifting alignment"""
        try:
            # Simplified ICS implementation
            return self.dtw_alignment(data, reference_spectrum, **kwargs)
            
        except Exception as e:
            raise PreprocessingError(f"ICS alignment failed: {e}")
    
    def pafft_alignment(self, data, reference_spectrum=None, **kwargs):
        """Phase-And-FFT alignment"""
        try:
            data_array = data.values.copy()
            
            if reference_spectrum is None:
                reference_spectrum = np.mean(data_array, axis=0)
            
            aligned_data = data_array.copy()
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                
                # FFT-based phase alignment
                fft_ref = np.fft.fft(reference_spectrum)
                fft_spec = np.fft.fft(spectrum)
                
                # Cross-correlation in frequency domain
                cross_corr = fft_ref * np.conj(fft_spec)
                cross_corr_time = np.fft.ifft(cross_corr)
                
                # Find shift
                shift = np.argmax(np.abs(cross_corr_time))
                if shift > len(spectrum) // 2:
                    shift -= len(spectrum)
                
                # Apply shift
                if shift != 0:
                    aligned_data[i, :] = np.roll(spectrum, shift)
            
            result = pd.DataFrame(aligned_data, index=data.index, columns=data.columns)
            return result
            
        except Exception as e:
            raise PreprocessingError(f"PAFFT alignment failed: {e}")
    
    def simple_alignment(self, data, reference_spectrum=None, **kwargs):
        """Simple correlation-based alignment"""
        try:
            return self.dtw_alignment(data, reference_spectrum, **kwargs)
            
        except Exception as e:
            raise PreprocessingError(f"Simple alignment failed: {e}")
    
    # ======================== New specialized algorithms ========================
    
    def raman_fluorescence_removal(self, data, method='modpoly', **kwargs):
        """Raman fluorescence background removal"""
        try:
            print(f"🔧 Raman fluorescence background removal: {method}")
            
            fluor_remover = RamanFluorescenceRemoval(method=method, **kwargs)
            result = fluor_remover.fit_transform(data)
            
            print(f"✅ Fluorescence background removal completed")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Raman fluorescence background removal failed: {e}")
    
    def raman_shift_calibration(self, data, wavenumbers, standard_spectrum, **kwargs):
        """Raman Shift calibration"""
        try:
            print(f"🔧 Raman Shift calibration")
            
            calibration = RamanShiftCalibration(**kwargs)
            shift, scale = calibration.calibrate(wavenumbers, standard_spectrum)
            
            print(f"✅ Raman Shift calibration completed: shift={shift:.2f}, scale={scale:.6f}")
            return data  # Note: In actual application, wavenumber axis needs to be adjusted
            
        except Exception as e:
            raise PreprocessingError(f"Raman Shift calibration failed: {e}")
    
    def atmospheric_compensation(self, data, wavenumbers, **kwargs):
        """MIR atmospheric compensation"""
        try:
            print(f"🔧 MIR atmospheric compensation")
            
            compensator = AtmosphericCompensation(**kwargs)
            _, compensated_data = compensator.fit_transform(wavenumbers, data)
            
            print(f"✅ Atmospheric compensation completed")
            return compensated_data
            
        except Exception as e:
            raise PreprocessingError(f"MIR atmospheric compensation failed: {e}")
    
    def water_peak_removal(self, data, wavelengths, **kwargs):
        """NIR water peak removal"""
        try:
            print(f"🔧 NIR water peak removal")
            
            remover = WaterPeakRemoval(wavelengths=wavelengths, **kwargs)
            result = remover.fit_transform(data)
            
            print(f"✅ Water peak removal completed")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"NIR water peak removal failed: {e}")
    
    def model_transfer(self, data_master, data_slave, method='pds', **kwargs):
        """Model transfer"""
        try:
            print(f"🔧 Model transfer: {method}")
            
            if method.lower() == 'pds':
                transfer_model = PDS(**kwargs)
            elif method.lower() == 'sbc':
                transfer_model = SBC(**kwargs)
            else:
                raise ValueError(f"Unknown model transfer method: {method}")
            
            # Fit transfer model
            transfer_model.fit(data_master, data_slave)
            
            # Apply transfer
            result = transfer_model.transform(data_slave)
            
            print(f"✅ Model transfer completed")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Model transfer failed: {e}")
