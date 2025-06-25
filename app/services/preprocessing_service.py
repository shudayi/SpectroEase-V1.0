
# app/services/preprocessing_service.py

from utils.exceptions import PreprocessingError
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm

class PreprocessingService:
    def __init__(self, plugins: dict = None):
        self.plugins = plugins or {}    

    def apply_method(self, data: pd.DataFrame, method: str, params: dict) -> pd.DataFrame:
        """Apply a preprocessing method to the spectral data"""
        print(f"🔧 PreprocessingService.apply_method:")
        print("INFO: Processing data")
        print("INFO: Processing data")
        print("INFO: Processing data")
        
          
        if method in self.plugins:
            print("INFO: Processing data")
            algorithm: PreprocessingAlgorithm = self.plugins[method]
            result = algorithm.apply(data, params)
            print("INFO: Processing data")
            return result
        else:
              
            print("INFO: Processing data")
            
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
            elif method == "Standard Normal Variate (SNV)":
                return self.standard_normal_variate(data, **params)
            elif method == "Multiplicative Scatter Correction (MSC)":
                return self.msc(data, **params)
            elif method == "Normalize":
                return self.normalize(data, **params)
            elif method == "Standard Scale":
                return self.standard_scale(data)
            elif method == "Min-Max Scale":
                return self.min_max_scale(data)
            elif method == "First Derivative":
                return self.first_derivative(data)
            elif method == "Second Derivative":
                return self.second_derivative(data)
            else:
                error_msg = f"Unsupported preprocessing method: {method}"
                print(f"❌ {error_msg}")
                print("INFO: Processing data")
                raise PreprocessingError(error_msg)

    def baseline_correction(self, data, polynomial_order=3, **kwargs):
        """Apply baseline correction using polynomial fitting"""
        try:
            print("INFO: Processing data")
            
            # Handle parameter extraction
            if isinstance(polynomial_order, dict):
                polynomial_order = polynomial_order.get('polynomial_order', 3)
            
            # Ensure polynomial_order is an integer
            try:
                polynomial_order = int(polynomial_order)
            except (ValueError, TypeError):
                polynomial_order = 3
            
            corrected_data = data.copy()
            data_array = corrected_data.values
            
            for i in range(data_array.shape[0]):
                spectrum = data_array[i, :]
                x = np.arange(len(spectrum))
                
                  
                coeffs = np.polyfit(x, spectrum, polynomial_order)
                baseline = np.polyval(coeffs, x)
                
                  
                data_array[i, :] = spectrum - baseline
            
            corrected_data.iloc[:, :] = data_array
            print("INFO: Processing data")
            return corrected_data
            
        except Exception as e:
            raise PreprocessingError(f"Baseline Correction failed: {e}")

    def savitzky_golay_filter(self, data, window_length=15, polyorder=2, **kwargs):
        return self.savgol_filter(data, window_length, polyorder, **kwargs)

    def savgol_filter(self, data, window_length=15, polyorder=2, **kwargs):
        """Apply Savitzky-Golay smoothing filter"""
        try:
            print("INFO: Processing data")
            
              
            if window_length % 2 == 0:
                window_length += 1
                print("INFO: Processing data")
            
              
            window_length = min(window_length, data.shape[1])
            if window_length <= polyorder:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
                print("INFO: Processing data")
            
            filtered_data = savgol_filter(data.values, window_length=window_length, 
                                        polyorder=polyorder, axis=1)
            result = pd.DataFrame(filtered_data, index=data.index, columns=data.columns)
            print("INFO: Processing data")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Savitzky-Golay Filter failed: {e}")

    def moving_average(self, data, window_size=5, **kwargs):
        """Apply moving average smoothing"""
        try:
            print("INFO: Processing data")
            result = data.rolling(window=window_size, min_periods=1, axis=1).mean()
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Moving Average failed: {e}")

    def median_filter(self, data, kernel_size=3, **kwargs):
        """Apply median filter for noise reduction"""
        try:
            print("INFO: Processing data")
            result = data.rolling(window=kernel_size, min_periods=1, center=True, axis=1).median()
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Median Filter failed: {e}")

    def gaussian_smooth(self, data, sigma=1.0, **kwargs):
        """Apply Gaussian smoothing filter"""
        try:
            print("INFO: Processing data")
            from scipy.ndimage import gaussian_filter
            smoothed_data = gaussian_filter(data.values, sigma=sigma, axes=1)
            result = pd.DataFrame(smoothed_data, index=data.index, columns=data.columns)
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Gaussian Smooth failed: {e}")

    def standard_normal_variate_snv(self, data, center=True, scale=True, min_std=1e-6, **kwargs):
        return self.standard_normal_variate(data, center, scale, min_std, **kwargs)

    def standard_normal_variate(self, data, center=True, scale=True, min_std=1e-6, **kwargs):
        """Apply Standard Normal Variate (SNV) normalization"""
        try:
            print("INFO: Processing data")
            
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
            print("INFO: Processing data")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Standard Normal Variate (SNV) failed: {e}")

    def multiplicative_scatter_correction_msc(self, data, **kwargs):
        return self.msc(data, **kwargs)

    def msc(self, data, **kwargs):
        """Apply Multiplicative Scatter Correction (MSC)"""
        try:
            print("INFO: Processing data")
            mean_spectrum = data.mean(axis=0)
            msc_data = data.copy()
            
            for i in range(data.shape[0]):
                spectrum = data.iloc[i, :]
                  
                coeffs = np.polyfit(mean_spectrum, spectrum, 1)
                  
                msc_data.iloc[i, :] = (spectrum - coeffs[1]) / coeffs[0]
            
            print("INFO: Processing data")
            return msc_data
        except Exception as e:
            raise PreprocessingError(f"Multiplicative Scatter Correction (MSC) failed: {e}")

    def normalize(self, data, norm='l2', **kwargs):
        """Apply L2 or L1 normalization to spectral data"""
        try:
            print("INFO: Processing data")
            from sklearn.preprocessing import normalize
            normalized_data = normalize(data.values, norm=norm, axis=1)
            result = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Normalize failed: {e}")

    def z_score_standardization(self, data, **kwargs):
        return self.standard_scale(data, **kwargs)

    def standard_scale(self, data, **kwargs):
        """Apply standard scaling (z-score normalization)"""
        try:
            print("INFO: Processing data")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.values)
            result = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Standard Scale failed: {e}")

    def min_max_normalization(self, data, **kwargs):
        return self.min_max_scale(data, **kwargs)

    def min_max_scale(self, data, **kwargs):
        """Apply min-max scaling (0-1 normalization)"""
        try:
            print("INFO: Processing data")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data.values)
            result = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
            print("INFO: Processing data")
            return result
        except Exception as e:
            raise PreprocessingError(f"Min-Max Scale failed: {e}")

    def first_derivative(self, data, **kwargs):
        """Calculate first derivative of spectral data"""
        try:
            print("INFO: Processing data")
            derivative = data.diff(axis=1).fillna(0)
            print("INFO: Processing data")
            return derivative
        except Exception as e:
            raise PreprocessingError(f"First Derivative failed: {e}")

    def second_derivative(self, data, **kwargs):
        """Calculate second derivative of spectral data"""
        try:
            print("INFO: Processing data")
            derivative = data.diff(axis=1).diff(axis=1).fillna(0)
            print("INFO: Processing data")
            return derivative
        except Exception as e:
            raise PreprocessingError(f"Second Derivative failed: {e}")
    
    def outlier_detection(self, data, threshold=3.0, **kwargs):
        """Detect and remove outliers using Z-score method"""
        try:
            print("INFO: Processing data")
            
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
            print("INFO: Operation completed")
            return result
            
        except Exception as e:
            raise PreprocessingError(f"Outlier Detection failed: {e}")
