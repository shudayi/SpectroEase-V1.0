import numpy as np
import pandas as pd
from typing import Tuple, Optional

def safe_float_convert(val):
    """Safely convert value to float, return NaN if conversion fails"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan



class SpectrumConverter:
    """Spectrum data format converter"""
    
    @staticmethod
    def read_spectrum_file(file_path: str, format_type: str = 'csv') -> Tuple[np.ndarray, np.ndarray]:
        """
        Read spectrum data file
        
        Args:
            file_path: File path
            format_type: File format ('csv', 'txt', 'xlsx')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (wavelength array, spectrum data array)
        """
        try:
            if format_type.lower() == 'csv':
                data = pd.read_csv(file_path)
            elif format_type.lower() == 'txt':
                data = pd.read_csv(file_path, delimiter='\t')
            elif format_type.lower() == 'xlsx':
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
            
            # **CRITICAL FIX: Smart detection of data structure**
            # Check if first column contains classification labels (could be numeric like 1,2 or string)
            first_col = data.iloc[:, 0]
            first_col_sample = first_col.iloc[:5] if len(first_col) > 5 else first_col
            
            print(f"DEBUG: First column name: '{data.columns[0]}'")
            print(f"DEBUG: First column sample values: {list(first_col_sample)}")
            
            # **CRITICAL FIX: Check if first column is 'Label' or contains classification data**
            has_classification_labels = False
            
            # Method 1: Check column name
            if data.columns[0].lower() in ['label', 'class', 'category', 'target', 'y']:
                has_classification_labels = True
                print("DEBUG: Detected classification column by name")
            
            # Method 2: Check if first column looks like target/label column for regression or classification
            elif data.shape[1] > 50:  # If there are many columns (like spectral data)
                # This suggests first column is likely a target/label column
                has_classification_labels = True
                print("DEBUG: Detected target column - many spectral columns present")
            
            # Method 3: Check if first column contains non-numeric strings
            else:
                for val in first_col_sample:
                    if pd.isna(val):
                        continue
                    try:
                        float(val)  # Try to convert to float
                    except (ValueError, TypeError):
                        has_classification_labels = True
                        print("DEBUG: Detected classification column by non-numeric values")
                        break
            
            if has_classification_labels:
                # First column contains classification labels, not wavelengths
                # In this case, assume wavelengths are column headers or need to be generated
                print("Detected classification labels in first column")
                print(f"Sample labels: {list(first_col_sample)}")
                
                # Extract spectral data (all columns except first)
                spectral_data = data.iloc[:, 1:]
                
                # **CRITICAL FIX: Check if column names are wavelengths**
                # For NIR spectral data, column names should be numeric wavelength values
                if spectral_data.columns.dtype == 'object':
                    # Try to convert column names to wavelengths
                    try:
                        # Test if all column names can be converted to numbers
                        wavelength_values = []
                        for col in spectral_data.columns:
                            try:
                                wave_val = float(col)
                                wavelength_values.append(wave_val)
                            except (ValueError, TypeError):
                                raise ValueError(f"Column '{col}' is not a valid wavelength")
                        
                        wavelengths = np.array(wavelength_values)
                        print(f"✅ Extracted wavelengths from column headers: {len(wavelengths)} points")
                        print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                        
                        # Validate wavelength range for NIR spectroscopy
                        if wavelengths.min() >= 180 and wavelengths.max() <= 3000:
                            print(f"✅ Valid NIR/Vis wavelength range detected")
                        else:
                            print(f"⚠️ Unusual wavelength range detected")
                            
                    except (ValueError, TypeError) as e:
                        # If column names are not numeric, generate sequential wavelengths
                        print(f"⚠️ Column names are not wavelengths: {e}")
                        wavelengths = np.arange(1, len(spectral_data.columns) + 1, dtype=float)
                        print(f"Generated sequential wavelengths: {len(wavelengths)} points")
                else:
                    wavelengths = np.arange(1, len(spectral_data.columns) + 1, dtype=float)
                
                # Convert spectral data to float, handling any remaining string values
                spectra_values = []
                for _, row in spectral_data.iterrows():
                    spectrum_row = []
                    for val in row:
                        try:
                            spectrum_row.append(float(val))
                        except (ValueError, TypeError):
                            spectrum_row.append(0.0)  # Replace non-numeric with 0
                    spectra_values.append(spectrum_row)
                
                spectra = np.array(spectra_values)
                print(f"✅ Extracted spectra: {spectra.shape[0]} samples x {spectra.shape[1]} wavelengths")
                
            else:
                # Traditional format: first column is wavelengths, rest are spectra
                try:
                    wavelengths = data.iloc[:, 0].values.astype(float)
                    spectra = data.iloc[:, 1:].values.astype(float).T
                    print(f"✅ Traditional format: {len(wavelengths)} wavelengths, {spectra.shape[0]} spectra")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to convert wavelength or spectral data to numeric: {e}")
            
            # Validate data dimensions
            if len(wavelengths) != spectra.shape[1]:
                raise ValueError(f"Wavelength array ({len(wavelengths)}) and spectrum data ({spectra.shape[1]}) dimension mismatch")
            
            return wavelengths, spectra
            
        except Exception as e:
            raise Exception(f"Failed to read data: {str(e)}")
    
    @staticmethod
    def save_spectrum_file(wavelengths: np.ndarray, spectra: np.ndarray, 
                          file_path: str, format_type: str = 'csv'):
        """
        Save spectrum data to file
        
        Args:
            wavelengths: Wavelength array
            spectra: Spectrum data array
            file_path: Save path
            format_type: File format ('csv', 'txt', 'xlsx')
        """
        try:
            # Create DataFrame
            data = pd.DataFrame(spectra.T, columns=[f'Spectrum_{i+1}' for i in range(spectra.shape[0])])
            data.insert(0, 'Wavelength', wavelengths)
            
            if format_type.lower() == 'csv':
                data.to_csv(file_path, index=False)
            elif format_type.lower() == 'txt':
                data.to_csv(file_path, index=False, sep='\t')
            elif format_type.lower() == 'xlsx':
                data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Failed to save data: {str(e)}")
    
    @staticmethod
    def transmittance_to_absorbance(transmittance: np.ndarray) -> np.ndarray:
        """
        Convert transmittance to absorbance
        
        Args:
            transmittance: Transmittance data
            
        Returns:
            np.ndarray: Absorbance data
        """
        return -np.log10(transmittance + 1e-10)  # Add small value to avoid log(0)
    
    @staticmethod
    def absorbance_to_transmittance(absorbance: np.ndarray) -> np.ndarray:
        """
        Convert absorbance to transmittance
        
        Args:
            absorbance: Absorbance data
            
        Returns:
            np.ndarray: Transmittance data
        """
        return 10**(-absorbance) 