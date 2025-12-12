# app/models/data_model.py

import pandas as pd
import numpy as np
from app.utils.label_processor import EnhancedLabelProcessor

class DataModel:
    def __init__(self, logger=None):
        """Initializes the DataModel."""
        self.data = None
        self.partitioned_data = None
        self.preprocessed_data = None
        self.file_path = None
        self.label_processor = EnhancedLabelProcessor()
        self._x = None
        self._y = None
        self.features = None
        self.X_processed = None
        
        # Attributes for direct access, compatible with existing code
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_selected = None
        self.X_test_selected = None
        self.selected_features = None
        self.label_mapping = None
        self.task_type = None
        
        # 🔧 Optimization #2: Centralized wavelength storage
        self.wavelengths = None  # Full wavelength array
        self.selected_wavelengths = None  # Selected feature wavelengths
        self.spectral_type = None  # Spectral type (Raman, NIR, etc.)

    def set_data(self, data, wavelengths=None):
        """
        Set data with optional wavelength information.
        
        Args:
            data: The spectral data DataFrame
            wavelengths: Optional wavelength array (defaults to column names)
        """
        self.data = data
        # When new data is set, reset derived attributes
        self._x = None
        self._y = None
        
        # Auto-extract wavelengths from data columns if not provided
        if wavelengths is not None:
            self.wavelengths = wavelengths
        elif data is not None and data.shape[1] > 1:
            # Assume first column is label, rest are features/wavelengths
            self.wavelengths = data.iloc[:, 1:].columns.values

    def get_data(self):
        return self.data

    def get_X(self):
        """Returns the feature data (X)."""
        if self._x is None and self.data is not None:
            # This logic assumes the first column is the label if more than one column exists.
            if self.data.shape[1] > 1:
                self._x = self.data.iloc[:, 1:]
            else:
                self._x = self.data.copy()
        return self._x

    def get_y(self):
        """Returns the label data (y)."""
        if self._y is None and self.data is not None:
            if self.data.shape[1] > 1:
                self._y = self.data.iloc[:, 0]
            else:
                # No labels if only one column
                self._y = None
        return self._y

    def get_wavelengths(self, selected_only=False):
        """
        Returns the wavelengths.
        
        Args:
            selected_only: If True, return selected wavelengths only (after feature selection)
            
        Returns:
            Wavelength array or None
        """
        if selected_only and self.selected_wavelengths is not None:
            return self.selected_wavelengths
        
        if self.wavelengths is not None:
            return self.wavelengths
        
        # Fallback: Extract from feature data column names
        X = self.get_X()
        if X is not None:
            return X.columns
        return None
    
    def set_wavelengths(self, wavelengths, is_selected=False):
        """
        Set wavelengths array.
        
        Args:
            wavelengths: Wavelength array
            is_selected: If True, set as selected wavelengths (after feature selection)
        """
        if is_selected:
            self.selected_wavelengths = wavelengths
        else:
            self.wavelengths = wavelengths

    def has_data(self):
        return self.data is not None

    def set_partitioned_data(self, partitioned_data):
        """
        Sets the partitioned data and updates direct access attributes
        for backward compatibility.
        """
        self.partitioned_data = partitioned_data
        if partitioned_data:
            self.X_train = partitioned_data.get('X_train')
            self.y_train = partitioned_data.get('y_train')
            self.X_test = partitioned_data.get('X_test')
            self.y_test = partitioned_data.get('y_test')
            self.label_mapping = partitioned_data.get('label_mapping')
        else:
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.label_mapping = None

    def get_partitioned_data(self):
        return self.partitioned_data

    def has_partitioned_data(self):
        return self.partitioned_data is not None

    def set_preprocessed_data(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def get_preprocessed_data(self):
        return self.preprocessed_data

    def has_preprocessed_data(self):
        return self.preprocessed_data is not None

    def get_modeling_data(self, prefer_selected=True):
        """
        Unified method to get data for modeling with intelligent fallback.
        
        Args:
            prefer_selected: If True, prioritize feature-selected data
            
        Returns:
            dict: {'X_train': DataFrame, 'X_test': DataFrame, 'y_train': Series, 'y_test': Series, 'source': str}
        """
        X_train = None
        X_test = None
        source = "unknown"
        
        # Priority 1: Feature-selected data (if available and preferred)
        if prefer_selected and self.X_train_selected is not None:
            X_train = self.X_train_selected
            X_test = self.X_test_selected
            source = "feature_selected"
        
        # Priority 2: Preprocessed data (if available and no feature selection)
        elif self.X_processed is not None:
            # 🔧 FIX: Handle two cases:
            # Case 1: Preprocessing done AFTER partitioning - X_train/X_test already updated
            if self.X_train is not None and hasattr(self, 'X_train_original') and self.X_train_original is not None:
                # X_train and X_test have been updated with preprocessed data
                X_train = self.X_train
                X_test = self.X_test
                source = "preprocessed"
            # Case 2: Preprocessing done BEFORE partitioning - need to split X_processed using train/test indices
            elif self.X_train is not None and self.X_test is not None:
                # Use train/test indices to select from X_processed
                train_indices = self.X_train.index
                test_indices = self.X_test.index
                
                # Select rows from X_processed using indices (preserves order)
                try:
                    # Try to match by index first (most accurate)
                    if all(idx in self.X_processed.index for idx in train_indices):
                        X_train = self.X_processed.loc[train_indices]
                    else:
                        # Fallback: use position-based splitting
                        print("⚠️ Warning: Index mismatch, using position-based splitting")
                        X_train = self.X_processed.iloc[:len(train_indices)]
                    
                    if len(test_indices) > 0:
                        if all(idx in self.X_processed.index for idx in test_indices):
                            X_test = self.X_processed.loc[test_indices]
                        else:
                            X_test = self.X_processed.iloc[len(train_indices):]
                    else:
                        X_test = None
                except Exception as e:
                    print(f"⚠️ Warning: Error splitting preprocessed data: {e}, using position-based fallback")
                    X_train = self.X_processed.iloc[:len(train_indices)]
                    X_test = self.X_processed.iloc[len(train_indices):] if len(test_indices) > 0 else None
                source = "preprocessed"
            elif self.X_train is not None:
                # Only train data available, use it directly
                train_indices = self.X_train.index
                try:
                    if all(idx in self.X_processed.index for idx in train_indices):
                        X_train = self.X_processed.loc[train_indices]
                    else:
                        X_train = self.X_processed.iloc[:len(train_indices)]
                except Exception:
                    X_train = self.X_processed.iloc[:len(train_indices)]
                X_test = None
                source = "preprocessed"
            else:
                # Fallback: use X_processed as training data (no split yet)
                X_train = self.X_processed
                X_test = None
                source = "preprocessed"
        
        # Priority 3: Original split data (fallback)
        else:
            X_train = self.X_train
            X_test = self.X_test
            source = "original"
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'source': source
        }
    
    def clear_data(self):
        """Clears all data attributes."""
        self.data = None
        self.partitioned_data = None
        self.preprocessed_data = None
        self.file_path = None
        self._x = None
        self._y = None
        self.features = None
        self.X_processed = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_selected = None
        self.X_test_selected = None
        self.selected_features = None
        self.label_mapping = None
        self.task_type = None
        
        # Clear wavelengths when clearing data
        if hasattr(self, 'wavelengths'):
            self.wavelengths = None
        if hasattr(self, 'selected_wavelengths'):
            self.selected_wavelengths = None