# app/controllers/preprocessing_controller.py

# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
try:
    from app.utils.exceptions import PreprocessingError
    from app.utils.logger import setup_logger
except ImportError:
    try:
        from exceptions import PreprocessingError
        from logger import setup_logger
    except ImportError:
        from ..utils.exceptions import PreprocessingError
        from ..utils.logger import setup_logger
try:
    from app.services.preprocessing_service import PreprocessingService
except ImportError:
    # Try direct import (for PyInstaller packaging)
    try:
        from preprocessing_service import PreprocessingService
    except ImportError:
        # Finally try relative import
        from ..services.preprocessing_service import PreprocessingService
try:
    from app.models.preprocessing_model import PreprocessingModel
    from app.utils.label_processor import EnhancedLabelProcessor
except ImportError:
    try:
        from preprocessing_model import PreprocessingModel
        from label_processor import EnhancedLabelProcessor
    except ImportError:
        from ..models.preprocessing_model import PreprocessingModel
        from ..utils.label_processor import EnhancedLabelProcessor

import pandas as pd

class PreprocessingController:
    def __init__(self, view, translator, data_model, plugins):
        self.view = view
        self.translator = translator
        self.data_model = data_model
        self.preprocessing_service = PreprocessingService(plugins)
        # V1.4.0: Let service access view for dynamically added plugins
        self.preprocessing_service.view = view
        self.logger = setup_logger()
        
        self.method_mapping = {
            'despiking': 'Despiking',
            'baseline_correction': 'Baseline Correction',
            'smoothing': 'Savitzky-Golay Filter',
            'scatter_correction': 'Standard Normal Variate (SNV)',
            'standardization': 'Standard Scale',
            'normalization': 'Min-Max Scale',
            'derivative': 'First Derivative',
            'outlier_detection': 'Outlier Detection',
            'denoising': 'Gaussian Smooth',
            'alignment': 'Peak Alignment'
        }
        
        self.scatter_method_mapping = {
            'SNV': 'Standard Normal Variate (SNV)',
            'MSC': 'Multiplicative Scatter Correction (MSC)',
            'EMSC': 'Extended Multiplicative Scatter Correction (EMSC)',
            'RNV': 'Robust Normal Variate (RNV)',
            'OSC': 'Orthogonal Signal Correction (OSC)'
        }

    def apply_preprocessing(self, methods, params_dict=None):
        """
        Apply preprocessing methods to the spectral data.
        Handles both partitioned (train/test) and unpartitioned data.
        """
        self.logger.info(f"Starting preprocessing...")
        if self.data_model is None:
            self.logger.error("DataModel is not initialized.")
            print("ERROR: DataModel not found.")
            return False

        # --- 1. Get the correct data source ---
        is_partitioned = self.data_model.has_partitioned_data()
        X_train, X_test, X_full = None, None, None

        if is_partitioned:
            self.logger.info("Processing partitioned data (X_train, X_test).")
            X_train = self.data_model.X_train
            X_test = self.data_model.X_test
            if X_train is None or X_test is None:
                self.logger.error("Partitioned data (X_train/X_test) is missing in DataModel.")
                print("ERROR: Partitioned data is missing.")
                return False
        else:
            self.logger.info("Processing full dataset.")
            X_full = self.data_model.get_X()
            if X_full is None:
                self.logger.error("Full dataset (X) is missing in DataModel.")
                print("ERROR: No data to process.")
                return False

        # --- 2. Define the processing logic ---
        def _process_data(data_to_process, ordered_methods, params_dict):
            processed_data = data_to_process.copy()
            applied_methods_log = []
            
            # Extract global parameters (wavelengths, spectral_type) that should be merged into all algorithm params
            global_params = {}
            if 'wavelengths' in params_dict and params_dict['wavelengths'] is not None:
                global_params['wavelengths'] = params_dict['wavelengths']
            if 'spectral_type' in params_dict and params_dict['spectral_type'] is not None:
                global_params['spectral_type'] = params_dict['spectral_type']
            
            for method in ordered_methods:
                if method in self.method_mapping:
                    service_method = self.method_mapping[method]
                else:
                    service_method = method
                
                params = params_dict.get(method, {}).copy()  # Make a copy to avoid modifying original
                
                # Merge global parameters into algorithm params (for custom algorithms that need wavelengths)
                # Global params take precedence if algorithm param is None or empty
                for key, value in global_params.items():
                    if key not in params or params[key] is None or (isinstance(params[key], list) and len(params[key]) == 0):
                        params[key] = value
                    elif params[key] is not None:
                        # Keep algorithm-specific param if it exists and is not empty
                        pass

                if method == 'scatter_correction' and 'method' in params:
                    scatter_method = params['method']
                    if scatter_method in self.scatter_method_mapping:
                        service_method = self.scatter_method_mapping[scatter_method]

                self.logger.info(f"Applying method: {service_method} with params: {params}")
                print(f"Applying method: {service_method}")
                if global_params:
                    print(f"   Global params included: {list(global_params.keys())}")
                try:
                    processed_data = self.preprocessing_service.apply_method(processed_data, service_method, params)
                    applied_methods_log.append(service_method)
                    self.logger.info(f"Successfully applied: {service_method}")
                    print(f"Successfully applied: {service_method}")
                except Exception as method_error:
                    self.logger.error(f"Failed to apply method {service_method}: {method_error}", exc_info=True)
                    print(f"Failed to apply method {service_method}: {method_error}")
                    import traceback
                    traceback.print_exc()
                    continue
            return processed_data, applied_methods_log

        try:
            # --- 3. Order methods (logic from original function) ---
            expert_order = [
                'Despiking', 'Peak Alignment', 'Baseline Correction', 
                'Standard Normal Variate (SNV)', 'Multiplicative Scatter Correction (MSC)',
                'Extended Multiplicative Scatter Correction (EMSC)', 'Robust Normal Variate (RNV)',
                'Orthogonal Signal Correction (OSC)', 'First Derivative', 'Second Derivative',
                'Savitzky-Golay Filter', 'Moving Average', 'Median Filter', 'Gaussian Smooth',
                'Wavelet Smooth', 'Normalize', 'Standard Scale', 'Min-Max Scale',
                'Vector Normalization', 'Area Normalization', 'Maximum Normalization'
            ]
            ordered_methods = []
            for expert_method in expert_order:
                if expert_method in methods:
                    ordered_methods.append(expert_method)
            for method in methods:
                if method not in ordered_methods:
                    ordered_methods.append(method)
            
            if methods:
                self.logger.info(f"Original user order: {methods}")
                self.logger.info(f"Expert chemometrics order: {ordered_methods}")
                print(f"🎯 Expert chemometrics order: {ordered_methods}")
            else:
                self.logger.info(f"Applying preprocessing methods")
                print(f"🎯 Applying preprocessing methods")

            # --- 4. Apply processing ---
            if not hasattr(self, 'preprocessing_model'):
                self.preprocessing_model = PreprocessingModel()
            self.preprocessing_model.applied_methods = {}
            
            final_applied_methods = []

            if is_partitioned:
                # 🔧 FIX: Preserve original split data on first preprocessing
                if not hasattr(self.data_model, 'X_train_original') or self.data_model.X_train_original is None:
                    self.data_model.X_train_original = X_train.copy()
                    self.data_model.X_test_original = X_test.copy() if X_test is not None else None
                    print(f"✅ Saved original partitioned data (before preprocessing)")
                    print(f"   X_train_original: {self.data_model.X_train_original.shape}")
                    if self.data_model.X_test_original is not None:
                        print(f"   X_test_original: {self.data_model.X_test_original.shape}")
                
                print(f"Input X_train shape: {X_train.shape}")
                X_train_processed, train_methods = _process_data(X_train, ordered_methods, params_dict)
                print(f"Processed X_train shape: {X_train_processed.shape}")

                print(f"Input X_test shape: {X_test.shape}")
                X_test_processed, test_methods = _process_data(X_test, ordered_methods, params_dict)
                print(f"Processed X_test shape: {X_test_processed.shape}")
                
                self.data_model.X_train = X_train_processed
                self.data_model.X_test = X_test_processed
                
                # 🔧 FIX: Preserve original index order, don't sort
                # This ensures train/test can be correctly split later
                combined_processed = pd.concat([X_train_processed, X_test_processed])
                # Only sort if indices are not already in correct order
                # But preserve the original order to maintain train/test split
                self.data_model.X_processed = combined_processed
                final_applied_methods = train_methods
            else:
                print(f"Input X_full shape: {X_full.shape}")
                X_full_processed, full_methods = _process_data(X_full, ordered_methods, params_dict)
                print(f"Processed X_full shape: {X_full_processed.shape}")

                self.data_model.X_processed = X_full_processed
                final_applied_methods = full_methods

            # --- 5. Finalize and log ---
            for method_name in final_applied_methods:
                original_method_key = next((key for key, value in self.method_mapping.items() if value == method_name), method_name)
                self.preprocessing_model.add_applied_method(method_name, str(params_dict.get(original_method_key, {})))

            self.logger.info(f"Preprocessing completed successfully")
            print(f"Preprocessing completed successfully")
            
            return True

        except Exception as e:
            self.logger.error(f"Preprocessing failed with a critical error: {e}", exc_info=True)
            print(f"Preprocessing failed with error: {e}")
            return False

    def get_available_methods(self):
        """Get list of available preprocessing methods"""
        return list(self.method_mapping.keys())
        
    def get_method_mapping(self):
        """Get the mapping between method names and service methods"""
        return self.method_mapping.copy()
    
    def _get_method_mapping(self):
        """Internal method to get method mapping"""
        return self.method_mapping.copy()
