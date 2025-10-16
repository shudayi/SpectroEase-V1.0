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
    # 尝试直接导入（用于PyInstaller打包）
    try:
        from preprocessing_service import PreprocessingService
    except ImportError:
        # 最后尝试相对导入
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
        self.logger.info(f"Starting preprocessing with methods: {methods}")
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
            for method in ordered_methods:
                if method in self.method_mapping:
                    service_method = self.method_mapping[method]
                else:
                    service_method = method
                
                params = params_dict.get(method, {})

                if method == 'scatter_correction' and 'method' in params:
                    scatter_method = params['method']
                    if scatter_method in self.scatter_method_mapping:
                        service_method = self.scatter_method_mapping[scatter_method]

                self.logger.info(f"Applying method: {service_method} with params: {params}")
                print(f"Applying method: {service_method}")
                try:
                    processed_data = self.preprocessing_service.apply_method(processed_data, service_method, params)
                    applied_methods_log.append(service_method)
                    self.logger.info(f"Successfully applied: {service_method}")
                    print(f"Successfully applied: {service_method}")
                except Exception as method_error:
                    self.logger.error(f"Failed to apply method {service_method}: {method_error}", exc_info=True)
                    print(f"Failed to apply method {service_method}: {method_error}")
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
            
            self.logger.info(f"Original user order: {methods}")
            self.logger.info(f"Expert chemometrics order: {ordered_methods}")
            print(f"🎯 Expert chemometrics order: {ordered_methods}")

            # --- 4. Apply processing ---
            if not hasattr(self, 'preprocessing_model'):
                self.preprocessing_model = PreprocessingModel()
            self.preprocessing_model.applied_methods = {}
            
            final_applied_methods = []

            if is_partitioned:
                print(f"Input X_train shape: {X_train.shape}")
                X_train_processed, train_methods = _process_data(X_train, ordered_methods, params_dict)
                print(f"Processed X_train shape: {X_train_processed.shape}")

                print(f"Input X_test shape: {X_test.shape}")
                X_test_processed, test_methods = _process_data(X_test, ordered_methods, params_dict)
                print(f"Processed X_test shape: {X_test_processed.shape}")
                
                self.data_model.X_train = X_train_processed
                self.data_model.X_test = X_test_processed
                
                combined_processed = pd.concat([X_train_processed, X_test_processed]).sort_index()
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

            self.logger.info(f"Preprocessing completed. Applied {len(final_applied_methods)} methods: {final_applied_methods}")
            print(f"Preprocessing completed successfully. Applied {len(final_applied_methods)} methods: {final_applied_methods}")
            
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
