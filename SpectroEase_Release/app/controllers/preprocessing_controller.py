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
        """Apply preprocessing methods to the current spectral data"""
        if self.data_model is None or self.data_model.data is None:
            print("ERROR: Operation failed")
            return False
            
        data = self.data_model.data.copy()
        print(f"Starting preprocessing with {len(methods)} methods")
        print(f"Input data shape: {data.shape}")

        try:
            if not hasattr(self, 'preprocessing_model'):
                self.preprocessing_model = PreprocessingModel()
                
            self.preprocessing_model.applied_methods = {}
            applied_methods = []

            label_processor = EnhancedLabelProcessor()
            
            first_col = data.iloc[:, 0]
            detected_task_type = label_processor.detect_task_type(first_col)
            print(f"🤖 Target column task type detection: {detected_task_type}")
            
            if detected_task_type == 'classification':
                print("🔧 Classification target detected - preserving string labels")
                target_column = first_col.name
                print(f"Target column '{target_column}' detected as classification target")
            else:
                print("🔧 Regression target detected - attempting numeric conversion")
                try:
                    sample_labels = [str(label) for label in first_col.head(5)]
                    has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                           for label in sample_labels if label.strip())
                    
                    if has_string_labels:
                        raise ValueError(f"String labels detected in target column: {sample_labels[:3]}")
                    
                    pd.to_numeric(first_col, errors='coerce')
                    numeric_result = pd.to_numeric(first_col, errors='coerce')
                    if pd.isna(numeric_result).any():
                        raise ValueError("String labels detected in numeric conversion")
                    target_column = first_col.name
                    print(f"Target column '{target_column}' detected as numeric target")
                except (ValueError, TypeError) as e:
                    print(f"❌ Numeric conversion failed: {e}")
                    print("🔧 Treating as classification target instead")
                    target_column = first_col.name
            
            has_string_labels = False
            
            try:
                sample_labels = [str(label) for label in first_col.head(5)]
                has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                       for label in sample_labels if label.strip())
                
                if has_string_labels:
                    raise ValueError(f"String labels detected: {sample_labels[:3]}")
                
                pd.to_numeric(first_col, errors='coerce')
                numeric_result = pd.to_numeric(first_col, errors='coerce')
                if pd.isna(numeric_result).any():
                    raise ValueError("String labels detected in second numeric conversion")
                features_data = data
                labels_column = None
                print("First column is numeric - treating as features")
            except (ValueError, TypeError):
                has_string_labels = True
                labels_column = first_col.copy()
                features_data = data.iloc[:, 1:].copy()
                print(f"Detected string labels in first column: {list(first_col.unique())}")
                print(f"Features data shape after removing labels: {features_data.shape}")
            
            processed_data = features_data

            for method in methods:
                if method in self.method_mapping:
                    service_method = self.method_mapping[method]
                    print(f"Mapping method: {method} -> {service_method}")
                else:
                    service_method = method
                    print(f"Using method directly: {method}")
                
                if params_dict and method in params_dict:
                    params = params_dict[method]
                else:
                    params = {}
                
                if method == 'scatter_correction' and 'method' in params:
                    scatter_method = params['method']
                    if scatter_method in self.scatter_method_mapping:
                        service_method = self.scatter_method_mapping[scatter_method]
                        print(f"Mapped scatter method: {scatter_method} -> {service_method}")
                
                print(f"Applying method: {service_method}")
                print(f"Parameters: {params}")
                
                try:
                    processed_data = self.preprocessing_service.apply_method(processed_data, service_method, params)
                    self.preprocessing_model.add_applied_method(service_method, str(params))
                    applied_methods.append(service_method)
                    print(f"Successfully applied: {service_method}")
                    
                except Exception as method_error:
                    print(f"Failed to apply method {service_method}: {method_error}")
                    print(f"Skipping method and continuing...")
                    continue
            
            if has_string_labels and labels_column is not None:
                final_data = pd.DataFrame()
                final_data[data.columns[0]] = labels_column
                
                for i, col in enumerate(data.columns[1:]):
                    final_data[col] = processed_data.iloc[:, i]
                
                print(f"Recombined data shape: {final_data.shape}")
                print(f"Final data columns: {list(final_data.columns[:5])}...")
            else:
                final_data = processed_data
            
            self.data_model.data = final_data
            
            print(f"Preprocessing completed successfully")
            print(f"Applied {len(applied_methods)} methods: {applied_methods}")
            print(f"Final data shape: {final_data.shape}")
            
            self.logger.info(f"Applied preprocessing methods: {applied_methods}")
            
            return True
            
        except Exception as e:
            print(f"Preprocessing failed with error: {e}")
            self.logger.error(f"Preprocessing failed: {e}")
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
