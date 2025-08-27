# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
from app.utils.logger import setup_logger
from app.utils.exceptions import FeatureSelectionError
from app.services.feature_selection_service import FeatureSelectionService
from app.models.feature_selection_model import FeatureSelectionModel
from app.utils.label_processor import EnhancedLabelProcessor

class FeatureSelectionController:
    def __init__(self, view, translator, data_model, plugins):
        self.view = view
        self.translator = translator
        self.data_model = data_model
        self.feature_selection_service = FeatureSelectionService(plugins)
        self.feature_selection_model = FeatureSelectionModel()
        self.logger = setup_logger()
        

        self.label_processor = EnhancedLabelProcessor()
        print("🔧 FeatureSelectionController initialized with EnhancedLabelProcessor")

    def apply_feature_selection(self, method, params):
        try:
            if not self.data_model.has_partitioned_data():
                raise ValueError("No partitioned data available")

            partitioned_data = self.data_model.get_partitioned_data()
            X_train = partitioned_data['X_train']
            y_train = partitioned_data['y_train']
            X_test = partitioned_data['X_test']
            y_test = partitioned_data['y_test']

            result = self.feature_selection_service.apply_feature_selection(
                method=method,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=params
            )

            self.feature_selection_model.set_selected_method(method)
            self.feature_selection_model.set_selected_features(result['selected_features'])
            self.feature_selection_model.set_feature_importance(result.get('feature_importance'))

            return result

        except Exception as e:
            self.view.show_error(f"Feature selection failed: {str(e)}")
            return None

    def get_selected_features(self):
        return self.feature_selection_model.selected_features

    def get_feature_importance(self):
        return self.feature_selection_model.feature_importance

    def apply_method(self, X_train, y_train, X_test, method, params):
        """A more direct method to apply feature selection without view interaction."""
        try:
            self.logger.info(f"Applying feature selection method: {method} with parameters: {params}")
            
            result = self.feature_selection_service.apply_method(
                X_train, y_train, X_test, method, params
            )
            
            self.logger.info("Feature selection service call completed.")
            return result
        except Exception as e:
            self.logger.error(f"Error in feature_selection_service.apply_method: {e}", exc_info=True)
            raise

    def apply_feature_selection_with_dialog(self):
        """DEPRECATED: Original method that gets parameters from a dialog."""
        try:
            method = self.view.get_selected_method()
            if method is None:
                self.view.display_message("Please select a feature selection method first", "Error")
                return
            
            try:
                params = self.view.get_parameters(method)
            except TypeError:
                params = self.view.get_parameters()
            
            if self.data_model.X_train is None or self.data_model.y_train is None:
                self.view.display_message("Training data is not available.", "Error")
                return
            
            X_train = self.data_model.X_train.copy()
            y_train = self.data_model.y_train.copy()
            X_test = self.data_model.X_test.copy() if self.data_model.X_test is not None else None
            
            self.logger.info(f"Applying feature selection method: {method} with parameters: {params}")
            
            feature_selection_result = self.feature_selection_service.apply_method(
                X_train, y_train, X_test, method, params
            )
            
            if not feature_selection_result or 'selected_features' not in feature_selection_result:
                self.view.display_message("No features were selected. Please try a different method or parameters.", "Warning")
                return
            
            selected_features = feature_selection_result.get('selected_features')
            if not selected_features or len(selected_features) == 0:
                self.view.display_message("No features were selected. Please try a different method or parameters.", "Warning")
                return
            
            self.data_model.X_train_selected = feature_selection_result.get('X_train_selected')
            self.data_model.X_test_selected = feature_selection_result.get('X_test_selected')
            self.data_model.selected_features = selected_features
            
            self.feature_selection_model.set_selected_method(method)
            self.feature_selection_model.set_selected_features(selected_features)
            if 'feature_importance' in feature_selection_result:
                self.feature_selection_model.set_feature_importance(feature_selection_result.get('feature_importance'))
            
            vis_window = VisualizationWindow(title="Feature Selection Result")
            vis_window.plot_feature_importance(
                feature_selection_result.get('feature_importance', None), 
                selected_features
            )
            vis_window.show()
            
            self.logger.info(f"Applied feature selection method: {method}, selected {len(selected_features)} features")
            self.view.display_message(f"Feature selection applied successfully using {method}. Selected {len(selected_features)} features.", "Information")
        except FeatureSelectionError as e:
            error_message = str(e)
            self.view.display_message(f"Feature selection error: {error_message}", "Error")
            self.logger.error(f"Feature selection error: {error_message}")
        except Exception as e:
            import traceback
            error_message = f"Unexpected error in feature selection: {str(e)}"
            self.logger.error(f"{error_message}\n{traceback.format_exc()}")
            self.view.display_message(error_message, "Error")
