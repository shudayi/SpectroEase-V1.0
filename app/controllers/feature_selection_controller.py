from app.utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
from app.utils.logger import setup_logger
from app.utils.exceptions import FeatureSelectionError
import pandas as pd
from app.services.feature_selection_service import FeatureSelectionService
from app.models.feature_selection_model import FeatureSelectionModel
from app.utils.label_processor import EnhancedLabelProcessor

class FeatureSelectionController:
    def __init__(self, view, translator, data_model, plugins):
        self.view = view
        self.translator = translator
        self.data_model = data_model
        self.feature_selection_service = FeatureSelectionService(plugins)
        # V1.4.0: Let service access view for dynamically added plugins
        self.feature_selection_service.view = view
        self.feature_selection_model = FeatureSelectionModel()
        self.logger = setup_logger()
        

        self.label_processor = EnhancedLabelProcessor()
        print("🔧 FeatureSelectionController initialized with EnhancedLabelProcessor")

    def apply_feature_selection(self, method, params):
        try:
            # --- Unsupervised PCA Branch ---
            if method == "Unsupervised PCA":
                plugin = self.feature_selection_service.plugins.get(method)
                if not plugin:
                    raise FeatureSelectionError(f"Plugin '{method}' not found.")

                if self.data_model.has_partitioned_data():
                    # --- Modeling Mode ---
                    self.logger.info("Applying PCA in Modeling Mode on partitioned data.")
                    partitioned_data = self.data_model.get_partitioned_data()
                    X_train, y_train, X_test = partitioned_data['X_train'], partitioned_data['y_train'], partitioned_data['X_test']

                    X_train_transformed, _ = plugin.select_features(X=X_train, y=y_train, params=params)
                    
                    pca_model = getattr(X_train_transformed, '_pca_object', None)
                    scaler = getattr(X_train_transformed, '_scaler_object', None)
                    if not pca_model or not scaler:
                        raise FeatureSelectionError("The PCA plugin did not return a fitted model. Cannot transform test set.")

                    X_test_scaled = scaler.transform(X_test)
                    X_test_transformed_raw = pca_model.transform(X_test_scaled)
                    X_test_transformed = pd.DataFrame(X_test_transformed_raw, index=X_test.index, columns=X_train_transformed.columns)

                    self.data_model.X_train_selected = X_train_transformed
                    self.data_model.X_test_selected = X_test_transformed
                    selected_features = X_train_transformed.columns.tolist()
                    self.data_model.selected_features = selected_features
                    
                    self.feature_selection_model.set_selected_method(method)
                    self.feature_selection_model.set_selected_features(selected_features)
                    self.feature_selection_model.set_feature_importance(None)

                    self.view.display_message(f"PCA applied for modeling. {len(selected_features)} components selected.", "Success")
                    return {'selected_features': selected_features, 'feature_importance': None}

                else:
                    # --- Exploratory Mode ---
                    self.logger.info("Applying PCA in Exploratory Mode on full dataset.")
                    X = self.data_model.get_X()
                    if X is None:
                        raise ValueError("Data (X) is not loaded.")
                    
                    # Call the plugin with exploratory_mode=True
                    plugin.select_features(X=X, y=None, params=params, exploratory_mode=True)
                    
                    # The plugin now handles the dialog display. 
                    # The controller's job is just to call it correctly.
                    self.view.display_message("PCA exploratory analysis complete.", "Success")
                    return None
            
            # --- Supervised Algorithms Branch (Original Logic) ---
            else:
                self.logger.info(f"Applying supervised method: {method}")
                if not self.data_model.has_partitioned_data():
                    raise ValueError(f"Method '{method}' requires partitioned data. Please partition data first.")

                partitioned_data = self.data_model.get_partitioned_data()
                X_train, y_train, X_test, y_test = partitioned_data['X_train'], partitioned_data['y_train'], partitioned_data['X_test'], partitioned_data['y_test']

                result = self.feature_selection_service.apply_feature_selection(
                    method=method, X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test, params=params
                )

                self.feature_selection_model.set_selected_method(method)
                self.feature_selection_model.set_selected_features(result['selected_features'])
                self.feature_selection_model.set_feature_importance(result.get('feature_importance'))

                return result

        except Exception as e:
            self.view.display_error(f"Feature selection failed: {str(e)}")
            self.logger.error(f"Feature selection failed: {str(e)}", exc_info=True)
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
            
            # Use correct method to get training data
            X_train = self.data_model.get_X_train()
            y_train = self.data_model.get_y_train()
            X_test = self.data_model.get_X_test()
            
            if X_train is None or y_train is None:
                self.view.display_message("Training data is not available. Please perform data partitioning first.", "Error")
                return
            
            # Create copies to avoid modifying original data
            X_train = X_train.copy()
            y_train = y_train.copy()
            X_test = X_test.copy() if X_test is not None else None
            
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
