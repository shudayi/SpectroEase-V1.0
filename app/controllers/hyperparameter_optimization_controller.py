# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
from app.utils.logger import setup_logger
from app.utils.exceptions import HyperparameterOptimizationError
from app.services.hyperparameter_optimization_service import HyperparameterOptimizationService
from app.models.hyperparameter_optimization_model import HyperparameterOptimizationModel
from app.utils.label_processor import EnhancedLabelProcessor

class HyperparameterOptimizationController:
    def __init__(self, view, translator, modeling_model=None):
        self.view = view
        self.logger = setup_logger()
        self.translator = translator
        self.modeling_model = modeling_model
        self.hyperopt_service = HyperparameterOptimizationService()
        self.hyperopt_model = HyperparameterOptimizationModel()
        self.label_processor = EnhancedLabelProcessor()
        print("🔧 HyperparameterOptimizationController initialized with EnhancedLabelProcessor")

    def apply_optimization(self):
        method = self.view.get_selected_method()
        params = self.view.get_parameters()
        
        if self.modeling_model.X_train is None or self.modeling_model.y_train is None:
            self.view.display_message("Training data is not available.", "Error")
            return
            
        if self.modeling_model.model is None:
            self.view.display_message("No trained model available for optimization.", "Error")
            return
            
        model_type = self.modeling_model.method
        X_train = self.modeling_model.X_train.copy()
        y_train = self.modeling_model.y_train.copy()
        
        try:
            param_grid = params.get('param_grid', {})
            scoring = params.get('scoring', 'accuracy')
            cv = params.get('cv', 5)
            
            best_params, best_model, cv_results = self.hyperopt_service.optimize(
                model_type, 
                self.modeling_model.model, 
                X_train, 
                y_train, 
                method, 
                param_grid, 
                scoring, 
                cv
            )
            
            self.modeling_model.model = best_model
            self.modeling_model.parameters.update(best_params)
            
            self.view.update_optimization_results(best_params, cv_results)
            self.view.display_message("Hyperparameter optimization completed successfully.")
            self.logger.info(f"Applied hyperparameter optimization method: {method} with parameters: {params}")
        except HyperparameterOptimizationError as e:
            self.view.display_message(str(e), "Error")
            self.logger.error(e)
