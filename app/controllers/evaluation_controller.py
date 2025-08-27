from app.utils.logger import setup_logger
from app.utils.exceptions import EvaluationError
from app.services.evaluation_service import EvaluationService
# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging

class EvaluationController:
    def __init__(self, view, translator, data_model=None, modeling_model=None):
        self.view = view
        self.translator = translator
        self.data_model = data_model
        self.modeling_model = modeling_model
        self.evaluation_service = EvaluationService()
        self.logger = setup_logger()

    def evaluate_model(self):
        metrics = self.view.get_selected_metrics()
        
        if self.modeling_model.trained_model is None:
            self.view.display_message("No trained model available for evaluation.", "Error")
            return
            
        if self.data_model.X_test is None or self.data_model.y_test is None:
            self.view.display_message("Test data is not available.", "Error")
            return
            
        X_test = self.data_model.X_test.copy()
        y_test = self.data_model.y_test.copy()
        
        try:
            results = self.evaluation_service.evaluate_model(
                self.modeling_model.trained_model, 
                X_test, 
                y_test, 
                metrics
            )
            
            self.view.update_evaluation_results(results)
            self.view.display_message("Model evaluation completed successfully.")
        except EvaluationError as e:
            self.view.display_message(str(e), "Error")
            self.logger.error(e)
