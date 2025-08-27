# app/models/modeling_model.py

class ModelingModel:
    def __init__(self):
        self.model = None
        self.trained_model = None
        self.parameters = {}
        self.method = ''
        self.X_train = None
        self.y_train = None
        self.evaluation_results = {}
        self.analysis_type = ''

    def set_model(self, model):
        self.model = model

    def set_parameters(self, params: dict):
        self.parameters = params

    def set_method(self, method: str):
        self.method = method

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_evaluation_results(self, results: dict):
        """Store model evaluation results"""
        self.evaluation_results = results

    def set_analysis_type(self, analysis_type: str):
        """Set the analysis type ('qualitative' or 'quantitative')"""
        self.analysis_type = analysis_type
