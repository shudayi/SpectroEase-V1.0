# app/models/hyperparameter_optimization_model.py

class HyperparameterOptimizationModel:
    def __init__(self):
        self.optimized_estimator = None
        self.method = ''
        self.parameters = {}

    def set_optimized_estimator(self, estimator):
        self.optimized_estimator = estimator

    def set_method(self, method: str):
        self.method = method

    def set_parameters(self, params: dict):
        self.parameters = params
