# utils/exceptions.py

class DataServiceError(Exception):
    """Exception raised for errors in the data service."""
    def __init__(self, message="An error occurred in the Data Service."):
        self.message = message
        super().__init__(self.message)


class PreprocessingError(Exception):
    """Exception raised for errors during preprocessing."""
    def __init__(self, message="An error occurred during preprocessing."):
        self.message = message
        super().__init__(self.message)


class FeatureSelectionError(Exception):
    """Exception raised for errors during feature selection."""
    def __init__(self, message="An error occurred during feature selection."):
        self.message = message
        super().__init__(self.message)


class ModelingError(Exception):
    """Exception raised for errors during modeling."""
    def __init__(self, message="An error occurred during modeling."):
        self.message = message
        super().__init__(self.message)


class HyperparameterOptimizationError(Exception):
    """Exception raised for errors during hyperparameter optimization."""
    def __init__(self, message="An error occurred during hyperparameter optimization."):
        self.message = message
        super().__init__(self.message)


class EvaluationError(Exception):
    """Exception raised for errors during evaluation."""
    def __init__(self, message="An error occurred during evaluation."):
        self.message = message
        super().__init__(self.message)
