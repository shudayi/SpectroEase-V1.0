# app/models/preprocessing_model.py

class PreprocessingModel:
    def __init__(self):
        self.preprocessed_data = None
        self.applied_methods = {}

    def add_applied_method(self, method_name, params):
        self.applied_methods[method_name] = params
