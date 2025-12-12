# app/models/feature_selection_model.py

class FeatureSelectionModel:
    def __init__(self):
        self.selected_features = []
        self.selected_method = None
        self.feature_importance = None
        
    def set_selected_method(self, method):
        self.selected_method = method
        
    def set_selected_features(self, features):
        self.selected_features = features
        
    def set_feature_importance(self, importance):
        self.feature_importance = importance
