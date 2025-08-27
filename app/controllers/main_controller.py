from app.controllers.data_controller import DataController
from app.controllers.preprocessing_controller import PreprocessingController
from app.controllers.feature_selection_controller import FeatureSelectionController
from app.controllers.modeling_controller import ModelingController
from app.controllers.evaluation_controller import EvaluationController
from app.controllers.hyperparameter_optimization_controller import HyperparameterOptimizationController
from plugin_loader import load_plugins
import os
import pandas as pd

class MainController:
    def __init__(self, view, translator):
        self.view = view
        self.translator = translator
        
        self.data_model = None
        self.modeling_model = None
        
        self.current_data = None
        self.train_data = None
        self.test_data = None
        self.preprocessed_train = None
        self.preprocessed_test = None
        self.feature_selected_train = None
        self.feature_selected_test = None
        self.trained_model = None
        
        self.preprocessing_plugins = view.preprocessing_plugins
        self.feature_selection_plugins = view.feature_selection_plugins
        self.modeling_plugins = view.modeling_plugins
        self.data_partitioning_plugins = view.data_partitioning_plugins
        
        self.initialize_controllers()
        
    def initialize_controllers(self):
        self.data_controller = DataController(
            self.view, 
            self.translator,
            self.data_partitioning_plugins
        )
        
        self.preprocessing_controller = PreprocessingController(
            view=self.view.preprocessing_view, 
            translator=self.translator,
            plugins=self.preprocessing_plugins,
            data_model=self.data_controller.data_model
        )
        
        self.feature_selection_controller = FeatureSelectionController(
            view=self.view.feature_selection_view, 
            translator=self.translator,
            plugins=self.feature_selection_plugins,
            data_model=self.data_controller.data_model
        )
        
        self.modeling_controller = ModelingController(
            view=self.view.modeling_view, 
            translator=self.translator,
            plugins=self.modeling_plugins,
            data_model=self.data_controller.data_model
        )
        
        self.hyperopt_controller = HyperparameterOptimizationController(
            view=self.view.hyperopt_view, 
            translator=self.translator,
            modeling_model=self.modeling_controller.modeling_model
        )
        
        self.evaluation_controller = EvaluationController(
            view=self.view.evaluation_view, 
            translator=self.translator,
            data_model=self.data_controller.data_model,
            modeling_model=self.modeling_controller.modeling_model
        )

    def handle_load_data(self):
        try:
            data = self.data_controller.load_data()
            if data is not None:
                self.data_model = self.data_controller.data_model
                self.update_data_flow(data, 'load')
        except Exception as e:
            self.view.display_message(f"Error loading data: {str(e)}", "Error")
            
    def check_data_ready(self):
        if self.current_data is None:
            if self.view.spectra_data is not None:
                self.current_data = pd.DataFrame(self.view.spectra_data)
                if hasattr(self, 'data_controller') and hasattr(self.data_controller, 'data_model'):
                    self.data_controller.data_model.data = self.current_data
                return True
            else:
                self.view.display_message("Please load data first", "Warning")
                return False
        return True
        
    def check_data_split(self):
        if self.train_data is None or self.test_data is None:
            self.view.display_message("Please split data first", "Warning")
            return False
        return True
        
    def check_model_trained(self):
        if self.trained_model is None:
            self.view.display_message("Please train model first", "Warning")
            return False
        return True

    def update_data_flow(self, data, step):
        if step == 'load':
            self.current_data = data
            self.view.update_data_view(data)
            
        elif step == 'split':
            self.train_data, self.test_data = data
            self.view.update_split_view(self.train_data, self.test_data)
            
        elif step == 'preprocess':
            self.preprocessed_train = data[0]
            self.preprocessed_test = data[1]
            
        elif step == 'feature_select':
            self.feature_selected_train = data[0]
            self.feature_selected_test = data[1]
            
        elif step == 'train':
            self.trained_model = data
