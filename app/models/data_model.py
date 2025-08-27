# app/models/data_model.py

import pandas as pd
import numpy as np
from app.utils.label_processor import EnhancedLabelProcessor

class DataModel:
    def __init__(self):
        self.data = None
        self.partitioned_data = None
        self.preprocessed_data = None
        self.file_path = None
        self.label_processor = EnhancedLabelProcessor()

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def has_data(self):
        return self.data is not None

    def set_partitioned_data(self, partitioned_data):
        self.partitioned_data = partitioned_data

    def get_partitioned_data(self):
        return self.partitioned_data

    def has_partitioned_data(self):
        return self.partitioned_data is not None

    def set_preprocessed_data(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def get_preprocessed_data(self):
        return self.preprocessed_data

    def has_preprocessed_data(self):
        return self.preprocessed_data is not None

    def get_X_train(self):
        if self.partitioned_data:
            return self.partitioned_data.get('X_train')
        return None

    def get_y_train(self):
        if self.partitioned_data:
            return self.partitioned_data.get('y_train')
        return None

    def get_X_test(self):
        if self.partitioned_data:
            return self.partitioned_data.get('X_test')
        return None

    def get_y_test(self):
        if self.partitioned_data:
            return self.partitioned_data.get('y_test')
        return None

    def clear_data(self):
        self.data = None
        self.partitioned_data = None
        self.preprocessed_data = None
        self.file_path = None