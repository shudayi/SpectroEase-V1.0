# plugins/preprocessing/custom_preprocessing.py

from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
import pandas as pd

class CustomPreprocessing(PreprocessingAlgorithm):
    def get_name(self):
        return "Demo1"

    def get_parameter_info(self) -> dict:
        return {
            'param1': {
                'type': 'int',
                'default': 5,
                'description': 'An example integer parameter'
            },
            'param2': {
                'type': 'float',
                'default': 0.1,
                'description': 'An example float parameter'
            }
        }

    def apply(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:

        processed_data = data.copy()

        processed_data['custom_column'] = processed_data.iloc[:, 0] * params['param1'] * params['param2']
        return processed_data
