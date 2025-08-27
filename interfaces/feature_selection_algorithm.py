# interfaces/feature_selection_algorithm.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class FeatureSelectionAlgorithm(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm"""
        pass

    @abstractmethod
    def get_params_info(self) -> Dict[str, Any]:
        """Return parameter information required by the algorithm, used to generate parameter setting interface"""
        pass

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> List[str]:
        """Select features and return list of selected feature names"""
        pass
