# interfaces/modeling_algorithm.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np

class ModelingAlgorithm(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm"""
        pass

    @abstractmethod
    def get_params_info(self) -> Dict[str, Any]:
        """Return parameter information required by the algorithm, used to generate parameter setting interface"""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> Any:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Use trained model for prediction"""
        pass
