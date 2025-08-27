# interfaces/preprocessing_algorithm.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class PreprocessingAlgorithm(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm"""
        pass

    @abstractmethod
    def get_params_info(self) -> Dict[str, Any]:
        """Return parameter information required by the algorithm, used to generate parameter setting interface"""
        pass

    @abstractmethod
    def apply(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply preprocessing algorithm to data"""
        pass
