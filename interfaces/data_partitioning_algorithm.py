# interfaces/data_partitioning_algorithm.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPartitioningAlgorithm(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm"""
        pass

    @abstractmethod
    def get_params_info(self) -> Dict[str, Any]:
        """Return parameter information required by the algorithm, used to generate parameter setting interface"""
        pass

    @abstractmethod
    def partition(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Partition data into training and testing sets
        
        Args:
            data: Complete dataset
            params: Partitioning parameters
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        pass
