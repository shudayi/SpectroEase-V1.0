class DataPartitioningView(QWidget):
    def __init__(self, plugins: dict):
        super().__init__()
        self.plugins = plugins
        self.method_list = QListWidget()
  

    def add_custom_algorithm(self, code):
        """Add custom data partitioning algorithm V1.3.2: Improved class recognition logic"""
        try:
            import types
            import inspect
            from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
            import pandas as pd
            import numpy as np
            
            mod = types.ModuleType('custom_splitter')
            
            # V1.3.2: Provide necessary imports
            mod.__dict__.update({
                'pd': pd,
                'np': np,
                'pandas': pd,
                'numpy': np,
                'DataPartitioningAlgorithm': DataPartitioningAlgorithm,
                'Dict': __import__('typing').Dict,
                'Tuple': __import__('typing').Tuple,
                'Any': __import__('typing').Any,
                'List': __import__('typing').List,
            })
            
            # V1.3.7: Add sklearn support (if available)
            try:
                from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
                mod.__dict__.update({
                    'train_test_split': train_test_split,
                    'StratifiedShuffleSplit': StratifiedShuffleSplit,
                })
            except ImportError:
                pass
            
            # Record class list before execution
            classes_before = set(item for item in mod.__dict__.values() if isinstance(item, type))
            
            exec(code, mod.__dict__)
            
            # Record class list after execution, find newly added classes
            classes_after = set(item for item in mod.__dict__.values() if isinstance(item, type))
            new_classes = classes_after - classes_before
            
            algorithm_found = False
            for item in new_classes:
                # V1.3.2: Must be subclass and not abstract class
                if (issubclass(item, DataPartitioningAlgorithm) and 
                    not inspect.isabstract(item)):
                    algorithm = item()
                    self.plugins[algorithm.get_name()] = algorithm
                    self.method_list.addItem(algorithm.get_name())
                    algorithm_found = True
                    break
            
            if not algorithm_found:
                raise Exception("No valid algorithm class found in code")
                    
        except Exception as e:
            raise Exception(f"Error loading custom splitter: {str(e)}") 