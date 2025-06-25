import importlib
import os
import pkgutil
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
from interfaces.modeling_algorithm import ModelingAlgorithm
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm

def load_plugins(plugin_base_dir):
    preprocessing_plugins = {}
    feature_selection_plugins = {}
    modeling_plugins = {}
    data_partitioning_plugins = {}

    preprocessing_dir = os.path.join(plugin_base_dir, 'preprocessing')
    if os.path.exists(preprocessing_dir):
        for finder, name, ispkg in pkgutil.iter_modules([preprocessing_dir]):
            module = importlib.import_module(f'plugins.preprocessing.{name}')
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, PreprocessingAlgorithm) and cls is not PreprocessingAlgorithm:
                    instance = cls()
                    preprocessing_plugins[instance.get_name()] = instance

    feature_selection_dir = os.path.join(plugin_base_dir, 'feature_selection')
    if os.path.exists(feature_selection_dir):
        for finder, name, ispkg in pkgutil.iter_modules([feature_selection_dir]):
            module = importlib.import_module(f'plugins.feature_selection.{name}')
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, FeatureSelectionAlgorithm) and cls is not FeatureSelectionAlgorithm:
                    instance = cls()
                    feature_selection_plugins[instance.get_name()] = instance

    modeling_dir = os.path.join(plugin_base_dir, 'modeling')
    if os.path.exists(modeling_dir):
        for finder, name, ispkg in pkgutil.iter_modules([modeling_dir]):
            module = importlib.import_module(f'plugins.modeling.{name}')
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, ModelingAlgorithm) and cls is not ModelingAlgorithm:
                    instance = cls()
                    modeling_plugins[instance.get_name()] = instance

    data_partitioning_dir = os.path.join(plugin_base_dir, 'data_partitioning')
    if os.path.exists(data_partitioning_dir):
        for finder, name, ispkg in pkgutil.iter_modules([data_partitioning_dir]):
            module = importlib.import_module(f'plugins.data_partitioning.{name}')
            for attr in dir(module):
                cls = getattr(module, attr)
                if isinstance(cls, type) and issubclass(cls, DataPartitioningAlgorithm) and cls is not DataPartitioningAlgorithm:
                    instance = cls()
                    data_partitioning_plugins[instance.get_name()] = instance

    return preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins
