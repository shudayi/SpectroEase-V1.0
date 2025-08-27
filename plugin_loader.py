#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plugin Loader for SpectroEase
Dynamically loads plugins from the plugins directory
"""

import os
import sys
import importlib
import inspect
from typing import Dict, Tuple, Any

def load_plugins(plugin_base_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load plugins from the specified directory structure.
    
    Args:
        plugin_base_dir: Base directory containing plugin subdirectories
        
    Returns:
        Tuple of (preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins)
    """
    
    preprocessing_plugins = {}
    feature_selection_plugins = {}
    modeling_plugins = {}
    data_partitioning_plugins = {}
    
    # Define plugin subdirectories
    plugin_dirs = {
        'preprocessing': preprocessing_plugins,
        'feature_selection': feature_selection_plugins,
        'modeling': modeling_plugins,
        'data_partitioning': data_partitioning_plugins
    }
    
    # Check if base directory exists
    if not os.path.exists(plugin_base_dir):
        print(f"Warning: Plugin directory {plugin_base_dir} not found. Using empty plugin collections.")
        return preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins
    
    # Load plugins from each subdirectory
    for subdir_name, plugin_dict in plugin_dirs.items():
        subdir_path = os.path.join(plugin_base_dir, subdir_name)
        
        if not os.path.exists(subdir_path):
            print(f"Warning: Plugin subdirectory {subdir_path} not found. Skipping.")
            continue
            
        # Add the plugin directory to Python path temporarily
        if subdir_path not in sys.path:
            sys.path.insert(0, subdir_path)
        
        try:
            # Load all Python files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]  # Remove .py extension
                    
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Look for classes that might be plugins
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # Skip imported classes
                            if obj.__module__ != module_name:
                                continue
                                
                            # Check if it's a plugin class (has specific interfaces)
                            if _is_plugin_class(obj, subdir_name):
                                plugin_dict[name] = obj
                                print(f"Loaded {subdir_name} plugin: {name}")
                    
                    except Exception as e:
                        print(f"Warning: Failed to load plugin {filename} from {subdir_name}: {e}")
                        continue
        
        except Exception as e:
            print(f"Warning: Failed to scan plugin directory {subdir_path}: {e}")
            continue
        
        finally:
            # Remove the plugin directory from Python path
            if subdir_path in sys.path:
                sys.path.remove(subdir_path)
    
    print(f"Plugin loading summary:")
    print(f"  - Preprocessing: {len(preprocessing_plugins)} plugins")
    print(f"  - Feature Selection: {len(feature_selection_plugins)} plugins") 
    print(f"  - Modeling: {len(modeling_plugins)} plugins")
    print(f"  - Data Partitioning: {len(data_partitioning_plugins)} plugins")
    
    return preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins


def _is_plugin_class(cls, plugin_type: str) -> bool:
    """
    Check if a class is a valid plugin for the given type.
    
    Args:
        cls: The class to check
        plugin_type: The type of plugin (preprocessing, feature_selection, modeling, data_partitioning)
        
    Returns:
        True if the class appears to be a valid plugin
    """
    
    # Basic checks
    if not hasattr(cls, '__init__'):
        return False
    
    # Check for expected methods based on plugin type
    expected_methods = {
        'preprocessing': ['process', 'apply'],
        'feature_selection': ['select_features', 'fit', 'transform'],
        'modeling': ['train', 'predict', 'fit'],
        'data_partitioning': ['split', 'partition']
    }
    
    if plugin_type in expected_methods:
        # Check if class has at least one of the expected methods
        for method_name in expected_methods[plugin_type]:
            if hasattr(cls, method_name):
                return True
    
    # If no specific methods found, check for common plugin patterns
    method_names = [method for method in dir(cls) if not method.startswith('_')]
    
    # Look for methods that suggest this is a plugin
    plugin_indicators = ['process', 'apply', 'run', 'execute', 'transform', 'fit', 'predict']
    for indicator in plugin_indicators:
        if any(indicator in method.lower() for method in method_names):
            return True
    
    return False


if __name__ == "__main__":
    # Test the plugin loader
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(current_dir, 'plugins')
    
    preprocessing, feature_selection, modeling, data_partitioning = load_plugins(plugin_dir)
    
    print("\nPlugin loading test completed.")
