# app/utils/dynamic_loader.py
"""
Dynamic algorithm loader - Works in both development and EXE environments
"""

import os
import sys
import types
import importlib.util
import tempfile
from pathlib import Path


class DynamicAlgorithmLoader:
    """
    Load algorithm code dynamically with EXE compatibility
    """
    
    def __init__(self):
        # Create directory for custom algorithms (relative to exe or script)
        if getattr(sys, 'frozen', False):
            # Running in a bundle (EXE)
            base_dir = Path(sys.executable).parent
        else:
            # Running in normal Python environment
            base_dir = Path.cwd()
        
        self.custom_algo_dir = base_dir / 'custom_algorithms'
        self.custom_algo_dir.mkdir(exist_ok=True)
        
        # Add to Python path if not already there
        if str(self.custom_algo_dir) not in sys.path:
            sys.path.insert(0, str(self.custom_algo_dir))
    
    def load_from_code(self, code: str, algorithm_name: str = None):
        """
        Load algorithm from code string
        
        Args:
            code: Algorithm code as string
            algorithm_name: Optional name for the algorithm file
            
        Returns:
            Algorithm class instance or None
        """
        try:
            # Method 1: Try direct exec (fast, works in most cases)
            return self._load_via_exec(code)
        except Exception as e:
            print(f"⚠️  Direct exec failed: {e}")
            print("📝 Trying file-based loading for EXE compatibility...")
            
            try:
                # Method 2: Save to file and import (EXE-compatible)
                return self._load_via_file(code, algorithm_name)
            except Exception as e2:
                print(f"❌ File-based loading also failed: {e2}")
                raise Exception(f"Failed to load algorithm: {e2}")
    
    def _load_via_exec(self, code: str):
        """
        Load algorithm using exec() - Fast but may fail in EXE
        V1.3.1: Added imports for all interface classes
        """
        mod = types.ModuleType('dynamic_algorithm')
        
        # V1.3.1: Import all required interface classes
        try:
            from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
            from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
            from interfaces.modeling_algorithm import ModelingAlgorithm
            from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
        except ImportError as e:
            print(f"⚠️  Warning: Could not import interface classes: {e}")
            PreprocessingAlgorithm = None
            FeatureSelectionAlgorithm = None
            ModelingAlgorithm = None
            DataPartitioningAlgorithm = None
        
        # Provide necessary imports in the module namespace
        mod.__dict__.update({
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'pandas': __import__('pandas'),
            'numpy': __import__('numpy'),
            # V1.3.1: Add interface classes
            'PreprocessingAlgorithm': PreprocessingAlgorithm,
            'FeatureSelectionAlgorithm': FeatureSelectionAlgorithm,
            'ModelingAlgorithm': ModelingAlgorithm,
            'DataPartitioningAlgorithm': DataPartitioningAlgorithm,
            # Other commonly used imports
            'Dict': __import__('typing').Dict,
            'List': __import__('typing').List,
            'Any': __import__('typing').Any,
            'Tuple': __import__('typing').Tuple,
        })
        
        exec(code, mod.__dict__)
        
        # Find and instantiate the algorithm class
        for item in mod.__dict__.values():
            if isinstance(item, type) and hasattr(item, 'get_name'):
                try:
                    return item()
                except Exception as e:
                    print(f"Failed to instantiate {item.__name__}: {e}")
                    continue
        
        raise Exception("No valid algorithm class found in code")
    
    def _load_via_file(self, code: str, algorithm_name: str = None):
        """
        Load algorithm by saving to file first - EXE compatible
        """
        # Generate a unique filename
        if algorithm_name is None:
            import re
            # Try to extract class name from code
            class_match = re.search(r'class\s+(\w+)', code)
            if class_match:
                algorithm_name = class_match.group(1)
            else:
                algorithm_name = "CustomAlgorithm"
        
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in algorithm_name)
        
        # Create unique filename with timestamp to avoid conflicts
        import time
        timestamp = int(time.time())
        filename = f"{safe_name}_{timestamp}.py"
        filepath = self.custom_algo_dir / filename
        
        # Write code to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"📁 Saved algorithm to: {filepath}")
        
        # Import the module
        spec = importlib.util.spec_from_file_location(safe_name, filepath)
        if spec is None or spec.loader is None:
            raise Exception(f"Failed to create module spec from {filepath}")
        
        mod = importlib.util.module_from_spec(spec)
        sys.modules[safe_name] = mod
        spec.loader.exec_module(mod)
        
        # Find and instantiate the algorithm class
        for item in mod.__dict__.values():
            if isinstance(item, type) and hasattr(item, 'get_name'):
                try:
                    instance = item()
                    print(f"✅ Successfully loaded: {instance.get_name()}")
                    return instance
                except Exception as e:
                    print(f"Failed to instantiate {item.__name__}: {e}")
                    continue
        
        raise Exception("No valid algorithm class found in file")
    
    def cleanup_old_files(self, keep_recent=10):
        """
        Clean up old algorithm files to prevent clutter
        """
        try:
            files = sorted(
                self.custom_algo_dir.glob("*.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent files
            for old_file in files[keep_recent:]:
                try:
                    old_file.unlink()
                    print(f"🗑️  Removed old algorithm file: {old_file.name}")
                except Exception as e:
                    print(f"Warning: Could not remove {old_file}: {e}")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")


# Global loader instance
_loader = None

def get_loader():
    """Get the global DynamicAlgorithmLoader instance"""
    global _loader
    if _loader is None:
        _loader = DynamicAlgorithmLoader()
    return _loader

