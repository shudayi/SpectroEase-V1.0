"""
Custom Algorithm Cache Service
Manages persistent storage of user-defined custom algorithms
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import shutil


class CustomAlgorithmCache:
    """Service for caching and managing custom algorithms"""
    
    def __init__(self, cache_file_path: str = None):
        """
        Initialize the cache service
        
        Args:
            cache_file_path: Path to the cache JSON file. If None, uses default location.
        """
        if cache_file_path is None:
            # Default location in config directory
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            os.makedirs(config_dir, exist_ok=True)
            cache_file_path = os.path.join(config_dir, 'custom_algorithms.json')
        
        self.cache_file_path = cache_file_path
        self.backup_file_path = cache_file_path + '.backup'
        
        # Initialize cache structure if file doesn't exist
        if not os.path.exists(self.cache_file_path):
            self._initialize_cache()
        
        # Validate and load cache
        self._validate_and_load()
    
    def _initialize_cache(self):
        """Initialize an empty cache file"""
        initial_data = {
            "preprocessing": [],
            "feature_selection": [],
            "modeling": [],
            "data_partitioning": []
        }
        self._save_cache_data(initial_data)
    
    def _validate_and_load(self):
        """Validate cache file and load data"""
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            required_keys = ["preprocessing", "feature_selection", "modeling", "data_partitioning"]
            for key in required_keys:
                if key not in data:
                    data[key] = []
            
            # If validation passed, save normalized data
            self._save_cache_data(data)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Cache file corrupted or unreadable: {e}")
            
            # Try to restore from backup
            if os.path.exists(self.backup_file_path):
                try:
                    print("Attempting to restore from backup...")
                    shutil.copy(self.backup_file_path, self.cache_file_path)
                    print("Successfully restored from backup")
                except Exception as restore_error:
                    print(f"Failed to restore from backup: {restore_error}")
                    self._initialize_cache()
            else:
                print("No backup found, initializing new cache")
                self._initialize_cache()
    
    def _load_cache_data(self) -> Dict:
        """Load cache data from file"""
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {
                "preprocessing": [],
                "feature_selection": [],
                "modeling": [],
                "data_partitioning": []
            }
    
    def _save_cache_data(self, data: Dict):
        """Save cache data to file with backup"""
        try:
            # Create backup of existing file
            if os.path.exists(self.cache_file_path):
                shutil.copy(self.cache_file_path, self.backup_file_path)
            
            # Save new data
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            raise
    
    def save_algorithm(self, algorithm_type: str, name: str, code: str) -> bool:
        """
        Save a custom algorithm to cache
        
        Args:
            algorithm_type: Type of algorithm (preprocessing, feature_selection, modeling, data_partitioning)
            name: Unique name of the algorithm
            code: Python code of the algorithm
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = self._load_cache_data()
            
            # Validate algorithm type
            if algorithm_type not in data:
                print(f"Error: Invalid algorithm type '{algorithm_type}'")
                return False
            
            # Check if algorithm with same name already exists
            algorithms = data[algorithm_type]
            existing_index = None
            for i, algo in enumerate(algorithms):
                if algo['name'] == name:
                    existing_index = i
                    break
            
            # Create algorithm entry
            timestamp = datetime.now().isoformat()
            algorithm_entry = {
                "name": name,
                "code": code,
                "created_at": timestamp if existing_index is None else algorithms[existing_index].get('created_at', timestamp),
                "updated_at": timestamp
            }
            
            # Add or update algorithm
            if existing_index is not None:
                algorithms[existing_index] = algorithm_entry
                print(f"Updated existing algorithm '{name}' in {algorithm_type}")
            else:
                algorithms.append(algorithm_entry)
                print(f"Added new algorithm '{name}' to {algorithm_type}")
            
            # Save to file
            self._save_cache_data(data)
            return True
            
        except Exception as e:
            print(f"Error saving algorithm: {e}")
            return False
    
    def load_algorithms(self, algorithm_type: str) -> List[Dict]:
        """
        Load all algorithms of a specific type
        
        Args:
            algorithm_type: Type of algorithm to load
            
        Returns:
            List of algorithm dictionaries
        """
        try:
            data = self._load_cache_data()
            
            if algorithm_type not in data:
                print(f"Warning: Invalid algorithm type '{algorithm_type}'")
                return []
            
            return data[algorithm_type]
            
        except Exception as e:
            print(f"Error loading algorithms: {e}")
            return []
    
    def delete_algorithm(self, algorithm_type: str, name: str) -> bool:
        """
        Delete a specific algorithm
        
        Args:
            algorithm_type: Type of algorithm
            name: Name of the algorithm to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            data = self._load_cache_data()
            
            if algorithm_type not in data:
                print(f"Error: Invalid algorithm type '{algorithm_type}'")
                return False
            
            # Find and remove algorithm
            algorithms = data[algorithm_type]
            original_length = len(algorithms)
            data[algorithm_type] = [algo for algo in algorithms if algo['name'] != name]
            
            if len(data[algorithm_type]) < original_length:
                # Save updated data (this will create a new backup automatically)
                self._save_cache_data(data)
                # V1.4.4: Also ensure backup file is updated to prevent restoration of deleted algorithms
                if os.path.exists(self.backup_file_path):
                    try:
                        # Update backup file to match current state
                        with open(self.backup_file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"✅ Updated backup file to match deletion")
                    except Exception as backup_error:
                        print(f"⚠️  Warning: Failed to update backup file: {backup_error}")
                
                print(f"Deleted algorithm '{name}' from {algorithm_type}")
                return True
            else:
                print(f"Algorithm '{name}' not found in {algorithm_type}")
                return False
                
        except Exception as e:
            print(f"Error deleting algorithm: {e}")
            return False
    
    def list_all_algorithms(self) -> Dict[str, List[Dict]]:
        """
        Get all algorithms grouped by type
        
        Returns:
            Dictionary mapping algorithm types to lists of algorithms
        """
        try:
            return self._load_cache_data()
        except Exception as e:
            print(f"Error listing algorithms: {e}")
            return {
                "preprocessing": [],
                "feature_selection": [],
                "modeling": [],
                "data_partitioning": []
            }
    
    def algorithm_exists(self, algorithm_type: str, name: str) -> bool:
        """
        Check if an algorithm with the given name exists
        
        Args:
            algorithm_type: Type of algorithm
            name: Name of the algorithm
            
        Returns:
            True if algorithm exists, False otherwise
        """
        try:
            algorithms = self.load_algorithms(algorithm_type)
            return any(algo['name'] == name for algo in algorithms)
        except Exception as e:
            print(f"Error checking algorithm existence: {e}")
            return False
    
    def get_algorithm_count(self, algorithm_type: str = None) -> int:
        """
        Get count of algorithms
        
        Args:
            algorithm_type: If specified, count only this type. Otherwise count all.
            
        Returns:
            Number of algorithms
        """
        try:
            if algorithm_type:
                return len(self.load_algorithms(algorithm_type))
            else:
                data = self._load_cache_data()
                return sum(len(algos) for algos in data.values())
        except Exception as e:
            print(f"Error counting algorithms: {e}")
            return 0
    
    def clear_all_algorithms(self, algorithm_type: str = None) -> bool:
        """
        Clear algorithms (dangerous operation)
        
        Args:
            algorithm_type: If specified, clear only this type. Otherwise clear all.
            
        Returns:
            True if successful
        """
        try:
            data = self._load_cache_data()
            
            if algorithm_type:
                if algorithm_type in data:
                    data[algorithm_type] = []
            else:
                for key in data:
                    data[key] = []
            
            self._save_cache_data(data)
            print(f"Cleared algorithms: {algorithm_type if algorithm_type else 'all types'}")
            return True
            
        except Exception as e:
            print(f"Error clearing algorithms: {e}")
            return False





