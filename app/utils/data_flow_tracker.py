# -*- coding: utf-8 -*-
"""
Data flow tracker
Used to track and verify data changes throughout the data processing pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import time
from datetime import datetime


class DataFlowTracker:
    """Data flow tracker - Records and verifies data processing pipeline"""
    
    def __init__(self):
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.flow_sequence: List[str] = []
        self.comparisons: List[Dict[str, Any]] = []
        
    def snapshot(self, data: Any, stage_name: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Record data snapshot
        
        Args:
            data: Data (DataFrame or ndarray)
            stage_name: Stage name (e.g., "input", "after_baseline_correction")
            metadata: Additional metadata (e.g., algorithm parameters)
        
        Returns:
            Snapshot information dictionary
        """
        # Convert to numpy array for processing
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            has_columns = True
            columns = data.columns.tolist()
            index = data.index.tolist()
        else:
            data_array = np.array(data)
            has_columns = False
            columns = None
            index = None
        
        # Calculate data summary
        snapshot_info = {
            'stage_name': stage_name,
            'timestamp': datetime.now().isoformat(),
            'shape': data_array.shape,
            'dtype': str(data_array.dtype),
            'has_columns': has_columns,
            'columns': columns,
            'index': index,
            
            # Data statistics
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'median': float(np.median(data_array)),
            
            # Data quality
            'has_nan': bool(np.any(np.isnan(data_array))),
            'has_inf': bool(np.any(np.isinf(data_array))),
            'nan_count': int(np.sum(np.isnan(data_array))),
            'inf_count': int(np.sum(np.isinf(data_array))),
            
            # Data summary (first 3 samples, first 5 features)
            'data_preview': self._get_data_preview(data_array),
            
            # Data hash (for detecting changes)
            'data_hash': self._compute_hash(data_array),
            
            # Metadata
            'metadata': metadata or {}
        }
        
        # Save snapshot
        self.snapshots[stage_name] = snapshot_info
        self.flow_sequence.append(stage_name)
        
        return snapshot_info
    
    def compare(self, stage1: str, stage2: str, 
                algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare data changes between two stages
        
        Args:
            stage1: First stage name (input)
            stage2: Second stage name (output)
            algorithm_name: Algorithm name
        
        Returns:
            Comparison result dictionary
        """
        if stage1 not in self.snapshots or stage2 not in self.snapshots:
            raise ValueError(f"Stage '{stage1}' or '{stage2}' not found in snapshots")
        
        snap1 = self.snapshots[stage1]
        snap2 = self.snapshots[stage2]
        
        # Basic comparison
        comparison = {
            'algorithm': algorithm_name or f"{stage1} → {stage2}",
            'stage1': stage1,
            'stage2': stage2,
            'timestamp': datetime.now().isoformat(),
            
            # Shape comparison
            'shape_before': snap1['shape'],
            'shape_after': snap2['shape'],
            'shape_changed': snap1['shape'] != snap2['shape'],
            
            # Range comparison
            'range_before': [snap1['min'], snap1['max']],
            'range_after': [snap2['min'], snap2['max']],
            'range_change': [snap2['min'] - snap1['min'], snap2['max'] - snap1['max']],
            
            # Statistical changes
            'mean_change': snap2['mean'] - snap1['mean'],
            'mean_change_pct': abs(snap2['mean'] - snap1['mean']) / (abs(snap1['mean']) + 1e-10) * 100,
            'std_change': snap2['std'] - snap1['std'],
            
            # Data change detection
            'data_hash_before': snap1['data_hash'],
            'data_hash_after': snap2['data_hash'],
            'truly_changed': snap1['data_hash'] != snap2['data_hash'],
            
            # Data quality changes
            'nan_introduced': snap2['has_nan'] and not snap1['has_nan'],
            'inf_introduced': snap2['has_inf'] and not snap1['has_inf'],
            'nan_count_change': snap2['nan_count'] - snap1['nan_count'],
            'inf_count_change': snap2['inf_count'] - snap1['inf_count'],
        }
        
        # Save comparison result
        self.comparisons.append(comparison)
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any], verbose: bool = True):
        """Print comparison results (formatted output)"""
        print("─" * 60)
        print(f"📈 Data change verification: {comparison['algorithm']}")
        print(f"   Output shape: {comparison['shape_after']}", end="")
        if comparison['shape_changed']:
            print(f" ⚠️  Shape changed (before: {comparison['shape_before']})")
        else:
            print(" ✅ Shape unchanged")
        
        # Range changes
        r_before = comparison['range_before']
        r_after = comparison['range_after']
        print(f"   Output range: [{r_after[0]:.2f}, {r_after[1]:.2f}]", end="")
        
        range_change_significant = (
            abs(r_after[0] - r_before[0]) > abs(r_before[0]) * 0.1 or
            abs(r_after[1] - r_before[1]) > abs(r_before[1]) * 0.1
        )
        if range_change_significant:
            print(f" 📊 Range changed (before: [{r_before[0]:.2f}, {r_before[1]:.2f}])")
        else:
            print(" ✅ Range similar")
        
        # Mean changes
        mean_change = comparison['mean_change']
        mean_change_pct = comparison['mean_change_pct']
        print(f"   Mean change: {mean_change:.2f} (relative: {mean_change_pct:.2f}%)")
        
        # Data hash
        if comparison['truly_changed']:
            print(f"   Data hash: {comparison['data_hash_after'][:8]}... ✅ Data changed")
        else:
            print(f"   Data hash: {comparison['data_hash_after'][:8]}... ⚠️  Data unchanged!")
        
        # NaN/Inf detection
        if comparison['nan_introduced'] or comparison['inf_introduced']:
            print(f"   ❌ Warning: Anomalies introduced!")
            if comparison['nan_introduced']:
                print(f"      Added {comparison['nan_count_change']} NaN values")
            if comparison['inf_introduced']:
                print(f"      Added {comparison['inf_count_change']} Inf values")
        else:
            print(f"   NaN/Inf detection: ✅ No anomalies")
        
        if verbose:
            print(f"   Timestamp: {comparison['timestamp']}")
    
    def generate_report(self) -> str:
        """Generate complete data flow report"""
        report = []
        report.append("═" * 60)
        report.append("📊 Data Flow Tracking Report")
        report.append("═" * 60)
        report.append(f"Total stages: {len(self.snapshots)}")
        report.append(f"Total comparisons: {len(self.comparisons)}")
        report.append(f"Data flow path: {' → '.join(self.flow_sequence)}")
        report.append("")
        
        # Detailed information for each stage
        for stage_name in self.flow_sequence:
            snap = self.snapshots[stage_name]
            report.append("─" * 60)
            report.append(f"🔹 Stage: {stage_name}")
            report.append(f"   Data shape: {snap['shape']}")
            report.append(f"   Data range: [{snap['min']:.2f}, {snap['max']:.2f}]")
            report.append(f"   Data mean: {snap['mean']:.2f} ± {snap['std']:.2f}")
            report.append(f"   Data hash: {snap['data_hash'][:16]}...")
            if snap['has_nan'] or snap['has_inf']:
                report.append(f"   ⚠️  Anomalies: NaN={snap['nan_count']}, Inf={snap['inf_count']}")
            if snap['metadata']:
                report.append(f"   Metadata: {snap['metadata']}")
        
        report.append("═" * 60)
        return "\n".join(report)
    
    def _get_data_preview(self, data_array: np.ndarray, n_samples: int = 3, n_features: int = 5) -> List:
        """Get data preview (first few samples, first few features)"""
        if data_array.ndim == 1:
            return data_array[:n_features].tolist()
        else:
            n_samples = min(n_samples, data_array.shape[0])
            n_features = min(n_features, data_array.shape[1])
            return data_array[:n_samples, :n_features].tolist()
    
    def _compute_hash(self, data_array: np.ndarray) -> str:
        """Calculate data hash value (for detecting data changes)"""
        # Use byte representation of data to calculate MD5 hash
        # Note: Only use part of data to improve speed
        if data_array.size > 10000:
            # Large dataset: sample
            sample_indices = np.linspace(0, data_array.size - 1, 10000, dtype=int)
            sample_data = data_array.flat[sample_indices]
        else:
            sample_data = data_array
        
        # Convert to bytes and calculate hash
        data_bytes = sample_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()
    
    def clear(self):
        """Clear all records"""
        self.snapshots.clear()
        self.flow_sequence.clear()
        self.comparisons.clear()


# Global data flow tracker instance
data_flow_tracker = DataFlowTracker()












