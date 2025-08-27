"""
Unified Data Processor - Solves duplicate data processing logic issues
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class UnifiedDataProcessor:
    """
    Unified Data Processor - Completely solves duplicate data processing logic issues
    
    Core principles:
    1. All labels always maintain original string format for display
    2. Use unified numeric indices for internal processing
    3. Provide unified conversion interface
    4. Eliminate string-to-float conversion issues in visualization components
    """
    
    def __init__(self):
        self.original_labels = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.task_type = None
        self.is_initialized = False
        
    def initialize_from_data(self, labels: Union[pd.Series, np.ndarray, list]) -> Dict[str, Any]:
        """
        从数据初始化处理器
        
        Args:
            labels: 原始标签数据
            
        Returns:
            Dict: 初始化结果信息
        """
        try:
            # 1. 标准化标签格式
            if isinstance(labels, pd.Series):
                labels_array = labels.values
            elif isinstance(labels, list):
                labels_array = np.array(labels)
            else:
                labels_array = np.array(labels)
            
            # 2. 确保所有标签都是字符串格式
            self.original_labels = np.array([str(label).strip() for label in labels_array], dtype='<U50')
            
            # 3. 获取唯一标签
            unique_labels = np.unique(self.original_labels)
            
            # 4. 判断任务类型
            self.task_type = self.detect_task_type(unique_labels)
            
            # 5. 创建标签映射
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
            
            self.is_initialized = True
            
            logger.info(f"✅ Unified data processor initialization completed:")
            logger.info(f"   Task type: {self.task_type}")
            logger.info(f"   Number of labels: {len(unique_labels)}")
            logger.info(f"   Number of samples: {len(self.original_labels)}")
            
            return {
                'task_type': self.task_type,
                'num_classes': len(unique_labels),
                'num_samples': len(self.original_labels),
                'unique_labels': unique_labels.tolist(),
                'label_mapping': {
                    'label_to_index': self.label_to_index,
                    'index_to_label': self.index_to_label
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Unified data processor initialization failed: {e}")
            raise
    
    def detect_task_type(self, labels=None):
        """Detect task type"""
        # 尝试将所有标签转换为数字
        numeric_labels = []
        has_non_numeric = False
        
        for label in labels:
            try:
                # 尝试转换为浮点数
                numeric_val = float(label)
                numeric_labels.append(numeric_val)
            except (ValueError, TypeError):
                has_non_numeric = True
                break
        
        # 如果有非数字标签，肯定是分类任务
        if has_non_numeric:
            return 'classification'
        
        # **CRITICAL FIX: More accurate numeric task judgment**
        # 如果都是数字，根据更严格的条件判断
        unique_count = len(labels)
        total_samples = len(self.original_labels)
        
        # 条件1: 类别数量很少（≤10个）且大多是整数 -> 分类
        if unique_count <= 10:
            integer_count = sum(1 for val in numeric_labels if val == int(val))
            if integer_count / len(numeric_labels) > 0.8:
                return 'classification'
        
        # 条件2: 类别数量适中（11-20个）但都是小整数（0-100范围） -> 分类  
        elif unique_count <= 20:
            if all(val == int(val) and 0 <= val <= 100 for val in numeric_labels):
                return 'classification'
        
        # 条件3: 类别数量很多（>20个）或包含小数 -> 回归
        # 特别是像辛烷值这样的连续数值（83.4, 85.25等）
        if unique_count > 20 or any(val != int(val) for val in numeric_labels):
            return 'regression'
        
        # 条件4: 类别数量占样本比例很大（>50%） -> 回归
        if unique_count / total_samples > 0.5:
            return 'regression'
            
        # 默认：如果不确定，偏向分类
        return 'classification'
    
    def get_display_labels(self) -> np.ndarray:
        """
        获取用于显示的标签（始终为字符串）
        
        Returns:
            np.ndarray: 字符串标签数组
        """
        if not self.is_initialized:
            raise ValueError("处理器未初始化")
        return self.original_labels.copy()
    
    def get_numeric_labels(self) -> np.ndarray:
        """
        获取用于机器学习的数字标签
        
        Returns:
            np.ndarray: 数字标签数组
        """
        if not self.is_initialized:
            raise ValueError("处理器未初始化")
        
        if self.task_type == 'classification':
            # 分类任务：返回标签索引
            return np.array([self.label_to_index[label] for label in self.original_labels])
        else:
            # 回归任务：尝试转换为浮点数
            try:
                return np.array([float(label) for label in self.original_labels])
            except ValueError as e:
                logger.error(f"回归任务标签转换失败: {e}")
                raise ValueError(f"回归任务标签无法转换为数字: {e}")
    
    def convert_predictions_to_display(self, predictions: np.ndarray) -> np.ndarray:
        """
        将预测结果转换为显示格式
        
        Args:
            predictions: 模型预测结果
            
        Returns:
            np.ndarray: 用于显示的预测结果
        """
        if not self.is_initialized:
            raise ValueError("处理器未初始化")
        
        if self.task_type == 'classification':
            # 分类任务：将索引转换回标签
            return np.array([self.index_to_label.get(int(pred), f"Unknown_{pred}") 
                           for pred in predictions], dtype='<U50')
        else:
            # 回归任务：直接返回数字
            return predictions
    
    def get_safe_labels_for_visualization(self) -> np.ndarray:
        """
        获取用于可视化的安全标签（保证不会触发浮点数转换错误）
        
        Returns:
            np.ndarray: 安全的字符串标签数组
        """
        if not self.is_initialized:
            raise ValueError("处理器未初始化")
        
        # **CRITICAL FIX: For regression tasks, do not add Class_ prefix**
        if self.task_type == 'regression':
            # 回归任务：直接返回原始数值标签，可视化组件会正确处理
            return self.original_labels.copy()
        else:
            # 分类任务：确保标签是安全的字符串格式，不会被误解为数字
            safe_labels = []
            for label in self.original_labels:
                # 如果标签看起来像数字，添加前缀以确保它被视为字符串
                if self._looks_like_number(label):
                    safe_labels.append(f"Class_{label}")
                else:
                    safe_labels.append(str(label))
            
            return np.array(safe_labels, dtype='<U50')
    
    def _looks_like_number(self, label: str) -> bool:
        """
        检查标签是否看起来像数字
        
        Args:
            label: 标签字符串
            
        Returns:
            bool: 是否看起来像数字
        """
        try:
            float(label)
            return True
        except (ValueError, TypeError):
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取处理器信息
        
        Returns:
            Dict: 处理器信息
        """
        if not self.is_initialized:
            return {'initialized': False}
        
        return {
            'initialized': True,
            'task_type': self.task_type,
            'num_classes': len(self.index_to_label),
            'num_samples': len(self.original_labels),
            'unique_labels': list(self.index_to_label.values()),
            'sample_labels': self.original_labels[:5].tolist()
        }

# 全局实例
unified_processor = UnifiedDataProcessor() 