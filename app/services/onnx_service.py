"""
ONNX Model Export Service
自动将训练好的scikit-learn模型转换并保存为ONNX格式
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)

class ONNXService:
    def __init__(self, save_directory: str = "models"):
        """
        初始化ONNX服务
        
        Args:
            save_directory: 模型保存目录
        """
        # Handle exe environment - use absolute path
        if getattr(sys, 'frozen', False):
            # Running as exe
            exe_dir = os.path.dirname(sys.executable)
            self.save_directory = os.path.join(exe_dir, save_directory)
        else:
            # Running as script
            self.save_directory = os.path.abspath(save_directory)
        
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        """确保保存目录存在"""
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                logger.info(f"Created directory: {self.save_directory}")
        except PermissionError:
            # Fallback to user documents folder
            user_docs = os.path.expanduser("~/Documents/SpectroEase")
            self.save_directory = os.path.join(user_docs, "models")
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                logger.info(f"Created fallback directory: {self.save_directory}")
    
    def can_export_to_onnx(self, model: Any) -> bool:
        """
        检查模型是否可以导出到ONNX格式
        
        Args:
            model: 训练好的模型
            
        Returns:
            bool: 是否可以导出
        """
        try:
            # 检查是否为支持的模型类型
            model_type = type(model).__name__
            
            # 支持的模型类型列表
            supported_models = [
                'LogisticRegression',
                'RandomForestClassifier',
                'RandomForestRegressor',
                'SVC',
                'SVR',
                'KNeighborsClassifier',
                'KNeighborsRegressor',
                'DecisionTreeClassifier',
                'DecisionTreeRegressor',
                'GradientBoostingClassifier',
                'GradientBoostingRegressor',
                'LinearRegression',
                'Ridge',
                'Lasso',
                'ElasticNet',
                'MLPClassifier',
                'MLPRegressor',
                'AdaBoostClassifier',
                'AdaBoostRegressor',
                'ExtraTreesClassifier',
                'ExtraTreesRegressor',
                'GaussianNB',
                'MultinomialNB',
                'BernoulliNB'
            ]
            
            if model_type in supported_models:
                return True
            
            # 检查集成模型（如VotingClassifier等）
            if hasattr(model, 'estimators_'):
                return True
                
            logger.warning(f"Model type {model_type} may not be supported for ONNX export")
            return False
            
        except Exception as e:
            logger.error(f"Error checking ONNX compatibility: {e}")
            return False
    
    def export_model_to_onnx(self, model: Any, X_sample: np.ndarray, 
                           model_name: str = None, method: str = None) -> Optional[str]:
        """
        将模型导出为ONNX格式
        由于skl2onnx库在当前环境中存在问题，暂时禁用ONNX转换
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据，用于推断输入形状
            model_name: 模型名称
            method: 训练方法名称
            
        Returns:
            str: 保存的ONNX文件路径，如果失败返回None
        """
        try:
            print("🔧 尝试ONNX格式导出...")
            
            # 暂时禁用ONNX转换，因为skl2onnx在当前环境中有问题
            print("⚠️ ONNX转换暂时禁用，因为skl2onnx库存在兼容性问题")
            print("💡 将使用pickle格式保存模型")
            return None
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
            logger.error(f"Failed to export model to ONNX: {e}")
            return None
    
    def auto_save_model(self, model: Any, X_sample: np.ndarray, 
                      model_name: str = None, method: str = None,
                      evaluation_results: Dict = None) -> Optional[str]:
        """
        自动保存模型，优先尝试ONNX格式，失败则使用pickle
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据
            model_name: 模型名称
            method: 训练方法名称
            evaluation_results: 评估结果
            
        Returns:
            str: 保存的文件路径
        """
        try:
            print("🚀 开始自动保存模型...")
            
            # 首先尝试ONNX格式
            print("🔄 尝试ONNX格式保存...")
            onnx_path = self.export_model_to_onnx(model, X_sample, model_name, method)
            
            if onnx_path:
                print(f"✅ ONNX格式保存成功: {onnx_path}")
                # 保存模型信息
                self.save_model_info(onnx_path, model, method, evaluation_results)
                return onnx_path
            else:
                print("⚠️ ONNX格式保存失败，回退到pickle格式...")
                # 回退到pickle格式
                pickle_path = self.save_model_as_pickle(model, X_sample, model_name, method)
                if pickle_path:
                    print(f"✅ Pickle格式保存成功: {pickle_path}")
                    # 保存模型信息
                    self.save_model_info(pickle_path, model, method, evaluation_results)
                    return pickle_path
                else:
                    print("❌ 所有格式保存都失败了")
                    return None
                    
        except Exception as e:
            print(f"❌ 自动保存模型失败: {e}")
            logger.error(f"Auto save model failed: {e}")
            return None
    
    def save_model_info(self, onnx_path: str, model: Any, method: str = None, 
                       evaluation_results: Dict = None):
        """
        保存模型相关信息到文本文件
        
        Args:
            onnx_path: ONNX文件路径
            model: 原始模型
            method: 训练方法
            evaluation_results: 评估结果
        """
        try:
            # 生成信息文件路径
            info_path = onnx_path.replace('.onnx', '_info.txt')
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Model Information\n")
                f.write(f"=================\n\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Type: {type(model).__name__}\n")
                f.write(f"Training Method: {method or 'Unknown'}\n")
                f.write(f"ONNX File: {os.path.basename(onnx_path)}\n\n")
                
                if hasattr(model, 'task_type'):
                    f.write(f"Task Type: {model.task_type}\n")
                
                if hasattr(model, 'n_features_in_'):
                    f.write(f"Number of Features: {model.n_features_in_}\n")
                
                if hasattr(model, 'classes_'):
                    f.write(f"Number of Classes: {len(model.classes_)}\n")
                    f.write(f"Classes: {list(model.classes_)}\n")
                
                # 保存评估结果
                if evaluation_results:
                    f.write(f"\nEvaluation Results:\n")
                    f.write(f"------------------\n")
                    for key, value in evaluation_results.items():
                        if key not in ['Confusion Matrix', 'Classification Report']:
                            if isinstance(value, float):
                                f.write(f"{key}: {value:.4f}\n")
                            else:
                                f.write(f"{key}: {value}\n")
                
                # 保存模型参数
                if hasattr(model, 'get_params'):
                    f.write(f"\nModel Parameters:\n")
                    f.write(f"----------------\n")
                    params = model.get_params()
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"Model info saved to: {info_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model info: {e}")
    
    def save_model_as_pickle(self, model: Any, X_sample: np.ndarray, 
                           model_name: str = None, method: str = None) -> Optional[str]:
        """
        使用pickle格式保存模型
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据
            model_name: 模型名称
            method: 训练方法名称
            
        Returns:
            str: 保存的文件路径
        """
        try:
            import pickle
            
            # 生成文件名
            if model_name is None:
                model_name = type(model).__name__
            
            if method:
                filename = f"{method}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            else:
                filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            filepath = os.path.join(self.save_directory, filename)
            
            # 保存模型
            print(f"💾 保存pickle模型到: {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'model_name': model_name,
                    'method': method,
                    'input_shape': X_sample.shape,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            logger.info(f"✅ Model successfully saved as pickle: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Pickle保存失败: {e}")
            logger.error(f"Failed to save model as pickle: {e}")
            return None 