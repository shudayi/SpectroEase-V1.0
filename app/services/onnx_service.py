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
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据，用于推断输入形状
            model_name: 模型名称
            method: 训练Method名称
            
        Returns:
            str: 保存的ONNX文件路径，如果失败返回None
        """
        try:
            print("🔧 尝试ONNX格式导出...")
            
            # 检查模型是否支持ONNX导出
            if not self.can_export_to_onnx(model):
                print(f"⚠️ 模型类型 {type(model).__name__} 不支持ONNX导出")
                return None
            
            # 尝试导入skl2onnx库
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError:
                print("⚠️ skl2onnx库未安装，无法导出ONNX格式")
                print("💡 请安装: pip install skl2onnx onnxruntime")
                return None
            
            # 推断输入形状
            if len(X_sample.shape) == 1:
                initial_type = [('float_input', FloatTensorType([None, 1]))]
            else:
                n_features = X_sample.shape[1]
                initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # 转换模型
            try:
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                
                # 生成文件名
                if model_name is None:
                    model_name = type(model).__name__
                
                if method:
                    filename = f"{method}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
                else:
                    filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
                
                filepath = os.path.join(self.save_directory, filename)
                
                # 保存ONNX模型
                with open(filepath, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                
                print(f"✅ ONNX模型保存成功: {filepath}")
                return filepath
                
            except Exception as e:
                print(f"❌ ONNX转换失败: {e}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def auto_save_model(self, model: Any, X_sample: np.ndarray, 
                      model_name: str = None, method: str = None,
                      evaluation_results: Dict = None) -> Optional[str]:
        """
        自动保存模型为pickle格式（针对边缘设备部署优化）
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据
            model_name: 模型名称
            method: 训练Method名称
            evaluation_results: 评估结果
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 使用print而不是logger.info，避免阻塞UI
            print("🚀 开始自动保存模型...")
            
            # 直接使用pickle格式（快速且适合边缘设备）
            print("💾 使用pickle格式保存（适合边缘设备部署）...")
            pickle_path = self.save_model_as_pickle(model, X_sample, model_name, method, evaluation_results)
            
            if pickle_path:
                print(f"✅ Pickle格式保存成功: {pickle_path}")
                # 保存模型信息
                self.save_model_info(pickle_path, model, method, evaluation_results)
                return pickle_path
            else:
                print("❌ 模型保存失败")
                return None
                    
        except Exception as e:
            print(f"❌ 自动保存模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_model_info(self, model_path: str, model: Any, method: str = None, 
                       evaluation_results: Dict = None):
        """
        保存模型相关信息到文本文件
        
        Args:
            model_path: 模型文件路径（.onnx 或 .pkl）
            model: Original模型
            method: 训练Method
            evaluation_results: 评估结果
        """
        try:
            # 生成信息文件路径，正确处理.pkl和.onnx文件
            if model_path.endswith('.onnx'):
                info_path = model_path.replace('.onnx', '_info.txt')
                model_file_label = "ONNX File"
            elif model_path.endswith('.pkl'):
                info_path = model_path.replace('.pkl', '_info.txt')
                model_file_label = "Model File"
            else:
                # 默认处理
                base_name = os.path.splitext(model_path)[0]
                info_path = f"{base_name}_info.txt"
                model_file_label = "Model File"
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Model Information\n")
                f.write(f"=================\n\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Type: {type(model).__name__}\n")
                f.write(f"Training Method: {method or 'Unknown'}\n")
                f.write(f"{model_file_label}: {os.path.basename(model_path)}\n\n")
                
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
                
                # 保存模型Parameters
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
                           model_name: str = None, method: str = None,
                           evaluation_results: Dict = None) -> Optional[str]:
        """
        使用pickle格式保存模型（针对边缘设备部署优化）
        
        Args:
            model: 训练好的模型
            X_sample: 样本数据
            model_name: 模型名称
            method: 训练Method名称
            evaluation_results: 评估结果
            
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
            
            # 构建部署包（包含边缘设备需要的所有信息）
            deployment_package = {
                # 核心模型
                'model': model,
                
                # 模型元数据
                'model_name': model_name,
                'model_type': type(model).__name__,
                'method': method,
                'timestamp': datetime.now().isoformat(),
                
                # 输入输出规格
                'input_shape': X_sample.shape,
                'n_features': X_sample.shape[1] if len(X_sample.shape) > 1 else X_sample.shape[0],
                'n_samples_trained': X_sample.shape[0],
                
                # 模型属性（用于推理验证）
                'model_attributes': {}
            }
            
            # 提取模型关键属性（用于边缘设备验证）
            if hasattr(model, 'classes_'):
                deployment_package['model_attributes']['classes'] = model.classes_.tolist()
                deployment_package['model_attributes']['n_classes'] = len(model.classes_)
            
            if hasattr(model, 'n_features_in_'):
                deployment_package['model_attributes']['n_features_in'] = model.n_features_in_
            
            if hasattr(model, 'feature_names_in_'):
                deployment_package['model_attributes']['feature_names'] = model.feature_names_in_.tolist()
            
            # 添加评估结果（用于质量监控）
            if evaluation_results:
                # 只保留数值型指标，排除大型对象
                deployment_package['performance_metrics'] = {
                    k: v for k, v in evaluation_results.items()
                    if isinstance(v, (int, float, str)) and k not in ['Confusion Matrix', 'Classification Report']
                }
            
            # 保存模型
            print(f"💾 保存pickle模型到: {filepath}")
            print(f"📦 部署包内容: model + metadata + performance_metrics")
            
            with open(filepath, 'wb') as f:
                pickle.dump(deployment_package, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 显示文件大小
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✅ Model successfully saved as pickle: {filepath}")
            print(f"📊 File size: {file_size:.2f} MB")
            print(f"🎯 Ready for edge device deployment")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Pickle保存失败: {e}")
            import traceback
            traceback.print_exc()
            return None 