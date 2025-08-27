# app/services/modeling_service.py

from app.utils.exceptions import ModelingError
from interfaces.modeling_algorithm import ModelingAlgorithm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import inspect
from typing import Dict, Any, List, Tuple, Union, Optional
from app.utils.data_compatibility import standardize_classification_labels
from app.services.evaluation_service import (
    safe_accuracy, safe_precision, safe_recall, safe_f1
)
from app.utils.data_compatibility import prepare_labels_safely
import time

class ModelingService:
    def __init__(self, plugins: dict):
        self.plugins = plugins  # 动态加载的插件字典
        # 全局random_state管理（需求E）
        self.global_random_state = 42

    def create_leak_proof_pipeline(self, model_name: str, params: dict, task_type: str, 
                                   enable_scaling: bool = True, enable_feature_selection: bool = False,
                                   n_features_to_select: int = None) -> Pipeline:
        """创建防信息泄漏的Pipeline（需求E）
        
        Args:
            model_name: 模型名称
            params: 模型参数
            task_type: 任务类型 ('classification' or 'regression')
            enable_scaling: 是否启用标准化
            enable_feature_selection: 是否启用特征选择
            n_features_to_select: 特征选择数量
            
        Returns:
            Pipeline: 完整的处理流水线
        """
        steps = []
        
        # 步骤1：标准化/缩放（在CV内执行）
        if enable_scaling:
            steps.append(('scaler', StandardScaler()))
        
        # 步骤2：特征选择（在CV内执行）
        if enable_feature_selection and n_features_to_select:
            if task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            else:
                selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
            steps.append(('feature_selection', selector))
        
        # 步骤3：模型（确保使用全局random_state）
        model_params = params.copy()
        if 'random_state' not in model_params:
            model_params['random_state'] = self.global_random_state
        
        # 根据模型名称创建模型
        if model_name == "Logistic Regression":
            model = LogisticRegression(**model_params)
        elif model_name == "Random Forest" or model_name == "rf":
            if task_type == 'classification':
                model = RandomForestClassifier(**model_params)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**model_params)
        elif model_name == "Support Vector Machine (SVM)" or model_name == "svm":
            if task_type == 'classification':
                model = SVC(**model_params)
            else:
                from sklearn.svm import SVR
                model = SVR(**model_params)
        elif model_name == "K-Nearest Neighbors (KNN)" or model_name == "knn":
            if task_type == 'classification':
                model = KNeighborsClassifier(**model_params)
            else:
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor(**model_params)
        elif model_name == "Decision Tree" or model_name == "dt":
            if task_type == 'classification':
                model = DecisionTreeClassifier(**model_params)
            else:
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(**model_params)
        elif model_name == "Neural Network":
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            if task_type == 'classification':
                model = MLPClassifier(**model_params)
            else:
                model = MLPRegressor(**model_params)
        else:
            raise ModelingError(f"Unsupported model for pipeline: {model_name}")
        
        steps.append(('model', model))
        
        # 创建Pipeline
        pipeline = Pipeline(steps)
        
        # 存储元信息
        pipeline._task_type = task_type
        pipeline._global_random_state = self.global_random_state
        pipeline._model_name = model_name
        
        print(f"✅ Created leak-proof pipeline for {model_name}:")
        print(f"   Steps: {[step[0] for step in steps]}")
        print(f"   Task type: {task_type}")
        print(f"   Global random_state: {self.global_random_state}")
        
        return pipeline

    def train_model_with_cv(self, model_name: str, params: dict, X: pd.DataFrame, y: pd.Series,
                           cv_params: dict = None, hyperopt_params: dict = None):
        """使用交叉验证训练模型，确保防泄漏（需求E）
        
        Args:
            model_name: 模型名称
            params: 模型参数
            X: 特征数据
            y: 目标变量
            cv_params: 交叉验证参数
            hyperopt_params: 超参数优化参数
            
        Returns:
            训练好的模型或Pipeline
        """
        # 检测任务类型
        task_type = self._detect_task_type(y)
        
        # 默认CV参数
        if cv_params is None:
            cv_params = {
                'enable_scaling': True,
                'enable_feature_selection': False,
                'n_features_to_select': min(20, X.shape[1] // 2)
            }
        
        # 创建防泄漏Pipeline
        pipeline = self.create_leak_proof_pipeline(
            model_name=model_name,
            params=params,
            task_type=task_type,
            enable_scaling=cv_params.get('enable_scaling', True),
            enable_feature_selection=cv_params.get('enable_feature_selection', False),
            n_features_to_select=cv_params.get('n_features_to_select', None)
        )
        
        # 处理标签（分类任务）
        if task_type == 'classification':
            if isinstance(y, pd.Series):
                y_processed = y.astype(str)
            else:
                y_processed = np.array([str(label) for label in y])
            
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_processed)
            
            # 存储标签编码器
            pipeline._label_encoder = label_encoder
            y_final = y_encoded
        else:
            y_final = y
        
        # 超参数优化（可选）
        if hyperopt_params and hyperopt_params.get('enable', False):
            param_grid = hyperopt_params.get('param_grid', {})
            cv_folds = hyperopt_params.get('cv_folds', 5)
            
            # 为Pipeline参数添加前缀
            pipeline_param_grid = {}
            for param_name, param_values in param_grid.items():
                pipeline_param_grid[f'model__{param_name}'] = param_values
            
            if pipeline_param_grid:
                search_method = hyperopt_params.get('method', 'grid')
                if search_method == 'random':
                    n_iter = hyperopt_params.get('n_iter', 10)
                    search = RandomizedSearchCV(
                        pipeline, pipeline_param_grid, 
                        cv=cv_folds, n_iter=n_iter, 
                        random_state=self.global_random_state,
                        n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        pipeline, pipeline_param_grid, 
                        cv=cv_folds, 
                        n_jobs=-1
                    )
                
                print(f"🔍 Performing {search_method} search with CV={cv_folds}")
                search.fit(X, y_final)
                best_pipeline = search.best_estimator_
                
                print(f"✅ Best parameters: {search.best_params_}")
                print(f"✅ Best CV score: {search.best_score_:.4f}")
                
                return best_pipeline
        
        # 直接训练Pipeline
        print(f"🚀 Training {model_name} pipeline...")
        pipeline.fit(X, y_final)
        print("✅ Pipeline training completed")
        
        return pipeline

    def _detect_task_type(self, y):
        """检测任务类型"""
        if not hasattr(self, 'label_processor'):
            from app.utils.label_processor import EnhancedLabelProcessor
            self.label_processor = EnhancedLabelProcessor()
        
        return self.label_processor.detect_task_type(y)

    def train_model(self, model_name: str, params: dict, X: pd.DataFrame, y: pd.Series):
        # Ensure params is a dictionary without model_method
        if params is None:
            params = {}
        elif 'model_method' in params:
            # Remove model_method parameter to avoid passing to sklearn models
            params = {k: v for k, v in params.items() if k != 'model_method'}
        
        # Check if task_type is explicitly specified
        task_type = params.pop('task_type', None)
        
  
        force_regression = False
        force_classification = False
        
        # DIRECT APPROACH: Check stack frames for analysis_type
        for frame_info in inspect.stack():
            if 'main_window.py' in frame_info.filename:
                frame = frame_info.frame
                if 'analysis_type' in frame.f_locals:
                    analysis_type = frame.f_locals['analysis_type']
                    if analysis_type.startswith('Quantitative'):
                        print(f"Detected quantitative analysis context: {analysis_type}")
  
                        force_regression = True
                        task_type = 'regression'
                        break
                    elif analysis_type.startswith('Qualitative'):
                        print(f"Detected qualitative analysis context: {analysis_type}")
  
                        force_classification = True
                        task_type = 'classification'
                        break
        
  
        if force_regression:
            is_classification = False
            print("Enforced: Regression model forced by UI selection")
        elif force_classification:
            is_classification = True
            print("Enforced: Classification model forced by UI selection")
  
        elif task_type:
            is_classification = (task_type == 'classification')
            print(f"Using explicitly specified task type: {task_type}")
  
        else:
  
            is_classification = self._is_classification_task(y)
            print(f"Auto-detected task type based on data: {'classification' if is_classification else 'regression'}")
        
  
        task_type_str = "classification" if is_classification else "regression"
        print(f"Final decision - Task type: {task_type_str}")
        print(f"  • Force regression: {force_regression}")
        print(f"  • Force classification: {force_classification}")
        print(f"  • Target variable unique values: {y.nunique()}")
        print(f"  • Target variable sample count: {len(y)}")
            
        if model_name in self.plugins:
            algorithm: ModelingAlgorithm = self.plugins[model_name]
            algorithm.train(X, y, params)
            return algorithm
        else:
            # Handle built-in models
            # **CRITICAL FIX: Remove unsafe label conversion - let the unified processing handle it**
            def safe_astype_int(y_data):
                """DEPRECATED: Do not convert labels here, let unified processing handle it"""
                # Always return original data to avoid string->float conversion errors
                return y_data
            
            try:
                if model_name == "Logistic Regression":
                    if not is_classification:
                        raise ModelingError("Logistic Regression只支持classification任务。")
  
                    y = safe_astype_int(y)
                    
                    model = LogisticRegression(**params)
                    setattr(model, 'task_type', 'classification')
                elif model_name == "Random Forest" or model_name == "rf":
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = RandomForestClassifier(**params)
  
                        setattr(model, 'task_type', 'classification')
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(**params)
  
                        setattr(model, 'task_type', 'regression')
                elif model_name == "Support Vector Machine (SVM)" or model_name == "svm":
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = SVC(**params)
                        setattr(model, 'task_type', 'classification')
                    else:
                        from sklearn.svm import SVR
                        model = SVR(**params)
                        setattr(model, 'task_type', 'regression')
                elif model_name == "K-Nearest Neighbors (KNN)" or model_name == "knn":
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = KNeighborsClassifier(**params)
                        setattr(model, 'task_type', 'classification')
                    else:
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(**params)
                        setattr(model, 'task_type', 'regression')
                elif model_name == "Decision Tree" or model_name == "dt":
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = DecisionTreeClassifier(**params)
                        setattr(model, 'task_type', 'classification')
                    else:
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(**params)
                        setattr(model, 'task_type', 'regression')
                elif model_name == "Gradient Boosting" or model_name == "gb":
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = GradientBoostingClassifier(**params)
                        setattr(model, 'task_type', 'classification')
                    else:
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(**params)
                        setattr(model, 'task_type', 'regression')
                elif model_name == "XGBoost":
                    try:
                        import xgboost as xgb
                        if is_classification:
  
                            y = safe_astype_int(y)
                                
                            model = xgb.XGBClassifier(**params)
                            setattr(model, 'task_type', 'classification')
                        else:
                            model = xgb.XGBRegressor(**params)
                            setattr(model, 'task_type', 'regression')
                    except ImportError:
                        raise ModelingError("XGBoost is not installed. Please install it with: pip install xgboost")
                elif model_name == "Neural Network":
                    from sklearn.neural_network import MLPClassifier, MLPRegressor
                    if is_classification:
  
                        y = safe_astype_int(y)
                            
                        model = MLPClassifier(**params)
                        setattr(model, 'task_type', 'classification')
                    else:
                        model = MLPRegressor(**params)
                        setattr(model, 'task_type', 'regression')
                else:
                    raise ModelingError(f"Unsupported model: {model_name}")
                
                print(f"Fitting {model.__class__.__name__} for {task_type_str} task")
                
                # **CRITICAL FIX: Handle mixed label types before calling sklearn fit**
                if is_classification:
                    # Convert all labels to consistent string type first
                    print("=== CRITICAL FIX: Standardizing labels for sklearn compatibility ===")
                    
                    if isinstance(y, pd.Series):
                        y_consistent = y.astype(str)
                    elif isinstance(y, pd.DataFrame):
                        y_consistent = y.iloc[:, 0].astype(str)
                    elif hasattr(y, 'astype'):
                        y_consistent = y.astype(str)
                    else:
                        y_consistent = np.array([str(label) for label in y])
                    
                    print(f"Labels converted to consistent string type: {type(y_consistent).__name__}")
                    print(f"Sample labels: {list(y_consistent[:5])}")
                    
                    # Now encode string labels to integers for sklearn
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y_consistent)
                    
                    print(f"Labels encoded to integers: {type(y_encoded).__name__}")
                    print(f"Label range: {y_encoded.min()} to {y_encoded.max()}")
                    print(f"Number of classes: {len(np.unique(y_encoded))}")
                    print("✅ Label type consistency verified - ready for sklearn fit")
                    
                    # Store label encoder for later use
                    setattr(model, '_label_encoder', label_encoder)
                    y = y_encoded
                    
                elif hasattr(y, 'dtype') and y.dtype.kind == 'f':  # 浮点类型
                    print(f"Converting {y.dtype} type labels to integer type")
                    # **CRITICAL FIX: Use enhanced label processor to check before conversion**
                    conversion_task_type = self.label_processor.detect_task_type(y)
                    if conversion_task_type == 'classification':
                        print("🔧 Classification labels detected - skipping numeric conversion")
                        # Keep as is for classification
                        pass
                    else:
                        print("🔧 Regression labels detected - proceeding with conversion")
                        y = safe_astype_int(y)
                
                # **CRITICAL FIX: Final safety check before sklearn fit**
                print(f"🔧 Final data check before model.fit:")
                print(f"   X shape: {X.shape}, dtype: {X.dtypes.iloc[0] if hasattr(X, 'dtypes') else type(X)}")
                print(f"   y type: {type(y)}, dtype: {y.dtype if hasattr(y, 'dtype') else 'no dtype'}")
                print(f"   y sample: {y[:5] if hasattr(y, '__getitem__') else y}")
                
                # Additional safety check for string labels in y
                if hasattr(y, 'dtype') and y.dtype.kind in 'UO':  # Unicode or Object
                    print("❌ CRITICAL ERROR: y still contains string labels before sklearn fit!")
                    # This should never happen at this point, but let's handle it
                    from sklearn.preprocessing import LabelEncoder
                    emergency_encoder = LabelEncoder()
                    y = emergency_encoder.fit_transform(y.astype(str))
                    setattr(model, '_emergency_encoder', emergency_encoder)
                    print(f"🚨 Emergency label encoding applied, y now: {y[:5]}")
                
                model.fit(X, y)
                print("✅ model.fit() completed successfully")
                
  
                if not hasattr(model, 'task_type'):
                    task_type_val = "classification" if is_classification else "regression"
                    setattr(model, 'task_type', task_type_val)
                
                return model
            except Exception as e:
                raise ModelingError(f"Failed to train model {model_name}: {e}")
                
    def _is_classification_task(self, y):
        """
        DEPRECATED: Use enhanced label processor instead  
        This method is kept for backward compatibility
        """
        # **CRITICAL FIX: Use enhanced label processor for consistent detection**
        if not hasattr(self, 'label_processor'):
            from app.utils.label_processor import EnhancedLabelProcessor
            self.label_processor = EnhancedLabelProcessor()
        
        task_type = self.label_processor.detect_task_type(y)
        return task_type == 'classification'

    def predict(self, model_id: str, X: np.ndarray) -> Dict[str, Any]:
        """
        使用指定模型进行prediction
        
        Args:
            model_id: 模型ID
            X: 输入featuresdata
            
        Returns:
            Dict: 包含predictionresults的字典
        """
        try:
            model_data = self.get_model_data(model_id)
            if not model_data or not hasattr(model_data, 'model') or model_data.model is None:
                return {"error": "模型不存在或未training"}
                
            model = model_data.model
            
            # 准备输入数据
            try:
                if isinstance(X, pd.DataFrame):
                    X_data = X
                elif isinstance(X, np.ndarray):
                    if hasattr(model_data, 'feature_names'):
                        feature_names = model_data.feature_names
                        if len(feature_names) == X.shape[1]:
                            X_data = pd.DataFrame(X, columns=feature_names)
                        else:
                            X_data = pd.DataFrame(X)
                    else:
                        X_data = pd.DataFrame(X)
                else:
                    X_data = np.array(X)
            except Exception as e:
                print(f"Data preparation error: {e}")
                X_data = X
            
            # 进行预测
            predictions = model.predict(X_data)
            print(f"🔍 Raw model predictions: {predictions[:10]}")
            print(f"   Type: {type(predictions)}, dtype: {predictions.dtype if hasattr(predictions, 'dtype') else 'N/A'}")
            
            # **CRITICAL FIX: 检查并使用存储的LabelEncoder进行逆转换**
            if hasattr(model, '_label_encoder'):
                print("🗺️ Found stored LabelEncoder, converting predictions back to original labels")
                label_encoder = model._label_encoder
                try:
                    # 将整数预测结果转换回原始标签
                    predictions_original = label_encoder.inverse_transform(predictions)
                    print(f"✅ Converted predictions back to original labels:")
                    print(f"   Type: {type(predictions_original)}, dtype: {predictions_original.dtype if hasattr(predictions_original, 'dtype') else 'N/A'}")
                    print(f"   Sample: {predictions_original[:10]}")
                    
                    return {
                        "predictions": predictions_original.tolist() if hasattr(predictions_original, 'tolist') else list(predictions_original),
                        "task_type": "classification",
                        "shape": (len(predictions_original),),
                        "label_conversion": "success"
                    }
                except Exception as e:
                    print(f"❌ Label inverse transform failed: {e}")
                    # 继续使用原始预测结果
            elif hasattr(model, '_emergency_encoder'):
                print("🚨 Found emergency encoder, using for inverse transform")
                emergency_encoder = model._emergency_encoder
                try:
                    predictions_original = emergency_encoder.inverse_transform(predictions)
                    print(f"✅ Emergency conversion successful: {predictions_original[:10]}")
                    
                    return {
                        "predictions": predictions_original.tolist() if hasattr(predictions_original, 'tolist') else list(predictions_original),
                        "task_type": "classification",
                        "shape": (len(predictions_original),),
                        "label_conversion": "emergency_success"
                    }
                except Exception as e:
                    print(f"❌ Emergency label inverse transform failed: {e}")
            else:
                print("⚠️ No label encoder found in model, predictions may be in encoded format")
            
            # 获取任务类型
            task_type = None
            if hasattr(model, 'task_type'):
                task_type = model.task_type
            elif hasattr(model_data, 'task_type'):
                task_type = model_data.task_type
                
            # 如果没有找到编码器，尝试使用通用处理
            try:
                predictions_processed, actual_task_type = prepare_labels_safely(predictions, task_type)
                
                if task_type and actual_task_type != task_type:
                    print(f"Task type mismatch: expected {task_type}, got {actual_task_type}")
                
                if isinstance(predictions_processed, np.ndarray):
                    predictions_list = predictions_processed.tolist()
                else:
                    predictions_list = list(predictions_processed)
                    
                return {
                    "predictions": predictions_list,
                    "task_type": actual_task_type,
                    "shape": (len(predictions_list),),
                    "label_conversion": "generic"
                }
            except Exception as proc_error:
                print(f"Generic processing failed: {proc_error}")
                # 最后的兜底方案
                try:
                    predictions_list = [str(p) for p in predictions]
                    return {
                        "predictions": predictions_list,
                        "note": "predictionresults已conversion为字符串",
                        "error_detail": str(proc_error),
                        "label_conversion": "fallback"
                    }
                except:
                    return {"error": "无法processingpredictionresults", "detail": str(proc_error)}
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}
            
    def evaluate(self, model_data, X_test, y_test):
        """
        Enhanced model evaluation with proper task type handling
        
        Args:
            model_data: Model data containing trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results
        """
        try:
            model = model_data.model if hasattr(model_data, 'model') else model_data
            
            print(f"🔍 Model evaluation - Model type: {type(model).__name__}")
            print(f"   Test data shape: X={X_test.shape}, y={len(y_test)}")
            print(f"   Test labels sample: {list(y_test[:5])}")
            
            # **CRITICAL FIX: Enhanced task type detection**
            task_type = None
            is_classification = None
            
            # Method 1: Check model type
            model_class_name = model.__class__.__name__
            print(f"   Model class: {model_class_name}")
            
            # First, try to get task type from model attributes
            if hasattr(model_data, 'task_type'):
                task_type = model_data.task_type
                is_classification = (task_type == 'classification')
                print(f"✅ Task type from model_data: {task_type}")
            elif hasattr(model, 'task_type'):
                task_type = model.task_type
                is_classification = (task_type == 'classification')
                print(f"✅ Task type from model: {task_type}")
            else:
                # Method 2: Infer from model class name
                if model_class_name in ['RandomForestClassifier', 'SVC', 'LogisticRegression', 'GaussianNB', 
                                      'KNeighborsClassifier', 'DecisionTreeClassifier', 'QualitativeAnalyzer']:
                    is_classification = True
                    task_type = 'classification'
                    print(f"✅ CLASSIFICATION inferred from model class: {model_class_name}")
                elif model_class_name in ['RandomForestRegressor', 'SVR', 'LinearRegression', 'Ridge', 'Lasso']:
                    is_classification = False
                    task_type = 'regression'
                    print(f"✅ REGRESSION inferred from model class: {model_class_name}")
                elif model_class_name in ['KMeans', 'AgglomerativeClustering', 'DBSCAN']:
                    task_type = 'clustering'
                    is_classification = False
                    print(f"✅ CLUSTERING inferred from model class: {model_class_name}")
                    
            # Method 3: Analyze labels if task type still unknown
            if task_type is None:
                from app.utils.label_processor import EnhancedLabelProcessor
                label_processor = EnhancedLabelProcessor()
                detected_task_type = label_processor.detect_task_type(y_test)
                is_classification = (detected_task_type == 'classification')
                task_type = detected_task_type
                print(f"✅ Task type detected from labels: {task_type}")
            
            print(f"🎯 Final task type: {task_type}, is_classification: {is_classification}")
            
            # **CRITICAL FIX: Validate task type and labels compatibility**
            if task_type == 'regression':
                # For regression, check if labels are actually numeric
                try:
                    sample_labels = list(y_test[:5])
                    has_string_labels = any(any(c.isalpha() for c in str(label)) for label in sample_labels)
                    
                    if has_string_labels:
                        print("❌ TASK TYPE MISMATCH: Regression model with string labels!")
                        print(f"   Sample labels: {sample_labels}")
                        print("   Forcing classification mode...")
                        task_type = 'classification'
                        is_classification = True
                except:
                    pass
            
            # Get model predictions
            print("🔮 Getting model predictions...")
            y_pred = model.predict(X_test)
            print(f"   Predictions sample: {list(y_pred[:5])}")
            
            # Process evaluation based on task type
            if is_classification:
                # **CLASSIFICATION EVALUATION**
                print("📊 Performing classification evaluation...")
                
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    confusion_matrix
                )
                
                try:
                    # Use enhanced label processing for proper handling
                    from app.utils.label_processor import EnhancedLabelProcessor
                    label_processor = EnhancedLabelProcessor()
                    
                    # Process true labels
                    y_test_processed, test_metadata = label_processor.process_labels_smart(y_test, 'classification')
                    
                    # Process predictions - convert back to original format first
                    if hasattr(model, 'label_inverse_map') and model.label_inverse_map:
                        # Model has its own label mapping
                        y_pred_original = [model.label_inverse_map.get(int(pred), f"Unknown_{pred}") for pred in y_pred]
                    else:
                        y_pred_original = y_pred
                    
                    # Process predictions through same label processor
                    y_pred_processed, pred_metadata = label_processor.process_labels_smart(y_pred_original, 'classification')
                    
                    # Ensure both have same label mapping
                    label_mapping = test_metadata.get('mapping', {})
                    num_classes = test_metadata.get('num_classes', len(np.unique(y_test_processed)))
                    
                    print(f"   Classes: {num_classes}, Mapping: {label_mapping.get('unique_labels', 'N/A')}")

                    # Calculate metrics with proper error handling
                    def safe_metric_calc(metric_func, *args, **kwargs):
                        try:
                            return float(metric_func(*args, **kwargs))
                        except Exception as e:
                            print(f"   Warning: {metric_func.__name__} calculation failed: {e}")
                            return 0.0

                    # Calculate accuracy
                    accuracy = safe_metric_calc(accuracy_score, y_test_processed, y_pred_processed)
                    
                    # Calculate precision, recall, f1 based on number of classes
                    if num_classes == 2:
                        precision = safe_metric_calc(precision_score, y_test_processed, y_pred_processed, average='binary', zero_division=0)
                        recall = safe_metric_calc(recall_score, y_test_processed, y_pred_processed, average='binary', zero_division=0)
                        f1 = safe_metric_calc(f1_score, y_test_processed, y_pred_processed, average='binary', zero_division=0)
                    else:
                        precision = safe_metric_calc(precision_score, y_test_processed, y_pred_processed, average='weighted', zero_division=0)
                        recall = safe_metric_calc(recall_score, y_test_processed, y_pred_processed, average='weighted', zero_division=0)
                        f1 = safe_metric_calc(f1_score, y_test_processed, y_pred_processed, average='weighted', zero_division=0)

                    # Calculate confusion matrix
                    try:
                        cm = confusion_matrix(y_test_processed, y_pred_processed)
                        cm_dict = {
                            "matrix": cm.tolist(), 
                            "labels": label_mapping.get('unique_labels', list(range(num_classes)))
                        }
                    except Exception as cm_error:
                        print(f"   Warning: Confusion matrix calculation failed: {cm_error}")
                        cm_dict = {"matrix": [[0]], "labels": ["Unknown"]}
                    
                    results = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion_matrix": cm_dict,
                        "task_type": "classification",
                        "num_classes": num_classes
                    }
                    
                    print(f"✅ Classification evaluation completed:")
                    print(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    return results
                    
                except Exception as classification_error:
                    print(f"❌ Classification evaluation failed: {classification_error}")
                    # Fallback evaluation
                    try:
                        accuracy = accuracy_score(y_test, y_pred) if len(set(y_test)) == len(set(y_pred)) else 0.0
                        return {
                            "accuracy": float(accuracy),
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "task_type": "classification",
                            "error": "Fallback evaluation used",
                            "original_error": str(classification_error)
                        }
                    except Exception as fallback_error:
                        return {
                            "error": f"Classification evaluation completely failed: {str(fallback_error)}",
                            "task_type": "classification"
                        }
            
            else:
                # **REGRESSION EVALUATION**
                print("📈 Performing regression evaluation...")
                
                # **CRITICAL FIX: Strict validation for regression task**
                # Check if this is actually a regression task or misclassified
                if task_type == 'classification':
                    print("❌ ERROR: Classification task incorrectly routed to regression evaluation")
                    return {"error": "Task type mismatch: Classification data sent to regression evaluation", "task_type": "classification"}
                
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # **ENHANCED: Safe numeric conversion with validation**
                    def safe_numeric_conversion(values, context=""):
                        """Safely convert values to numeric with detailed error reporting"""
                        converted = []
                        errors = []
                        
                        for i, val in enumerate(values):
                            if val is None or pd.isna(val):
                                converted.append(np.nan)
                                continue
                                
                            try:
                                # Check for obvious string labels
                                val_str = str(val).strip()
                                if any(c.isalpha() for c in val_str):
                                    errors.append(f"Index {i}: '{val}' contains letters")
                                    converted.append(np.nan)
                                    continue
                                
                                # Try numeric conversion
                                numeric_val = float(val)
                                converted.append(numeric_val)
                                
                            except (ValueError, TypeError) as e:
                                errors.append(f"Index {i}: '{val}' - {str(e)}")
                                converted.append(np.nan)
                        
                        converted_array = np.array(converted, dtype=float)
                        error_ratio = len(errors) / len(values) if len(values) > 0 else 0
                        
                        # Report errors if significant
                        if errors:
                            print(f"   {context} conversion errors ({error_ratio:.1%}):")
                            for error in errors[:3]:  # Show first 3 errors
                                print(f"     {error}")
                            if len(errors) > 3:
                                print(f"     ... and {len(errors) - 3} more")
                        
                        return converted_array, error_ratio, errors

                    # Convert test labels and predictions
                    y_test_float, test_error_ratio, test_errors = safe_numeric_conversion(y_test, "Test labels")
                    y_pred_float, pred_error_ratio, pred_errors = safe_numeric_conversion(y_pred, "Predictions")
                    
                    # Check if conversion was successful enough for regression
                    if test_error_ratio > 0.1:  # More than 10% errors
                        print(f"❌ REGRESSION TASK VALIDATION FAILED:")
                        print(f"   {test_error_ratio:.1%} of test labels cannot be converted to numeric")
                        print(f"   Sample problematic labels: {[err.split(':')[1].strip() for err in test_errors[:3]]}")
                        print("   This suggests the task should be CLASSIFICATION, not REGRESSION")
                        
                        return {
                            "error": f"Regression task requires numeric labels, but {test_error_ratio:.1%} are non-numeric",
                            "task_type": "regression",
                            "suggestion": "Consider using CLASSIFICATION task instead",
                            "problematic_samples": test_errors[:5]
                        }
                    
                    # Calculate valid data mask
                    valid_mask = ~np.isnan(y_test_float) & ~np.isnan(y_pred_float)
                    valid_count = np.sum(valid_mask)
                    
                    if valid_count == 0:
                        return {
                            "error": "No valid numeric data available for regression evaluation", 
                            "task_type": "regression"
                        }
                    
                    # Extract valid data
                    y_test_clean = y_test_float[valid_mask]
                    y_pred_clean = y_pred_float[valid_mask]
                    
                    print(f"   Valid samples for regression: {valid_count}/{len(y_test)}")
                    
                    # Calculate regression metrics
                    mse = mean_squared_error(y_test_clean, y_pred_clean)
                    mae = mean_absolute_error(y_test_clean, y_pred_clean)
                    r2 = r2_score(y_test_clean, y_pred_clean)
                    
                    results = {
                        "task_type": "regression",
                        "mse": float(mse),
                        "rmse": float(np.sqrt(mse)),
                        "mae": float(mae),
                        "r2_score": float(r2),
                        "data_points": int(valid_count),
                        "data_coverage": float(valid_count / len(y_test))
                    }
                    
                    print(f"✅ Regression evaluation completed:")
                    print(f"   MSE: {mse:.6f}, RMSE: {np.sqrt(mse):.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
                    return results
                    
                except Exception as regression_error:
                    print(f"❌ Regression evaluation failed: {regression_error}")
                    return {
                        "error": f"Regression evaluation error: {str(regression_error)}", 
                        "task_type": "regression"
                    }
                    
        except Exception as e:
            print(f"❌ Model evaluation completely failed: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}

    def evaluate_model(self, model, X_test, y_test, task_type=None):
        """
        兼容性method - 提供与evaluation_service.evaluate_model相同的接口
        
        This is a bridge method to resolve the issue where controller calls evaluate_model but service only has evaluate method.
        该method将parametersprocessing为evaluatemethod所需的格式，并调用evaluatemethod。
        
        Args:
            model: training好的模型
            X_test: 测试features
            y_test: 测试labels
            task_type: 任务类型（可选，classification或regression）
            
        Returns:
            dict: evaluationresults字典
        """
        print("Calling evaluation through ModelingService.evaluate_model bridge method...")
        
  
        class ModelData:
            def __init__(self, model, task_type):
                self.model = model
                self.task_type = task_type
        
  
        if task_type is None and hasattr(model, 'task_type'):
            task_type = model.task_type
        
  
        model_data = ModelData(model, task_type)
        
  
        return self.evaluate(model_data, X_test, y_test)
