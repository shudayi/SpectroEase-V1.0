import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List, Optional
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

class QualitativeAnalyzer:
    """Qualitative analysis class"""
    
    def __init__(self, method: str = 'lda'):
        """
        Initialize qualitative analyzer
        
        Args:
            method: analysismethod ('kmeans', 'hierarchical', 'lda', 'qda', 'svm', 'rf', 'knn', 'dt', 'nn', 'xgboost', 'lightgbm')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.task_type = 'classification' 
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        train model
        
        Args:
            X: spectral data
            y: labels (for supervised learning methods)
            **kwargs: model parameters
        """

        if X is None or X.size == 0:
            raise ValueError("Input data X cannot be empty")
            

        if np.isnan(X).any():
            print("English text")               
            X_filled = X.copy()
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    X_filled[mask, col] = np.mean(X[~mask, col])
            X = X_filled
            

        feature_std = np.std(X, axis=0)
        if np.any(feature_std < 1e-10):
            print("English text")
            

        X_scaled = self.scaler.fit_transform(X)
        

        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.values  # convert to numpy array
            elif not isinstance(y, np.ndarray):
                y = np.array(y)  # ensure y is numpy array
        

        method_params = {}
        if self.method == 'kmeans':
            method_params = {
                'n_clusters': 3, 
                'random_state': 42, 
                'n_init': 10,
                'max_iter': 300,
                'tol': 1e-4,
                'algorithm': 'auto'
            }  
            method_params.update(kwargs)
            self.model = KMeans(**method_params)
            self.model.fit(X_scaled)
            
        elif self.method == 'hierarchical':
            method_params = {
                'n_clusters': 3, 
                'linkage': 'ward',
                'affinity': 'euclidean',
                'compute_full_tree': 'auto'
            }
            method_params.update(kwargs)
            self.model = AgglomerativeClustering(**method_params)
            self.model.fit(X_scaled)
            
        elif self.method in ['lda', 'qda']:
            if y is None:
                raise ValueError("Supervised learning methods require label data")
                

            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
                
            if self.method == 'lda':
                method_params = {
                    'solver': 'svd', 
                    'store_covariance': True,
                    'shrinkage': None,
                    'priors': None,
                    'n_components': None,
                    'tol': 1e-4
                }
                method_params.update(kwargs)
                self.model = LinearDiscriminantAnalysis(**method_params)
            else:  # qda
                method_params = {
                    'store_covariance': True,
                    'reg_param': 0.0,
                    'priors': None,
                    'tol': 1e-4
                }
                method_params.update(kwargs)
                self.model = QuadraticDiscriminantAnalysis(**method_params)
                
            self.model.fit(X_scaled, y_encoded)  
            self.label_encoder = label_encoder
            
        elif self.method == 'svm':
            if y is None:
                raise ValueError("Supervised learning methods require label data")
                

            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            

            method_params = {
                'C': 1.0, 
                'kernel': 'rbf', 
                'gamma': 'scale', 
                'probability': True, 
                'random_state': 42,
                'class_weight': 'balanced',
                'tol': 1e-3,
                'cache_size': 200
            }
            method_params.update(kwargs)
            self.model = SVC(**method_params)
            self.model.fit(X_scaled, y_encoded)

            self.label_encoder = label_encoder
            
        elif self.method == 'rf' or self.method.lower() == 'random forest':
            if y is None:
                raise ValueError("Supervised learning methods require label data")
                

            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
                

            method_params = {
                'n_estimators': 100, 
                'max_depth': None, 
                'min_samples_split': 2, 
                'min_samples_leaf': 1,
                'max_features': 'sqrt', 
                'bootstrap': True, 
                'random_state': 42,
                'class_weight': 'balanced',
                'oob_score': True,
                'n_jobs': -1,
                'warm_start': False,
                'criterion': 'gini'
            }
            method_params.update(kwargs)
            self.model = RandomForestClassifier(**method_params)
            self.model.fit(X_scaled, y_encoded)

            self.label_encoder = label_encoder
            
        elif self.method in ['knn', 'dt', 'nn', 'xgboost', 'lightgbm']:
            if y is None:
                raise ValueError("Supervised learning methods require label data")
                

            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            if self.method == 'knn':
  
                from sklearn.neighbors import KNeighborsClassifier
                method_params = {
                    'n_neighbors': 5, 
                    'weights': 'distance', 
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': 2,  # Euclidean distance
                    'metric': 'minkowski',
                    'n_jobs': -1
                }
                method_params.update(kwargs)
                self.model = KNeighborsClassifier(**method_params)
                
            elif self.method == 'dt':
  
                from sklearn.tree import DecisionTreeClassifier
                method_params = {
                    'max_depth': 10,  # limit tree depth to avoid overfitting
                    'random_state': 42, 
                    'class_weight': 'balanced',
                    'criterion': 'gini',
                    'splitter': 'best',
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': None,
                    'max_leaf_nodes': None
                }
                method_params.update(kwargs)
                self.model = DecisionTreeClassifier(**method_params)
                
            elif self.method == 'nn':
  
                try:
                    from sklearn.neural_network import MLPClassifier
                    method_params = {
                        'hidden_layer_sizes': (100,),
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.0001,
                        'batch_size': 'auto',
                        'learning_rate': 'adaptive',
                        'learning_rate_init': 0.001,
                        'max_iter': 200,
                        'shuffle': True,
                        'random_state': 42,
                        'tol': 1e-4,
                        'verbose': False,
                        'warm_start': False,
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'beta_1': 0.9,
                        'beta_2': 0.999
                    }
                    method_params.update(kwargs)
                    self.model = MLPClassifier(**method_params)
                except ImportError:
                    print("English text")
                    method_params = {
                        'n_estimators': 100, 
                        'max_depth': None, 
                        'random_state': 42,
                        'class_weight': 'balanced',
                        'n_jobs': -1
                    }
                    method_params.update(kwargs)
                    self.model = RandomForestClassifier(**method_params)
                    
            elif self.method == 'xgboost':
  
                try:
                    import xgboost as xgb
                    method_params = {
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'objective': 'multi:softprob',
                        'booster': 'gbtree',
                        'tree_method': 'auto',
                        'n_jobs': -1,
                        'gamma': 0,
                        'min_child_weight': 1,
                        'max_delta_step': 0,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'colsample_bylevel': 1,
                        'reg_alpha': 0,
                        'reg_lambda': 1,
                        'scale_pos_weight': 1,
                        'random_state': 42,
                        'missing': None
                    }
                    method_params.update(kwargs)
  
                    n_classes = len(np.unique(y_encoded))
                    if n_classes > 2:
                        method_params['num_class'] = n_classes
                    self.model = xgb.XGBClassifier(**method_params)
                except ImportError:
                    print("English text")
                    method_params = {
                        'n_estimators': 100, 
                        'max_depth': 6, 
                        'random_state': 42,
                        'class_weight': 'balanced',
                        'n_jobs': -1
                    }
                    method_params.update(kwargs)
                    self.model = RandomForestClassifier(**method_params)
                    
            elif self.method == 'lightgbm':
  
                try:
                    import lightgbm as lgb
                    method_params = {
                        'boosting_type': 'gbdt',
                        'num_leaves': 31,
                        'max_depth': -1,
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'subsample_for_bin': 200000,
                        'objective': 'multiclass',
                        'class_weight': 'balanced',
                        'min_split_gain': 0.0,
                        'min_child_weight': 0.001,
                        'min_child_samples': 20,
                        'subsample': 1.0,
                        'subsample_freq': 0,
                        'colsample_bytree': 1.0,
                        'reg_alpha': 0.0,
                        'reg_lambda': 0.0,
                        'random_state': 42,
                        'n_jobs': -1,
                        'silent': True
                    }
                    method_params.update(kwargs)
  
                    n_classes = len(np.unique(y_encoded))
                    if n_classes > 2:
                        method_params['num_class'] = n_classes
                    self.model = lgb.LGBMClassifier(**method_params)
                except ImportError:
                    print("English text")
                    method_params = {
                        'n_estimators': 100, 
                        'max_depth': None, 
                        'random_state': 42,
                        'class_weight': 'balanced',
                        'n_jobs': -1
                    }
                    method_params.update(kwargs)
                    self.model = RandomForestClassifier(**method_params)
            
  
            self.model.fit(X_scaled, y_encoded)
  
            self.label_encoder = label_encoder
            
        else:
            raise ValueError(f"Unsupported analysis method: {self.method}")
        
  
        if self.model is not None:
            setattr(self.model, 'task_type', 'classification')
            
  
        print("English text")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict categories
        
        Args:
            X: spectral data
            
        Returns:
            np.ndarray: predict categories
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
  
        if np.isnan(X).any():
            print("English text")
  
            X_filled = X.copy()
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    X_filled[mask, col] = np.mean(X[~mask, col])
            X = X_filled
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
  
        if hasattr(self, 'label_encoder') and self.method not in ['kmeans', 'hierarchical']:
            return self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, **kwargs) -> Dict:
        """
        Perform cross-validation and return evaluation metrics
        
        Args:
            X: spectral data
            y: labels
            n_splits: number of cross-validation splits
            
        Returns:
            Dict: dictionary containing evaluation metrics
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.model_selection import cross_val_score
            
  
        X_scaled = self.scaler.fit_transform(X)
        
  
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values  # convert to numpy array
        
  
        if isinstance(y, np.ndarray) and y.dtype.kind == 'f':  # if it is a float array
            y = y.astype(int)
        
        task_type = 'classification'  # force use of classification task type
        
        if self.method in ['kmeans', 'hierarchical']:
  
            from sklearn.metrics import silhouette_score
            scores = {'silhouette_score': silhouette_score(X_scaled, self.model.fit_predict(X_scaled))}
            return scores
        
  
        if task_type == 'classification':
            # **CRITICAL FIX: Handle mixed label types before cross_val_score**
            # Standardize labels to prevent mixed type errors
            import pandas as pd
            if isinstance(y, pd.Series):
                y_safe = y.astype(str)
            else:
                y_safe = np.array([str(label) for label in y])
            
            # Encode to integers for sklearn
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_safe)
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
            precision_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='precision_weighted')
            recall_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='recall_weighted')
            
            scores = {
                'accuracy': accuracy_scores.mean(),
                'f1_score': f1_scores.mean(),
                'precision': precision_scores.mean(),
                'recall': recall_scores.mean()
            }
        else:
  
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            from sklearn.metrics import make_scorer, mean_squared_error, r2_score
            rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
            
            # **CRITICAL FIX: Ensure y is numeric for regression**
            if isinstance(y, pd.Series):
                y_numeric = pd.to_numeric(y, errors='coerce')
            else:
                y_numeric = np.array([float(val) for val in y])
            
            rmse_scores = -cross_val_score(self.model, X_scaled, y_numeric, cv=cv, scoring=rmse_scorer)
            r2_scores = cross_val_score(self.model, X_scaled, y_numeric, cv=cv, scoring='r2')
            
            scores = {
                'rmse': rmse_scores.mean(),
                'r2': r2_scores.mean()
            }
        
        return scores
    
    def plot_dendrogram(self, X: np.ndarray) -> None:
        """
        Plot hierarchical clustering dendrogram
        
        Args:
            X: spectral data
        """
        if self.method != 'hierarchical':
            raise ValueError("Only hierarchical clustering methods support dendrograms")
            
        X_scaled = self.scaler.fit_transform(X)
        linkage_matrix = linkage(X_scaled, method='ward')
        
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('层次clustering树状图')
        plt.xlabel('samples索引')
        plt.ylabel('距离')
        plt.show()
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        获取model parameters
        
        Returns:
            Dict[str, Any]: model parameters
        """
        if self.model is None:
            return {}
            
        if self.method == 'kmeans':
            return {
                'n_clusters': self.model.n_clusters,
                'cluster_centers_': self.model.cluster_centers_,
                'labels_': self.model.labels_
            }
        elif self.method == 'hierarchical':
            return {
                'n_clusters': self.model.n_clusters,
                'labels_': self.model.labels_
            }
        elif self.method == 'rf':
            return {
                'feature_importances_': self.model.feature_importances_,
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth
            }
        else:
            return self.model.get_params() 