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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.utils.data_compatibility import standardize_classification_labels

class QualitativeAnalyzer:
    """Qualitative Analysis Class"""
    
    def __init__(self, method: str = 'lda'):
        """
        Initialize qualitative analyzer
        
        Args:
            method: Analysis method ('kmeans', 'hierarchical', 'lda', 'qda', 'svm', 'rf', 'knn', 'dt', 'nn', 'xgboost', 'lightgbm')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.task_type = 'classification'  # Explicitly mark task type as classification
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Train the model
        
        Args:
            X: Spectral data
            y: Labels (for supervised learning methods)
            **kwargs: Model parameters
        """
        # Check input data
        if X is None or X.size == 0:
            raise ValueError("Input data X cannot be empty")
            
        # Check and handle missing values
        if np.isnan(X).any():
            print("Warning: Input data contains missing values, using column mean to fill")
            # Use column mean to fill missing values
            X_filled = X.copy()
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    X_filled[mask, col] = np.mean(X[~mask, col])
            X = X_filled
            
        # Check feature standard deviation, avoid constant features
        feature_std = np.std(X, axis=0)
        if np.any(feature_std < 1e-10):
            print("Warning: Data contains near-constant features, which may affect analysis results")
            
        # Data standardization
        X_scaled = self.scaler.fit_transform(X)
        
        # Check y type, handle pandas.Series or DataFrame
        if y is not None:
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y = y.values  # Convert to numpy array
            elif not isinstance(y, np.ndarray):
                y = np.array(y)  # Ensure y is numpy array
        
        # Create optimized default parameters
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
            # Update user provided parameters
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
                raise ValueError("Supervised learning method requires label data")
                
            # More scientifically convert labels to format suitable for classification task
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
            # Save label encoder
            self.label_encoder = label_encoder
            
        elif self.method == 'svm':
            if y is None:
                raise ValueError("Supervised learning method requires label data")
                
            # More scientifically convert labels to format suitable for classification task
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Set optimized default parameters for SVM
            method_params = {
                'C': 10.0,           # Increase regularization parameters, better for high-dimensional data
                'kernel': 'rbf', 
                'gamma': 'auto',     # Change to auto, more suitable for spectral data
                'probability': True, 
                'random_state': 42,
                'class_weight': 'balanced',
                'tol': 1e-4,         # Improve precision
                'cache_size': 500    # Increase cache
            }
            method_params.update(kwargs)
            self.model = SVC(**method_params)
            self.model.fit(X_scaled, y_encoded)
            # Save label encoder
            self.label_encoder = label_encoder
            
        elif self.method == 'rf' or self.method.lower() == 'random forest':
            if y is None:
                raise ValueError("Supervised learning method requires label data")
                
            # More scientifically convert labels to format suitable for classification task
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
                
            # Set optimized default parameters for Random Forest
            method_params = {
                'n_estimators': 200,     # Increase number of trees
                'max_depth': None,       # Allow deeper trees
                'min_samples_split': 5,  # Prevent overfitting
                'min_samples_leaf': 2,   # Prevent overfitting
                'max_features': 'log2',  # Better for high-dimensional data
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
            # Save label encoder
            self.label_encoder = label_encoder
            
        elif self.method in ['knn', 'dt', 'nn', 'xgboost', 'lightgbm']:
            if y is None:
                raise ValueError("Supervised learning method requires label data")
                
            # More scientifically convert labels to format suitable for classification task
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            if self.method == 'knn':
                # Import KNN model
                from sklearn.neighbors import KNeighborsClassifier
                method_params = {
                    'n_neighbors': 7,        # Increase number of neighbors
                    'weights': 'distance', 
                    'algorithm': 'auto',
                    'leaf_size': 20,         # Reduce leaf size, improve precision
                    'p': 2,
                    'metric': 'minkowski',
                    'n_jobs': -1
                }
                method_params.update(kwargs)
                self.model = KNeighborsClassifier(**method_params)
                
            elif self.method == 'dt':
                # Import Decision Tree model
                from sklearn.tree import DecisionTreeClassifier
                method_params = {
                    'max_depth': 15,         # Increase maximum depth
                    'random_state': 42, 
                    'class_weight': 'balanced',
                    'criterion': 'gini',
                    'splitter': 'best',
                    'min_samples_split': 5,  # Prevent overfitting
                    'min_samples_leaf': 2,   # Prevent overfitting
                    'max_features': 'sqrt',  # Feature subset selection
                    'max_leaf_nodes': None
                }
                method_params.update(kwargs)
                self.model = DecisionTreeClassifier(**method_params)
                
            elif self.method == 'nn':
                # Use MLPClassifier (Multi-Layer Perceptron)
                try:
                    from sklearn.neural_network import MLPClassifier
                    method_params = {
                        'hidden_layer_sizes': (200, 100),  # Increase network complexity
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.001,                    # Increase regularization
                        'batch_size': 'auto',
                        'learning_rate': 'adaptive',
                        'learning_rate_init': 0.01,        # Increase learning rate
                        'max_iter': 500,                   # Increase maximum iterations
                        'shuffle': True,
                        'random_state': 42,
                        'tol': 1e-6,                       # Improve convergence precision
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
                    print("Warning: Neural network model not installed, using RandomForest as alternative")
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
                # Try to import XGBoost
                try:
                    import xgboost as xgb
                    method_params = {
                        'max_depth': 8,                    # Increase depth
                        'learning_rate': 0.05,             # Lower learning rate, more stable
                        'n_estimators': 200,               # Increase number of trees
                        'objective': 'multi:softprob',
                        'booster': 'gbtree',
                        'tree_method': 'auto',
                        'n_jobs': -1,
                        'gamma': 0.1,                      # Increase gamma, prevent overfitting
                        'min_child_weight': 3,             # Increase min_child_weight
                        'max_delta_step': 0,
                        'subsample': 0.9,                  # Increase subsample
                        'colsample_bytree': 0.9,           # Increase colsample_bytree
                        'colsample_bylevel': 1,
                        'reg_alpha': 0.1,                  # Increase L1 regularization
                        'reg_lambda': 1.0,                 # Increase L2 regularization
                        'scale_pos_weight': 1,
                        'random_state': 42,
                        'missing': None
                    }
                    method_params.update(kwargs)
                    # Set num_class parameter
                    n_classes = len(np.unique(y_encoded))
                    if n_classes > 2:
                        method_params['num_class'] = n_classes
                    self.model = xgb.XGBClassifier(**method_params)
                except ImportError:
                    print("Warning: XGBoost not installed, using RandomForest as alternative")
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
                # Try to import LightGBM
                try:
                    import lightgbm as lgb
                    method_params = {
                        'boosting_type': 'gbdt',
                        'num_leaves': 63,                  # Increase number of leaves
                        'max_depth': 8,                    # Set maximum depth
                        'learning_rate': 0.05,             # Lower learning rate
                        'n_estimators': 200,               # Increase number of estimators
                        'subsample_for_bin': 200000,
                        'objective': 'multiclass',
                        'class_weight': 'balanced',
                        'min_split_gain': 0.1,             # Increase split gain threshold
                        'min_child_weight': 0.01,          # Increase child weight
                        'min_child_samples': 10,           # Reduce number of samples
                        'subsample': 0.9,                  # Increase subsample
                        'subsample_freq': 1,               # Enable subsample_freq
                        'colsample_bytree': 0.9,           # Increase feature sampling
                        'reg_alpha': 0.1,                  # Increase L1 regularization
                        'reg_lambda': 1.0,                 # Increase L2 regularization
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbosity': -1                    # Reduce output
                    }
                    method_params.update(kwargs)
                    # Set num_class parameter
                    n_classes = len(np.unique(y_encoded))
                    if n_classes > 2:
                        method_params['num_class'] = n_classes
                    self.model = lgb.LGBMClassifier(**method_params)
                except ImportError:
                    print("Warning: LightGBM not installed, using RandomForest as alternative")
                    method_params = {
                        'n_estimators': 100, 
                        'max_depth': None, 
                        'random_state': 42,
                        'class_weight': 'balanced',
                        'n_jobs': -1
                    }
                    method_params.update(kwargs)
                    self.model = RandomForestClassifier(**method_params)
            
            # Train model
            self.model.fit(X_scaled, y_encoded)
            # Save label encoder
            self.label_encoder = label_encoder
            
        else:
            raise ValueError(f"Unsupported analysis method: {self.method}")
        
        # Add task_type attribute to model
        if self.model is not None:
            setattr(self.model, 'task_type', 'classification')
            
        # Print model parameter information for debugging
        print(f"Model {self.method} training completed, parameters: {self.model.get_params() if hasattr(self.model, 'get_params') else 'Cannot be retrieved'}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class
        
        Args:
            X: Spectral data
            
        Returns:
            np.ndarray: Predicted class
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Check and handle missing values
        if np.isnan(X).any():
            print("Warning: Predictive data contains missing values, using column mean to fill")
            # Use column mean to fill missing values
            X_filled = X.copy()
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    X_filled[mask, col] = np.mean(X[~mask, col])
            X = X_filled
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # If there is a label encoder, convert back to original labels
        if hasattr(self, 'label_encoder') and self.method not in ['kmeans', 'hierarchical']:
            return self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, **kwargs) -> Dict:
        """
        Execute cross validation and return evaluation metrics
        
        Args:
            X: Spectral data
            y: Labels
            n_splits: Number of splits for cross validation
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        # Import standardized label processing
        from app.utils.data_compatibility import standardize_classification_labels
            
        # Data standardization
        X_scaled = self.scaler.fit_transform(X)
        
        # Check y type, handle pandas.Series or DataFrame
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values  # Convert to numpy array
        
        # Ensure y is numpy array
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Convert all types of labels to string, handle uniformly
        y_str = np.array([str(label) for label in y])
        
        task_type = 'classification'  # Force using classification task type
        
        if self.method in ['kmeans', 'hierarchical']:
            # Unsupervised methods are not suitable for standard cross validation
            from sklearn.metrics import silhouette_score
            scores = {'silhouette_score': silhouette_score(X_scaled, self.model.fit_predict(X_scaled))}
            return scores
        
        # For supervised learning methods execute cross validation
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # **CRITICAL FIX: Encode string labels to integers before cross_val_predict**
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_str)
            
            # Use cross_val_predict to get predictions for each fold
            y_pred = cross_val_predict(self.model, X_scaled, y_encoded, cv=cv)
            
            # Use standardization function to handle mixed type labels
            y_true_idx, y_pred_idx, _ = standardize_classification_labels(y_str, y_pred)
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_true_idx, y_pred_idx)
            f1 = f1_score(y_true_idx, y_pred_idx, average='weighted')
            precision = precision_score(y_true_idx, y_pred_idx, average='weighted')
            recall = recall_score(y_true_idx, y_pred_idx, average='weighted')
            
            scores = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            }
        else:
            # This code will never execute, but keep it just in case
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            from sklearn.metrics import make_scorer, mean_squared_error, r2_score
            rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
            
            # **CRITICAL FIX: Should not perform numeric conversion in qualitative analyzer**
            # If code reaches here, it indicates a logic error, should throw exception
            raise ValueError("Qualitative analyzer should not perform regression tasks! This indicates a logic error.")
            
            # The following code will never execute, but kept just in case
            # if isinstance(y, pd.Series):
            #     y_numeric = pd.to_numeric(y, errors='coerce')
            # else:
            #     y_numeric = np.array([float(val) for val in y])
            
            # Use cross_val_predict to get predictions
            # y_pred = cross_val_predict(self.model, X_scaled, y_numeric, cv=cv)
            
            # Calculate metrics
            # rmse = np.sqrt(mean_squared_error(y, y_pred))
            # r2 = r2_score(y, y_pred)
            
            # scores = {
            #     'rmse': float(rmse),
            #     'r2': float(r2)
            # }
        
        return scores
    
    def plot_dendrogram(self, X: np.ndarray) -> None:
        """
        Plot hierarchical clustering dendrogram
        
        Args:
            X: Spectral data
        """
        if self.method != 'hierarchical':
            raise ValueError("Only hierarchical clustering method supports dendrograms")
            
        X_scaled = self.scaler.fit_transform(X)
        linkage_matrix = linkage(X_scaled, method='ward')
        
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dict[str, Any]: Model parameters
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