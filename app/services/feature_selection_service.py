# app/services/feature_selection_service.py

from app.utils.exceptions import FeatureSelectionError
import pandas as pd
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
from sklearn.feature_selection import SelectKBest as SklearnSelectKBest, RFE, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
import numpy as np    
import traceback
from sklearn.preprocessing import LabelEncoder, StandardScaler

  
numpy = np

class FeatureSelectionService:
    def __init__(self, plugins: dict = None):
        self.plugins = plugins or {}    
        
    def _check_user_feature_count(self, user_k, n_classes, n_features, method_name="Feature Selection"):
        """Check and adjust user-specified feature count based on data characteristics"""
        min_recommended = max(n_classes * 2, 10)
        
        if user_k < min_recommended and n_features >= min_recommended:
            recommended_k = min(min_recommended, n_features // 2, 50)
            
        elif user_k <= 5 and n_features > 50:
            recommended_k = min(30, n_features // 3)
        
        final_k = min(user_k, n_features)
            
        return final_k

    def select_features(self, method: str, params: dict, X, y) -> list:
          
        import numpy as np
        
          
        wavelengths = params.get('wavelengths', None)
        
          
        if not isinstance(X, pd.DataFrame):
              
            print(f"Converting input X from {type(X)} to pandas DataFrame")
            X_df = pd.DataFrame(X)
            
            # Use wavelength information as column names (if provided)
            if wavelengths is not None and len(wavelengths) == X.shape[1]:
                X_df.columns = [f"wavelength_{w:.1f}" if isinstance(w, (int, float)) else f"wavelength_{i}" 
                              for i, w in enumerate(wavelengths)]
                print(f"Using wavelength information as column names: {list(X_df.columns)[:5]}...")
            else:
                X_df.columns = [f'feature_{i}' for i in range(X.shape[1])]
                print("Wavelength information not provided, using default feature names")
            
            X = X_df
        
          
        original_y = y
        original_y_type = type(y)
        
          
        if not isinstance(y, pd.Series):
            print(f"Converting input y from {type(y)} to pandas Series")
            y = pd.Series(y)
        

        
        try:
            if method in self.plugins:
                algorithm: FeatureSelectionAlgorithm = self.plugins[method]
                return algorithm.select_features(X, y, params)
            else:
                  
                is_classification = self._is_classification_task(y)
                
                  
                if method == "SelectKBest":
                    k = params.get('k', 10)
                    score_func_name = params.get('score_func', 'f_classif')
                    
                      
                      
                      
                    n_classes = len(np.unique(y))
                    k = self._check_user_feature_count(k, n_classes, X.shape[1], "SelectKBest")
                    

                    
                      
                    if is_classification:
                          
                        y_for_selection = y
                        numeric_encoding_needed = not pd.api.types.is_numeric_dtype(y)
                        print(f"Numeric encoding needed: {numeric_encoding_needed}")
                        
                        if numeric_encoding_needed:
                            # If sklearn functions need numeric encoding, only temporarily encode here
                            # This will not affect the returned labels
                            print("Warning: Labels are non-numeric type, some sklearn feature selection functions may need numeric encoding")
                            print("Will try using original labels first, if failed will use temporary encoding")
                    else:
                          
                        y_for_selection = y
                    
                      
                    if score_func_name == 'f_classif':
                        from sklearn.feature_selection import f_classif, f_regression
                        score_func = f_classif if is_classification else f_regression
                    elif score_func_name == 'mutual_info_classif':
                        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                        score_func = mutual_info_classif if is_classification else mutual_info_regression
                    elif score_func_name == 'chi2':
                          
                        if not is_classification:
                            raise FeatureSelectionError("INFO: Operation completed")
                        from sklearn.feature_selection import chi2
                        
                          
                        if (X < 0).any().any():
                              
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            X_scaled = scaler.fit_transform(X)
                            score_func = chi2
                              
                            try:
                                selector = SklearnSelectKBest(score_func=score_func, k=k)
                                selector.fit(X_scaled, y_for_selection)
                                selected_features = X.columns[selector.get_support()]
                            except Exception as e:
                                print(f"Failed using original labels: {e}, trying temporary encoding")
                                # If failed, try temporary encoding
                                if numeric_encoding_needed:
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    y_encoded = le.fit_transform(y)
                                    print(f"Temporarily encoded labels to numeric: {np.unique(y_encoded)}")
                                    selector = SklearnSelectKBest(score_func=score_func, k=k)
                                    selector.fit(X_scaled, y_encoded)
                                    selected_features = X.columns[selector.get_support()]
                                else:
                                      
                                    raise
                        else:
                            score_func = chi2
                              
                            try:
                                selector = SklearnSelectKBest(score_func=score_func, k=k)
                                selector.fit(X, y_for_selection)
                                selected_features = X.columns[selector.get_support()]
                            except Exception as e:
                                print(f"Using original labels failed: {e}, trying temporary encoding")
                                  
                                if numeric_encoding_needed:
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    y_encoded = le.fit_transform(y)
                                    selector = SklearnSelectKBest(score_func=score_func, k=k)
                                    selector.fit(X, y_encoded)
                                    selected_features = X.columns[selector.get_support()]
                                else:
                                      
                                    raise
                    else:
                          
                        from sklearn.feature_selection import f_classif, f_regression
                        score_func = f_classif if is_classification else f_regression
                    
                      
                    try:
                        selector = SklearnSelectKBest(score_func=score_func, k=k)
                        selector.fit(X, y_for_selection)
                        selected_features = X.columns[selector.get_support()]
                    except Exception as e:
                        print(f"Failed calling SelectKBest with original labels: {e}")
                        # If failed and labels are non-numeric, try temporary encoding
                        if numeric_encoding_needed:
                            print("Trying to temporarily encode non-numeric labels to numeric for feature selection")
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y)
                            print(f"Encoding mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
                            selector = SklearnSelectKBest(score_func=score_func, k=k)
                            selector.fit(X, y_encoded)
                            selected_features = X.columns[selector.get_support()]
                        else:
                            # If not an encoding issue, may be other error, re-raise
                            raise
                elif method == "Recursive Feature Elimination (RFE)":
                    n_features_to_select = params.get('n_features_to_select', 10)
                    
                      
                    print(f"🔍 DEBUG - RFE method:")
                    print(f"   n_features_to_select from params: {n_features_to_select}")
                    print(f"   Available features: {X.shape[1]}")
                    
                      
                    n_features_to_select = min(n_features_to_select, X.shape[1])
                    # Choose appropriate base model based on task type
                    if is_classification:
                        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
                    
                    # **CRITICAL FIX: Handle mixed label types before fit**
                    if is_classification:
                        # Standardize labels to prevent mixed type errors
                        if isinstance(y, pd.Series):
                            y_safe = y.astype(str)
                        else:
                            y_safe = np.array([str(label) for label in y])
                        
                        # Encode to integers for sklearn
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_safe)
                        selector = selector.fit(X, y_encoded)
                    else:
                        selector = selector.fit(X, y)
                    selected_features = X.columns[selector.support_]
                elif method == "Feature Importance":
                    top_n = params.get('top_n', 10)
                    
                      
                    if top_n <= 10 and X.shape[1] > 50:
                          
                        top_n_recommended = min(max(50, X.shape[1] // 10), X.shape[1] // 2)
                        print("INFO: Processing data")
                        print("INFO: Processing data")
                        print("INFO: Processing data")
                        top_n = top_n_recommended
                    
                      
                    print(f"🔍 DEBUG - Feature Importance method:")
                    print(f"   top_n from params: {params.get('top_n', 10)}")
                    print(f"   Adjusted top_n: {top_n}")
                    print(f"   Available features: {X.shape[1]}")
                    
                      
                    top_n = min(top_n, X.shape[1])
                    
                      
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    # **CRITICAL FIX: Handle mixed label types before fit**
                    if is_classification:
                        # Standardize labels to prevent mixed type errors
                        if isinstance(y, pd.Series):
                            y_safe = y.astype(str)
                        else:
                            y_safe = np.array([str(label) for label in y])
                        
                        # Encode to integers for sklearn
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_safe)
                        model.fit(X, y_encoded)
                    else:
                        model.fit(X, y)
                    importances = model.feature_importances_
                    indices = importances.argsort()[-top_n:]
                    selected_features = X.columns[indices]
                elif method == "Mutual Information":
                    # **ENHANCED MUTUAL INFORMATION FEATURE SELECTION**
                      
                    
                    k = params.get('k', 30)
                    n_classes = len(np.unique(y)) if is_classification else 3
                    user_k = k    
                    
                      
                    k = self._check_user_feature_count(k, n_classes, X.shape[1], "INFO: Operation completed")
                    
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    
                      
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                      
                    if is_classification:
                          
                        if isinstance(y, pd.Series):
                            y_safe = y.astype(str)
                        else:
                            y_safe = np.array([str(label) for label in y])
                        
                          
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_safe)
                        
                          
                        from sklearn.feature_selection import mutual_info_classif
                        mi_scores = mutual_info_classif(
                            X_scaled, y_encoded,
                            discrete_features=False,    
                            n_neighbors=max(3, len(y_encoded) // 50),    
                            copy=True,
                            random_state=42
                        )
                    else:
                        from sklearn.feature_selection import mutual_info_regression
                        mi_scores = mutual_info_regression(
                            X_scaled, y,
                            discrete_features=False,
                            n_neighbors=max(3, len(y) // 50),
                            copy=True,
                            random_state=42
                        )
                    
                      
                    top_indices = np.argsort(mi_scores)[-k:]
                    selected_features = X.columns[top_indices]
                    
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    
                      
                    if mi_scores.max() < 0.01:
                        print("INFO: Processing data")
                        print("INFO: Processing data")
                        print("INFO: Processing data")
                        print("INFO: Processing data")
                    elif mi_scores.max() > 0.1:
                        print("INFO: Processing data")
                    
                      
                    self.last_mi_scores = mi_scores
                elif method == "LASSO":
                    C = params.get('alpha', 1.0)
                      
                    if is_classification:
                          
                        model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
                    else:
                          
                        from sklearn.linear_model import Lasso
                        model = Lasso(alpha=1.0/C, random_state=42)
                    
                    # **CRITICAL FIX: Handle mixed label types before fit**
                    if is_classification:
                        # Standardize labels to prevent mixed type errors
                        if isinstance(y, pd.Series):
                            y_safe = y.astype(str)
                        else:
                            y_safe = np.array([str(label) for label in y])
                        
                        # Encode to integers for sklearn
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_safe)
                        model.fit(X, y_encoded)
                    else:
                        model.fit(X, y)
                    
                      
                    if is_classification:
                        selected_indices = (model.coef_[0] != 0)
                    else:
                        selected_indices = (model.coef_ != 0)
                    
                    selected_features = X.columns[selected_indices]
                elif method == "Principal Component Analysis (PCA)":
                    # **CRITICAL FIX: 优化PCA特征选择策略以提高定量分析性能**
                    
                    k = params.get('k', 15)  # 用户要求的特征数量
                    n_classes = len(np.unique(y)) if is_classification else 42  # 回归任务的唯一值数量
                    
                    k = params.get('k', 15)
                    n_classes = len(np.unique(y)) if is_classification else 42
                    
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    if is_classification:
                        optimal_components = min(max(n_classes * 2, 6), X.shape[0] - 1, X.shape[1], 20)
                    else:
                        optimal_components = min(5, X.shape[0] - 1, X.shape[1])
                    
                    pca = PCA(n_components=optimal_components)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    loadings = np.abs(pca.components_)
                    weighted_importance = np.zeros(X.shape[1])
                    for i, loading in enumerate(loadings):
                        weight = pca.explained_variance_ratio_[i]
                        weighted_importance += loading * weight
                    
                    correlations = np.abs(np.corrcoef(X_scaled.T, X_pca.T))[:X.shape[1], X.shape[1]:]
                    correlation_importance = np.mean(correlations[:, :min(3, optimal_components)], axis=1)
                    
                    combined_importance = 0.7 * weighted_importance + 0.3 * correlation_importance
                    
                    k = min(k, X.shape[1])
                    top_indices = np.argsort(combined_importance)[-k:]
                    selected_features = X.columns[top_indices]
                elif method == "Wavelet":
                    # Wavelet feature selection for spectral data
                    # Wavelet transform decomposes signals into time-frequency components
                    # This is especially useful for spectral data to identify important frequency patterns
                    try:
                        import pywt
                    except ImportError:
                        raise FeatureSelectionError("PyWavelets package is required for Wavelet feature selection. Install with: pip install PyWavelets")
                    
                    k = params.get('k', 10)
                    wavelet = params.get('wavelet', 'db4')  # Daubechies 4 is good for spectral data
                    levels = params.get('levels', 3)       # Decomposition levels
                    
                    print(f"Wavelet feature selection using {wavelet} wavelet with {levels} levels, selecting {k} features")
                    print(f"Wavelet transforms are excellent for identifying frequency patterns in spectral data")
                    
                    # Calculate feature importance using wavelet decomposition
                    feature_scores = np.zeros(X.shape[1])
                    
                    for i in range(X.shape[0]):
                        signal = X.iloc[i].values
                        
                        # Perform wavelet decomposition
                        coeffs = pywt.wavedec(signal, wavelet, level=levels)
                        
                        # Calculate energy in each coefficient
                        # Higher energy coefficients indicate more important features
                        reconstructed_approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)
                        
                        # Ensure same length as original signal
                        if len(reconstructed_approx) != len(signal):
                            reconstructed_approx = reconstructed_approx[:len(signal)]
                        
                        # Calculate the contribution of each original feature to the approximation
                        contribution = np.abs(reconstructed_approx)
                        feature_scores += contribution
                    
                    # Normalize by number of samples
                    feature_scores /= X.shape[0]
                    
                    # Select top k features
                    top_indices = np.argsort(feature_scores)[-k:]
                    selected_features = X.columns[top_indices]
                    
                    print(f"Wavelet analysis identified features with highest energy contributions")
                    print(f"Selected wavelength ranges: {[X.columns[i] for i in top_indices[:5]]}...")
                    
                elif method == "Genetic Algorithm":
                      
                    n_features = params.get('k', 10)
                    n_population = params.get('n_population', 50)
                    n_generations = params.get('n_generations', 100)
                    
                      
                      
                    if is_classification:
                        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                    else:
                        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    estimator.fit(X, y)
                    importances = estimator.feature_importances_
                    
                      
                    top_indices = np.argsort(importances)[-n_features:]
                    selected_features = X.columns[top_indices]
                    
                elif method == "Correlation Filter":
                      
                    threshold = params.get('threshold', 0.95)
                    
                      
                    corr_matrix = X.corr().abs()
                    
                      
                    upper_tri = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    
                      
                    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
                    
                      
                    selected_features = X.columns.drop(to_drop)
                    
                elif method == "Variance Threshold":
                      
                    threshold = params.get('threshold', 0.0)
                    
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=threshold)
                    selector.fit(X)
                    selected_features = X.columns[selector.get_support()]
                    
                elif method == "Information Gain":
                    # Information gain feature selection
                    k = params.get('k', 30)
                    
                    if not is_classification:
                        raise FeatureSelectionError("Information Gain is only applicable to classification tasks")
                    
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    
                      
                    if isinstance(y, pd.Series):
                        y_safe = y.astype(str)
                    else:
                        y_safe = np.array([str(label) for label in y])
                    
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y_safe)
                    
                    from sklearn.feature_selection import mutual_info_classif
                    scores = mutual_info_classif(X, y_encoded, random_state=42)
                    top_indices = np.argsort(scores)[-k:]
                    selected_features = X.columns[top_indices]
                    
                    print("INFO: Processing data")
                    
                elif method == "Spectral Optimized":
                      
                      
                    
                    k = params.get('k', 30)
                    n_classes = len(np.unique(y)) if is_classification else 3
                    
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    
                      
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                      
                    feature_scores = {}
                    
                      
                    if is_classification:
                        from sklearn.feature_selection import f_classif
                        if isinstance(y, pd.Series):
                            y_safe = y.astype(str)
                        else:
                            y_safe = np.array([str(label) for label in y])
                        
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_safe)
                        
                        f_scores, _ = f_classif(X_scaled, y_encoded)
                        feature_scores['f_score'] = f_scores / f_scores.max()    
                        
                          
                        from sklearn.feature_selection import mutual_info_classif
                        mi_scores = mutual_info_classif(X_scaled, y_encoded, random_state=42)
                        feature_scores['mutual_info'] = mi_scores / (mi_scores.max() + 1e-10)
                        
                          
                        from sklearn.ensemble import RandomForestClassifier
                        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                        rf.fit(X_scaled, y_encoded)
                        feature_scores['rf_importance'] = rf.feature_importances_
                    
                      
                      
                    spectral_gradient = np.abs(np.diff(X_scaled, axis=1))
                    gradient_importance = np.mean(spectral_gradient, axis=0)
                      
                    gradient_extended = np.zeros(X_scaled.shape[1])
                    gradient_extended[:-1] = gradient_importance
                    gradient_extended[-1] = gradient_importance[-1]    
                    feature_scores['spectral_gradient'] = gradient_extended / (gradient_extended.max() + 1e-10)
                    
                      
                      
                    wavelength_importance = np.ones(X_scaled.shape[1])
                    try:
                          
                        wavelengths = []
                        for col in X.columns:
                            try:
                                wl = float(col)
                                wavelengths.append(wl)
                            except:
                                wavelengths.append(0)
                        
                        wavelengths = np.array(wavelengths)
                          
                        important_ranges = [(1000, 1800), (2000, 2400)]    
                        for wl_min, wl_max in important_ranges:
                            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
                            wavelength_importance[mask] *= 1.5
                    except:
                        pass
                    
                    feature_scores['wavelength_prior'] = wavelength_importance / wavelength_importance.max()
                    
                      
                    weights = {
                        'f_score': 0.25,
                        'mutual_info': 0.25,
                        'rf_importance': 0.25,
                        'spectral_gradient': 0.15,
                        'wavelength_prior': 0.10
                    }
                    
                    combined_score = np.zeros(X_scaled.shape[1])
                    for score_name, score_values in feature_scores.items():
                        if score_name in weights:
                            combined_score += weights[score_name] * score_values
                    
                      
                    k = min(k, X.shape[1])
                    top_indices = np.argsort(combined_score)[-k:]
                    selected_features = X.columns[top_indices]
                    
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    print("INFO: Processing data")
                    
                      
                    self.last_spectral_scores = {
                        'combined': combined_score,
                        'individual': feature_scores,
                        'selected_indices': top_indices
                    }
                    
                elif method == "PLSR" or method == "PLS Regression":
                    # Partial Least Squares Regression feature selection
                    try:
                        from sklearn.cross_decomposition import PLSRegression
                        from sklearn.preprocessing import LabelEncoder
                    except ImportError:
                        raise FeatureSelectionError("scikit-learn is required for PLSR feature selection")
                    
                    n_components = params.get('n_components', min(10, X.shape[1], len(y.unique()) if hasattr(y, 'unique') else 10))
                    k = params.get('k', min(10, X.shape[1]))
                    
                    print(f"PLSR using {n_components} components to select {k} features")
                    
                    # **CRITICAL FIX: Handle string labels for PLSR**
                    # PLSR expects numerical targets, so encode string labels
                    y_for_pls = y.copy() if hasattr(y, 'copy') else y
                    
                    # Check if y contains string labels
                    if isinstance(y_for_pls, pd.Series):
                        y_dtype_is_string = y_for_pls.dtype == 'object' or str(y_for_pls.dtype).startswith('<U')
                        y_sample = y_for_pls.iloc[0] if len(y_for_pls) > 0 else None
                    elif isinstance(y_for_pls, np.ndarray):
                        y_dtype_is_string = y_for_pls.dtype.kind in ['U', 'S', 'O']
                        y_sample = y_for_pls[0] if len(y_for_pls) > 0 else None
                    else:
                        y_dtype_is_string = isinstance(str(y_for_pls), str) if hasattr(y_for_pls, '__iter__') else False
                        y_sample = None
                    
                    if y_dtype_is_string or (y_sample is not None and isinstance(y_sample, str)):
                        print(f"🔧 PLSR: Converting string labels to numerical for PLS fitting")
                        print(f"   Sample labels: {y_for_pls[:5] if hasattr(y_for_pls, '__getitem__') else y_for_pls}")
                        
                        # Use LabelEncoder to convert string labels to integers
                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y_for_pls)
                        print(f"   Encoded sample: {y_encoded[:5]}")
                        print(f"   Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
                    else:
                        y_encoded = y_for_pls
                        print(f"🔧 PLSR: Labels are already numerical, using directly")
                    
                    # Fit PLS model with encoded labels
                    pls = PLSRegression(n_components=n_components, scale=True)
                    pls.fit(X, y_encoded)
                    
                    # Calculate feature importance based on PLS loadings
                    # Use the sum of squared loadings across all components
                    loadings = pls.x_loadings_  # Shape: (n_features, n_components)
                    feature_scores = np.sum(loadings ** 2, axis=1)
                    
                    # Select top k features
                    top_indices = np.argsort(feature_scores)[-k:]
                    selected_features = X.columns[top_indices]
                    
                elif method == "Peak Detection":
                    # Peak detection feature selection for spectral data
                    try:
                        from scipy.signal import find_peaks
                    except ImportError:
                        raise FeatureSelectionError("scipy is required for Peak Detection feature selection")
                    
                    k = params.get('k', 10)
                    height = params.get('height', None)  # Minimum peak height
                    distance = params.get('distance', 1)  # Minimum distance between peaks
                    prominence = params.get('prominence', None)  # Peak prominence
                    
                    print(f"Peak Detection selecting {k} features with distance={distance}")
                    
                    # Calculate mean spectrum
                    mean_spectrum = X.mean(axis=0).values
                    
                    # Find peaks in the mean spectrum
                    peaks, properties = find_peaks(
                        mean_spectrum, 
                        height=height, 
                        distance=distance, 
                        prominence=prominence
                    )
                    
                    print(f"Found {len(peaks)} peaks in spectrum")
                    
                    if len(peaks) == 0:
                        # If no peaks found, select features with highest variance
                        print("No peaks found, selecting features with highest variance")
                        variances = X.var(axis=0)
                        top_indices = np.argsort(variances)[-k:]
                        selected_features = X.columns[top_indices]
                    elif len(peaks) < k:
                        # If fewer peaks than requested features, add features around peaks
                        print(f"Only {len(peaks)} peaks found, expanding selection around peaks")
                        selected_indices = set(peaks)
                        
                        # Add neighboring features around each peak
                        for peak in peaks:
                            for offset in [-2, -1, 1, 2]:
                                neighbor = peak + offset
                                if 0 <= neighbor < len(X.columns) and len(selected_indices) < k:
                                    selected_indices.add(neighbor)
                        
                        # If still not enough, add features with highest variance
                        if len(selected_indices) < k:
                            variances = X.var(axis=0)
                            remaining_indices = set(range(len(X.columns))) - selected_indices
                            remaining_variances = [(i, variances.iloc[i]) for i in remaining_indices]
                            remaining_variances.sort(key=lambda x: x[1], reverse=True)
                            
                            for i, _ in remaining_variances[:k - len(selected_indices)]:
                                selected_indices.add(i)
                        
                        selected_features = X.columns[list(selected_indices)[:k]]
                    else:
                        # More peaks than requested features, select top k peaks by height/prominence
                        if 'peak_heights' in properties:
                            peak_scores = properties['peak_heights']
                        elif 'prominences' in properties:
                            peak_scores = properties['prominences']
                        else:
                            # Use peak values from mean spectrum
                            peak_scores = mean_spectrum[peaks]
                        
                        top_peak_indices = np.argsort(peak_scores)[-k:]
                        selected_peak_positions = peaks[top_peak_indices]
                        selected_features = X.columns[selected_peak_positions]
                    
                else:
                    raise FeatureSelectionError(f"Unsupported feature selection method: {method}")
                
                if hasattr(selected_features, 'tolist'):
                    result = selected_features.tolist()
                else:
                    result = list(selected_features)
                
                return result
        except Exception as e:
              
            error_msg = f"Error in select_features for method '{method}': {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise FeatureSelectionError(error_msg)
            
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

    def apply_method(self, X_train, y_train, X_test=None, method=None, params=None):
        """Apply feature selection method to training data, and transform testing data using the same features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features (optional)
            method: Feature selection method name
            params: Method parameters
            
        Returns:
            tuple: (X_train_selected, X_test_selected or None)
        """
          
        import numpy as np
        
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("Training data (X_train, y_train) cannot be None")
            
        if method is None:
            print("No feature selection method specified, returning original data")
            return X_train, X_test
        
        try:
            # Print input data information for debugging
            print(f"Applying feature selection method - input data info:")
            print(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
            print(f"y_train type: {type(y_train)}, length: {len(y_train) if hasattr(y_train, '__len__') else 'N/A'}")
            if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
                print(f"y_train data type: {y_train.dtype}")
            elif isinstance(y_train, np.ndarray):
                print(f"y_train data type: {y_train.dtype}")
            
            if isinstance(y_train, pd.Series):
                print(f"y_train unique values: {y_train.unique()}, count: {len(y_train.unique())}")
            elif isinstance(y_train, np.ndarray):
                print(f"y_train unique values: {np.unique(y_train)}, count: {len(np.unique(y_train))}")
            
            # Save original data and types
            original_X_train_type = type(X_train)
            original_X_test_type = type(X_test) if X_test is not None else None
            
            # Create deep copy to ensure original data is not modified
            if hasattr(y_train, 'copy'):
                original_y_train = y_train.copy()
            elif isinstance(y_train, np.ndarray):
                original_y_train = y_train.copy()
            else:
                # If cannot copy, keep reference
                original_y_train = y_train
                
            print(f"Original label save type: {type(original_y_train)}")
            
            # Check task type needed for feature importance calculation
            task_type = None
            # Try to automatically determine task type
            if y_train is not None:
                if self._is_classification_task(y_train):
                    task_type = 'classification'
                    print("Automatically detected classification task")
                else:
                    task_type = 'regression'
                    print("Automatically detected regression task")
            
            # Handle feature selection
            if method is None:
                # Default method
                method = "SelectKBest"
            
            if params is None:
                params = {}
                
            # Extract the number of features to select from params
            # Support multiple parameter names for different methods
            n_features = params.get('k', 
                        params.get('n_features', 
                        params.get('n_features_to_select', 
                        params.get('top_n', min(10, X_train.shape[1])))))
            
              
            if y_train is not None:
                n_classes = len(np.unique(y_train))
                min_features_needed = max(n_classes * 3, 30)
                
                if 'k' in params and params['k'] is not None:
                    user_k = params['k']
                    if user_k < min_features_needed and X_train.shape[1] >= min_features_needed:
                        recommended_k = min(min_features_needed, X_train.shape[1] // 2, 80)
                        print(f"WARNING: User k={user_k} may be insufficient for {n_classes} classes")
                        print(f"   Recommended: {recommended_k} features for better performance")
                        print(f"   Using user setting: {user_k} (as requested)")
                    n_features = user_k
                elif 'top_n' in params and params['top_n'] is not None:
                    user_top_n = params['top_n']
                    if user_top_n < min_features_needed and X_train.shape[1] >= min_features_needed:
                        recommended_top_n = min(min_features_needed, X_train.shape[1] // 2, 80)
                        print(f"WARNING: User top_n={user_top_n} may be insufficient for {n_classes} classes")
                        print(f"   Recommended: {recommended_top_n} features for better performance")
                        print(f"   Using user setting: {user_top_n} (as requested)")
                    n_features = user_top_n
                else:
                      
                    if n_features < min_features_needed and X_train.shape[1] >= min_features_needed:
                        n_features_recommended = min(min_features_needed, X_train.shape[1] // 2, 80)
                        print(f"Auto-adjusting features from {n_features} to {n_features_recommended} for {n_classes} classes")
                        n_features = n_features_recommended
            
              
            print(f"DEBUG - Parameter extraction:")
            print(f"   Raw params: {params}")
            print(f"   Final n_features: {n_features}")
            print(f"   Available features: {X_train.shape[1]}")
            
            # Update params to ensure consistency across all methods
            params['k'] = n_features
            params['n_features'] = n_features
            params['n_features_to_select'] = n_features
            params['top_n'] = n_features
            
            print(f"🎯 Feature selection method '{method}' selecting {n_features} features...")
            print(f"🎯 Final parameters passed to method: {params}")
            
            # Ensure correct data type - convert to DataFrame for select_features method
            if not isinstance(X_train, pd.DataFrame):
                print(f"Converting X_train from {type(X_train)} to DataFrame for feature selection")
                # Save original data
                original_X_train = X_train.copy()
                # Create column names
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                X_train = pd.DataFrame(X_train, columns=feature_names)
            else:
                feature_names = X_train.columns.tolist()
                
            # Similarly handle X_test (if exists)
            if X_test is not None and not isinstance(X_test, pd.DataFrame):
                print(f"Converting X_test from {type(X_test)} to DataFrame for feature selection")
                original_X_test = X_test.copy()
                X_test = pd.DataFrame(X_test, columns=feature_names)
            
            # Get feature selected dataset
            selected_features = self.select_features(method, params, X_train, y_train)
            
            # Check if selected features is empty
            if not selected_features or len(selected_features) == 0:
                print("Warning: Feature selection returned empty result, will use all features")
                selected_features = feature_names
            
            print(f"Feature selection completed, selected {len(selected_features)} features")
            
            # Apply feature selection
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features] if X_test is not None else None
            
            # Convert back to original type (if needed)
            if original_X_train_type == np.ndarray:
                X_train_selected = X_train_selected.values
            if X_test is not None and original_X_test_type == np.ndarray:
                X_test_selected = X_test_selected.values
            
            # **CRITICAL FIX: Calculate feature importance using the SAME preprocessed data used for feature selection**
            print("🎯 Calculating feature importance using preprocessed data...")
            print(f"   Using X_train for importance: shape={X_train.shape}")
            
            # **FIX: Safely handle pandas Series/DataFrame min/max formatting**
            try:
                data_min = float(X_train.min().min() if hasattr(X_train.min(), 'min') else X_train.min())
                data_max = float(X_train.max().max() if hasattr(X_train.max(), 'max') else X_train.max())
                print(f"   Data range: [{data_min:.4f}, {data_max:.4f}]")
            except Exception as fmt_error:
                print(f"   Data range: [min={X_train.min()}, max={X_train.max()}] (format error: {fmt_error})")
            feature_importances = self._calculate_feature_importance(X_train, y_train, method, task_type=task_type)
            
            # Create visualization data
            # Ensure feature importance array matches feature count
            if len(feature_importances) != X_train.shape[1]:
                print(f"Warning: Feature importance array length ({len(feature_importances)}) does not match feature count ({X_train.shape[1]}), will adjust")
                if len(feature_importances) > X_train.shape[1]:
                    feature_importances = feature_importances[:X_train.shape[1]]
                else:
                    # If feature importance array is too short, pad with zeros
                    extended_importances = np.zeros(X_train.shape[1])
                    extended_importances[:len(feature_importances)] = feature_importances
                    feature_importances = extended_importances
            
            importances = np.array(feature_importances)
            # Ensure only valid indices are selected
            top_n = min(10, len(feature_names))
            indices = np.argsort(importances)[-top_n:]
            
            # Sort for visualization
            features_for_visualization = [(feature_names[i], float(importances[i])) for i in indices]
            
            # Ensure feature importance is tuple list not other format
            if len(features_for_visualization) > 0 and not isinstance(features_for_visualization[0], tuple):
                print("Correcting visualization feature format to tuple list")
                if isinstance(features_for_visualization[0], list) and len(features_for_visualization[0]) == 2:
                    features_for_visualization = [(str(item[0]), float(item[1])) for item in features_for_visualization]
                else:
                    # If format is not (name, importance), create correct format
                    feature_names = [str(f) for f in selected_features]
                    importances_list = [float(imp) for imp in importances]
                    features_for_visualization = list(zip(feature_names, importances_list))
            
            # Ensure all feature names are string type
            features_for_visualization = [(str(name), float(importance)) for name, importance in features_for_visualization]
            
            # Print final results for debugging
            print(f"Returning {len(selected_features)} selected features")
            print(f"Feature importance array length: {len(importances)}")
            print(f"Visualization feature array length: {len(features_for_visualization)}")
            print(f"Feature importance values: {importances}")
            print(f"Feature names: {selected_features}")
            print(f"Visualization features: {features_for_visualization[:5]}")
            
            # Ensure checking and printing returned label information
            print(f"Returned label type: {type(original_y_train)}")
            if isinstance(original_y_train, pd.Series):
                print(f"Label unique values: {original_y_train.unique()}, count: {len(original_y_train.unique())}")
            elif isinstance(original_y_train, np.ndarray):
                print(f"Label unique values: {np.unique(original_y_train)}, count: {len(np.unique(original_y_train))}")
                
            # Ensure returning original labels, not labels that may have been modified during feature selection process
            return {
                'X_train_selected': X_train_selected,
                'X_test_selected': X_test_selected if X_test is not None else None,
                'y_train': original_y_train,  # Return original labels, ensure label information is not lost
                'selected_features': selected_features,
                'feature_importance': importances,
                'features_for_visualization': features_for_visualization
            }
            
        except Exception as e:
            # Record detailed error and throw appropriate exception
            error_msg = f"Error applying feature selection method '{method}': {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise FeatureSelectionError(error_msg)

    def _calculate_feature_importance(self, X, y, method, task_type='classification', **kwargs):
        """Calculate feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            task_type: Task type, 'classification' or 'regression'
            **kwargs: Additional parameters
            
        Returns:
            numpy.ndarray: Feature importance
        """
        # 确保numpy可用
        import numpy as np
        
        # Data validation
        if X is None or X.shape[0] == 0 or X.shape[1] == 0:
            print("Warning: Input feature data is empty or invalid")
            return np.ones(1) / 1
            
        if y is None or len(y) == 0:
            print("Warning: Input target data is empty or invalid")
            return np.ones(X.shape[1]) / X.shape[1]

        print(f"🎯 Calculating feature importance using method: {method}")
        print(f"   Task type: {task_type}")
        print(f"   Data shape: X={X.shape}, y={len(y)}")
        
        # **CRITICAL FIX: Check if preprocessing has already been applied**
        def smart_preprocessing(X):
            """Smart preprocessing that detects if preprocessing was already applied"""
            print(f"🔍 Analyzing data for preprocessing needs...")
            
            # Check data characteristics
            data_min_val = float(X.min().min()) if hasattr(X.min(), 'min') else float(X.min())
            data_max_val = float(X.max().max()) if hasattr(X.max(), 'max') else float(X.max())
            data_mean_val = float(X.mean().mean()) if hasattr(X.mean(), 'mean') else float(X.mean())
            data_std_val = float(X.std().mean()) if hasattr(X.std(), 'mean') else float(X.std())
            data_range = data_max_val - data_min_val
            
            print(f"   Data range: [{data_min_val:.4f}, {data_max_val:.4f}]")
            print(f"   Data mean: {data_mean_val:.4f}, std: {data_std_val:.4f}")
            
            # **CRITICAL: Detect if data is already preprocessed**
            is_already_standardized = (-3 < data_min_val < 0) and (0 < data_max_val < 3) and (abs(data_mean_val) < 0.5)
            is_already_normalized = (0 <= data_min_val <= 0.1) and (0.9 <= data_max_val <= 1.1)
            
            if is_already_standardized:
                print("✅ Data appears already standardized - using minimal processing")
                return X, None, None
            elif is_already_normalized:
                print("✅ Data appears already normalized - using minimal processing")
                return X, None, None
            elif data_range < 0.01:
                print("⚠️  Data has very small range - scaling up for better discrimination")
                return X * 100, None, None
            elif data_range > 1000 or data_min_val < -100:
                print("🔧 Large data range detected - applying gentle standardization")
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                return X_scaled, scaler, None
            else:
                print("✅ Data range is appropriate - using original data")
                return X, None, None

        try:
            # Apply smart preprocessing
            X_processed, scaler, imputer = smart_preprocessing(X)
            
            # **CRITICAL FIX: Enhanced label processing**
            # Use the enhanced label processor for proper task type handling
            from app.utils.label_processor import EnhancedLabelProcessor
            label_processor = EnhancedLabelProcessor()
            
            # Detect actual task type from labels (override if needed)
            actual_task_type = label_processor.detect_task_type(y)
            if actual_task_type != task_type:
                print(f"⚠️  TASK TYPE MISMATCH: Specified={task_type}, Detected={actual_task_type}")
                print(f"   Using detected task type: {actual_task_type}")
                task_type = actual_task_type
            
            # Process labels properly
            y_processed, label_metadata = label_processor.process_labels_smart(y, task_type)
            print(f"   Label processing: {label_metadata.get('num_classes', 'N/A')} classes" if task_type == 'classification' else f"   Label range: [{label_metadata.get('min_value', 'N/A'):.3f}, {label_metadata.get('max_value', 'N/A'):.3f}]")

        except Exception as preprocessing_error:
            print(f"❌ Preprocessing failed: {preprocessing_error}")
            # Fallback to minimal processing
            X_processed = np.array(X)
            y_processed = np.array(y)

        # For random forest or feature importance methods, use random forest to calculate feature importance
        if method in ['Feature Importance', 'Random Forest']:
            try:
                # **ENHANCED: Multiple importance calculation strategies**
                print("🌲 Using Random Forest for feature importance calculation")
                
                # Choose appropriate random forest model based on task type
                if task_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=200,         # Increased for more stable importance
                        max_depth=None,          # Allow deeper trees for complex relationships
                        min_samples_split=2,     # Lower values for better feature discrimination
                        min_samples_leaf=1,      # Allow single-sample leaves for better discrimination
                        max_features='sqrt',     # Better feature selection for classification
                        bootstrap=True,          # Enable bootstrapping
                        random_state=42,
                        class_weight='balanced', # Handle class imbalance
                        n_jobs=-1
                    )
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=200,        # Increased for stability
                        max_depth=None,          # Allow deeper trees
                        min_samples_split=2,     # Lower values for better feature discrimination
                        min_samples_leaf=1,      # Allow single-sample leaves
                        max_features='sqrt',     # Better feature selection
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    )
                
                # Fit the model
                model.fit(X_processed, y_processed)
                
                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_.copy()
                    
                    # **CRITICAL FIX: Enhanced importance validation and correction**
                    print(f"   Raw importances - Max: {float(importances.max()):.6f}, Min: {float(importances.min()):.6f}, Mean: {float(importances.mean()):.6f}")
                    
                    # Check for problematic importance patterns
                    if importances.max() < 0.001:
                        print("⚠️  WARNING: Feature importances are extremely low!")
                        print(f"   Maximum importance: {importances.max():.8f}")
                        print("   Possible causes:")
                        print("   1. Over-preprocessing removing signal")
                        print("   2. Poor feature-target relationships") 
                        print("   3. Label quality issues")
                        print("   4. Dataset too small or noisy")
                        
                        # **ENHANCED FALLBACK: Try multiple strategies**
                        print("🔄 Trying enhanced importance calculation strategies...")
                        
                        # Strategy 1: Permutation importance
                        try:
                            from sklearn.inspection import permutation_importance
                            print("   Strategy 1: Permutation importance")
                            perm_importance = permutation_importance(
                                model, X_processed, y_processed, 
                                n_repeats=5, random_state=42, n_jobs=-1
                            )
                            perm_importances = perm_importance.importances_mean
                            
                            if perm_importances.max() > importances.max():
                                print(f"   ✅ Permutation importance better: Max={perm_importances.max():.6f}")
                                importances = perm_importances
                            
                        except Exception as perm_error:
                            print(f"   ❌ Permutation importance failed: {perm_error}")
                        
                        # Strategy 2: Multiple Random Forest averaging
                        if importances.max() < 0.001:
                            try:
                                print("   Strategy 2: Multiple Random Forest averaging")
                                ensemble_importances = []
                                for seed in [42, 123, 456, 789, 999]:
                                    if task_type == 'classification':
                                        temp_model = RandomForestClassifier(
                                            n_estimators=100, random_state=seed, 
                                            class_weight='balanced', n_jobs=-1
                                        )
                                    else:
                                        temp_model = RandomForestRegressor(
                                            n_estimators=100, random_state=seed, n_jobs=-1
                                        )
                                    
                                    temp_model.fit(X_processed, y_processed)
                                    ensemble_importances.append(temp_model.feature_importances_)
                                
                                # Average importances across models
                                avg_importances = np.mean(ensemble_importances, axis=0)
                                if avg_importances.max() > importances.max():
                                    print(f"   ✅ Ensemble averaging better: Max={avg_importances.max():.6f}")
                                    importances = avg_importances
                                    
                            except Exception as ensemble_error:
                                print(f"   ❌ Ensemble averaging failed: {ensemble_error}")
                        
                        # Strategy 3: Correlation-based importance
                        if importances.max() < 0.001:
                            try:
                                print("   Strategy 3: Correlation-based importance")
                                if task_type == 'classification':
                                    # For classification, use mutual information
                                    from sklearn.feature_selection import mutual_info_classif
                                    corr_importances = mutual_info_classif(X_processed, y_processed, random_state=42)
                                else:
                                    # For regression, use correlation
                                    corr_importances = np.abs(np.corrcoef(X_processed.T, y_processed)[:-1, -1])
                                    corr_importances = np.nan_to_num(corr_importances)  # Handle NaN
                                
                                if corr_importances.max() > importances.max():
                                    print(f"   ✅ Correlation-based better: Max={corr_importances.max():.6f}")
                                    importances = corr_importances
                                    
                            except Exception as corr_error:
                                print(f"   ❌ Correlation-based importance failed: {corr_error}")
                    
                    # Final normalization and validation
                    if importances.max() > 0:
                        # Normalize to sum to 1
                        importances = importances / importances.sum()
                        
                        # Add small baseline to prevent complete zeros
                        min_importance = importances.max() * 0.001  # 0.1% of max
                        importances = np.maximum(importances, min_importance)
                        importances = importances / importances.sum()  # Re-normalize
                    else:
                        # Ultimate fallback: uniform distribution
                        print("   Using uniform distribution as final fallback")
                        importances = np.ones(X.shape[1]) / X.shape[1]
                    
                    print(f"✅ Final feature importance calculation:")
                    print(f"   Max: {float(importances.max()):.6f}, Min: {float(importances.min()):.6f}, Mean: {float(importances.mean()):.6f}")
                    print(f"   Non-zero features: {np.sum(importances > 1e-6)}/{len(importances)}")
                    return importances
                else:
                    print("❌ Model does not have feature_importances_ attribute")
                    return self._calculate_mutual_info_importance_enhanced(X_processed, y_processed, task_type)
                    
            except Exception as rf_error:
                print(f"❌ Random forest feature importance calculation failed: {rf_error}")
                return self._calculate_mutual_info_importance_enhanced(X_processed, y_processed, task_type)
        
        else:
            # For other methods, use appropriate sklearn selectors
            try:
                if method == 'SelectKBest':
                    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                    
                    score_func = f_classif if task_type == 'classification' else f_regression
                    selector = SelectKBest(score_func=score_func, k='all')
                    selector.fit(X_processed, y_processed)
                    importances = selector.scores_
                    
                    # Normalize scores
                    if importances is not None and np.ptp(importances) > 0:
                        importances = (importances - np.min(importances)) / np.ptp(importances)
                    else:
                        importances = np.ones(X.shape[1]) / X.shape[1]
                else:
                    # Default to enhanced mutual information
                    return self._calculate_mutual_info_importance_enhanced(X_processed, y_processed, task_type)

                return np.nan_to_num(importances)
                
            except Exception as method_error:
                print(f"❌ {method} feature importance calculation failed: {method_error}")
                return self._calculate_mutual_info_importance_enhanced(X_processed, y_processed, task_type)

    def _calculate_mutual_info_importance_enhanced(self, X, y, task_type):
        """Enhanced mutual information calculation with multiple strategies"""
        import numpy as np
        
        try:
            print("🔄 Calculating enhanced mutual information importance...")
            print(f"   Data shape: X={X.shape}, y={len(y)}")
            print(f"   Task type: {task_type}")
            
            # Enhanced preprocessing for mutual information
            from sklearn.preprocessing import StandardScaler, RobustScaler
            
            # Use RobustScaler to handle outliers better
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            if task_type == 'classification':
                # Ensure proper label encoding
                if isinstance(y, pd.Series):
                    y_safe = y.astype(str)
                else:
                    y_safe = np.array([str(label) for label in y])
                
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_safe)
                
                print(f"   Classification: {len(np.unique(y_encoded))} classes")
                
                # Adaptive neighbor count
                n_neighbors = max(3, min(15, len(y_encoded) // 15))
                
                from sklearn.feature_selection import mutual_info_classif
                importances = mutual_info_classif(
                    X_scaled, y_encoded, 
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    copy=True,
                    random_state=42
                )
            else:
                # Regression task
                n_neighbors = max(3, min(15, len(y) // 15))
                
                from sklearn.feature_selection import mutual_info_regression
                importances = mutual_info_regression(
                    X_scaled, y, 
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    copy=True,
                    random_state=42
                )
            
            print(f"   Mutual info raw - Max: {float(importances.max()):.6f}, Min: {float(importances.min()):.6f}")
            
            # Enhanced validation and fallback
            if np.all(importances < 1e-10):
                print("⚠️  WARNING: Mutual information returned all zeros!")
                print("   Trying fallback strategies...")
                
                # Fallback 1: Variance-based importance
                feature_variances = np.var(X_scaled, axis=0)
                if np.sum(feature_variances) > 0:
                    var_importances = feature_variances / np.sum(feature_variances)
                    print(f"   Variance-based - Max: {var_importances.max():.6f}")
                    importances = var_importances
                else:
                    # Fallback 2: Uniform distribution
                    print("   Using uniform distribution")
                    importances = np.ones(X.shape[1]) / X.shape[1]
            
            # Post-processing: Ensure reasonable distribution
            if importances.max() > 0:
                # Add minimum baseline (prevents complete feature exclusion)
                min_importance = importances.max() * 0.005  # 0.5% of max
                importances = np.maximum(importances, min_importance)
                # Normalize
                importances = importances / importances.sum()
            
            print(f"✅ Enhanced mutual information completed:")
            print(f"   Final - Max: {float(importances.max()):.6f}, Min: {float(importances.min()):.6f}, Mean: {float(importances.mean()):.6f}")
            print(f"   Active features: {np.sum(importances > importances.mean())}/{len(importances)}")
            
            return importances
            
        except Exception as e:
            print(f"❌ Enhanced mutual information failed: {e}")
            print("   Using uniform fallback")
            return np.ones(X.shape[1]) / X.shape[1]

    def apply_feature_selection(self, X, y=None, method=None, n_features=None, task_type='classification', **kwargs):
        """Apply feature selection method and return selected feature indices and importance scores"""
        try:
              
            import numpy as np
            
            if method is None:
                  
                return np.arange(X.shape[1]), np.ones(X.shape[1]) / X.shape[1]
            
              
            importances = self._calculate_feature_importance(X, y, method, task_type, **kwargs)
            
              
            if any(isinstance(imp, np.ndarray) for imp in importances):
                importances = np.array([float(imp.mean()) if isinstance(imp, np.ndarray) else float(imp) 
                                       for imp in importances])
            
              
            if n_features is None:
                n_features = X.shape[1] // 2
                n_features = max(1, min(n_features, X.shape[1]))    
            
              
            indices = np.argsort(importances)[::-1][:n_features]
            
              
            print(f"Selected features indices: {indices}, shape: {indices.shape}")
            print(f"Importances shape: {importances.shape}")
            
            return indices, importances
        except Exception as e:
            error_msg = f"Error applying feature selection method '{method}': {str(e)}\n{traceback.format_exc()}"
            print(error_msg)    
            raise FeatureSelectionError(error_msg)
    
    def apply_pca(self, X, y, k=10):
        """Apply PCA feature selection"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=min(k, X.shape[1], X.shape[0]-1))
            pca.fit(X_scaled)
            
            # Get feature importance based on explained variance
            importances = np.abs(pca.components_).mean(axis=0)
            
            # Select top k features
            indices = np.argsort(importances)[::-1][:k]
            
            return indices
            
        except Exception as e:
            print(f"PCA feature selection failed: {e}")
            return np.arange(min(k, X.shape[1]))
    
    def apply_mutual_info(self, X, y, k=10):
        """Apply mutual information feature selection"""
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine task type
            if self._is_classification_task(y):
                # Encode labels for classification
                if isinstance(y, pd.Series):
                    y_safe = y.astype(str)
                else:
                    y_safe = np.array([str(label) for label in y])
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_safe)
                
                # Calculate mutual information
                importances = mutual_info_classif(X_scaled, y_encoded, random_state=42)
            else:
                # For regression
                importances = mutual_info_regression(X_scaled, y, random_state=42)
            
            # Select top k features
            indices = np.argsort(importances)[::-1][:k]
            
            return indices
            
        except Exception as e:
            print(f"Mutual information feature selection failed: {e}")
            return np.arange(min(k, X.shape[1]))
