# app/algorithms/wavelength_selection.py
"""
Wavelength selection algorithms - Core feature selection methods for spectral analysis
CARS: Competitive Adaptive Reweighted Sampling
SPA: Successive Projections Algorithm
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score


class CARS:
    """
    Competitive Adaptive Reweighted Sampling (CARS)
    
    CARS is one of the most popular wavelength selection algorithms in spectral analysis. 
    It uses Monte Carlo sampling and adaptive weighting of PLS regression coefficients 
    to gradually eliminate unimportant wavelength variables.
    
    Advantages:
    - Automatically selects optimal number of wavelengths
    - Reduces model complexity
    - Improves prediction accuracy
    - Suitable for collinear data
    
    References:
    Li, H., Liang, Y., Xu, Q., & Cao, D. (2009). 
    Key wavelengths screening using competitive adaptive reweighted sampling 
    method for multivariate calibration. 
    Analytica Chimica Acta, 648(1), 77-84.
    """
    
    def __init__(self, 
                 n_iterations: int = 50,
                 n_folds: int = 5,
                 pls_components: int = 5,
                 sampling_ratio: float = 0.9,
                 random_state: Optional[int] = 42):
        """
        Initialize CARS algorithm
        
        Parameters:
        -----------
        n_iterations : int
            Number of Monte Carlo sampling iterations, default 50
        n_folds : int
            Number of cross-validation folds, default 5
        pls_components : int
            Number of PLS components, default 5
        sampling_ratio : float
            Sample ratio for each sampling, default 0.9
        random_state : int
            Random seed
        """
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.pls_components = pls_components
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        
        self.selected_wavelengths_ = None
        self.selected_indices_ = None
        self.rmsecv_history_ = []
        self.wavelength_count_history_ = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CARS':
        """
        Execute CARS wavelength selection
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Spectral data matrix
        y : ndarray, shape (n_samples,)
            Target variable (quantitative analysis)
            
        Returns:
        --------
        self : CARS object
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        np.random.seed(self.random_state)
        
        n_samples, n_wavelengths = X.shape
        n_samples_per_iteration = int(n_samples * self.sampling_ratio)
        
        # Initialize all wavelengths as candidates
        remaining_indices = np.arange(n_wavelengths)
        
        best_rmsecv = np.inf
        best_indices = remaining_indices.copy()
        
        print(f"🔍 CARS started: {n_wavelengths} wavelengths")
        
        for iteration in range(self.n_iterations):
            # Monte Carlo sampling
            sample_indices = np.random.choice(n_samples, n_samples_per_iteration, replace=False)
            X_sampled = X[sample_indices][:, remaining_indices]
            y_sampled = y[sample_indices]
            
            # PLS modeling
            n_components = min(self.pls_components, X_sampled.shape[1], X_sampled.shape[0] - 1)
            if n_components < 1:
                break
                
            pls = PLSRegression(n_components=n_components)
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    pls, X_sampled, y_sampled, 
                    cv=min(self.n_folds, len(y_sampled)),
                    scoring='neg_mean_squared_error'
                )
                rmsecv = np.sqrt(-cv_scores.mean())
                
                self.rmsecv_history_.append(rmsecv)
                self.wavelength_count_history_.append(len(remaining_indices))
                
                # Record best result
                if rmsecv < best_rmsecv:
                    best_rmsecv = rmsecv
                    best_indices = remaining_indices.copy()
                
                # Calculate regression coefficient weights
                pls.fit(X_sampled, y_sampled)
                coef = np.abs(pls.coef_.ravel())
                
                # Adaptive Reweighted Sampling (ARS)
                # Exponential decay function controls elimination rate
                ratio = 0.5 * (1 + np.cos(iteration * np.pi / self.n_iterations))
                n_to_keep = max(int(len(remaining_indices) * ratio), self.pls_components)
                
                if n_to_keep >= len(remaining_indices):
                    break
                
                # Select wavelengths to keep based on weights
                weights = np.exp(-coef / coef.max())  # Higher weight means more likely to be eliminated
                
                # Competitive selection: wavelengths with lower weights are kept
                selected_local = np.argsort(weights)[:n_to_keep]
                remaining_indices = remaining_indices[selected_local]
                
                if iteration % 10 == 0:
                    print(f"  Iteration {iteration+1}/{self.n_iterations}: "
                          f"{len(remaining_indices)} wavelengths, RMSECV={rmsecv:.4f}")
                    
            except Exception as e:
                print(f"  ⚠ Iteration {iteration} error: {e}")
                break
        
        # Use wavelengths corresponding to best RMSECV
        self.selected_indices_ = best_indices
        self.selected_wavelengths_ = best_indices
        
        print(f"✅ CARS completed: Selected {len(self.selected_indices_)} wavelengths")
        print(f"   Best RMSECV: {best_rmsecv:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data, keeping only selected wavelengths
        
        Parameters:
        -----------
        X : ndarray
            Original spectral data
            
        Returns:
        --------
        X_selected : ndarray
            Spectral data with selected wavelengths
        """
        if self.selected_indices_ is None:
            raise ValueError("Must call fit method first")
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        else:
            return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Execute fit and transform"""
        return self.fit(X, y).transform(X)
    
    def get_selected_wavelengths(self) -> np.ndarray:
        """Return selected wavelength indices"""
        return self.selected_indices_


class SPA:
    """
    Successive Projections Algorithm (SPA)
    
    SPA is a forward variable selection algorithm that gradually selects wavelengths 
    with the most information through projection analysis. Faster than CARS, suitable for large-scale datasets.
    
    Advantages:
    - Fast computation
    - No iterative optimization required
    - Reduces collinearity
    - Selected wavelengths are representative
    
    References:
    Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., et al. (2001).
    The successive projections algorithm for variable selection in 
    spectroscopic multicomponent analysis.
    Chemometrics and Intelligent Laboratory Systems, 57(2), 65-73.
    """
    
    def __init__(self, 
                 n_wavelengths: int = 20,
                 min_wavelengths: int = 5,
                 max_wavelengths: int = 50):
        """
        Initialize SPA algorithm
        
        Parameters:
        -----------
        n_wavelengths : int
            Target number of wavelengths, default 20
        min_wavelengths : int
            Minimum number of wavelengths, default 5
        max_wavelengths : int
            Maximum number of wavelengths, default 50
        """
        self.n_wavelengths = n_wavelengths
        self.min_wavelengths = min_wavelengths
        self.max_wavelengths = max_wavelengths
        
        self.selected_indices_ = None
        self.projection_norms_ = []
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SPA':
        """
        Execute SPA wavelength selection
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_wavelengths)
            Spectral data matrix
        y : ndarray, optional
            Target variable (SPA is unsupervised, y only for compatibility)
            
        Returns:
        --------
        self : SPA object
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples, n_wavelengths = X.shape
        
        # Adjust target number of wavelengths
        n_target = min(self.n_wavelengths, self.max_wavelengths, n_wavelengths)
        n_target = max(n_target, self.min_wavelengths)
        
        print(f"🔍 SPA started: Selecting {n_target} from {n_wavelengths} wavelengths")
        
        selected = []
        remaining = list(range(n_wavelengths))
        
        # Select first wavelength (maximum norm)
        norms = np.linalg.norm(X, axis=0)
        first_idx = np.argmax(norms)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        self.projection_norms_.append(norms[first_idx])
        
        # Gradual projection selection
        for iteration in range(1, n_target):
            if not remaining:
                break
                
            # Projection matrix for currently selected wavelengths
            X_selected = X[:, selected]
            
            # Project remaining wavelengths
            max_norm = 0
            best_idx = None
            
            for idx in remaining:
                x_current = X[:, idx].reshape(-1, 1)
                
                # Project onto orthogonal complement space of currently selected wavelengths
                # P = I - X_selected @ (X_selected.T @ X_selected)^(-1) @ X_selected.T
                try:
                    # Use QR decomposition to speed up computation
                    Q, _ = np.linalg.qr(X_selected)
                    projection = x_current - Q @ (Q.T @ x_current)
                    norm = np.linalg.norm(projection)
                    
                    if norm > max_norm:
                        max_norm = norm
                        best_idx = idx
                except np.linalg.LinAlgError:
                    # If matrix is singular, use simplified method
                    norm = np.linalg.norm(x_current)
                    if norm > max_norm:
                        max_norm = norm
                        best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                self.projection_norms_.append(max_norm)
            else:
                break
            
            if (iteration + 1) % 5 == 0:
                print(f"  Selected {iteration + 1} wavelengths, projection norm={max_norm:.4f}")
        
        self.selected_indices_ = np.array(sorted(selected))
        
        print(f"✅ SPA completed: Selected {len(self.selected_indices_)} wavelengths")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data, keeping only selected wavelengths
        
        Parameters:
        -----------
        X : ndarray
            Original spectral data
            
        Returns:
        --------
        X_selected : ndarray
            Spectral data with selected wavelengths
        """
        if self.selected_indices_ is None:
            raise ValueError("Must call fit method first")
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        else:
            return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute fit and transform"""
        return self.fit(X, y).transform(X)
    
    def get_selected_wavelengths(self) -> np.ndarray:
        """Return selected wavelength indices"""
        return self.selected_indices_


# Compatibility wrapper functions
def cars_wavelength_selection(X: np.ndarray, y: np.ndarray, 
                               n_iterations: int = 50,
                               pls_components: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    CARS wavelength selection convenience function
    
    Returns:
    --------
    selected_indices : ndarray
        Selected wavelength indices
    X_selected : ndarray
        Data with selected wavelengths
    """
    cars = CARS(n_iterations=n_iterations, pls_components=pls_components)
    X_selected = cars.fit_transform(X, y)
    return cars.selected_indices_, X_selected


def spa_wavelength_selection(X: np.ndarray, n_wavelengths: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    SPA wavelength selection convenience function
    
    Returns:
    --------
    selected_indices : ndarray
        Selected wavelength indices
    X_selected : ndarray
        Data with selected wavelengths
    """
    spa = SPA(n_wavelengths=n_wavelengths)
    X_selected = spa.fit_transform(X)
    return spa.selected_indices_, X_selected

