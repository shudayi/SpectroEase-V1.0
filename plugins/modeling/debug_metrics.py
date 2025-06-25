import numpy as np
import pandas as pd
import sys
import logging
import traceback
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import LabelEncoder
sys.path.append('../..')

  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_metrics')

from plugins.modeling.qualitative_analyzer import QualitativeAnalyzer

def diagnose_data(X, y):
    """Diagnose data features and labels"""
  
    
    # analysisfeatures
  
  
  
  
  
  
    
    # analysislabels
  
  
    
  
    unique_types = set(type(label) for label in y)
  
    
  
    try:
        unique_values = np.unique(y)
  
  
    except:
        unique_values = []  # set default value
        
    # check label type distribution
    has_float = any(isinstance(label, float) for label in y)
    has_int = any(isinstance(label, int) for label in y)
    has_string = any(isinstance(label, str) for label in y)
    
  
    
  
    if len(unique_types) > 1:
        type_counts = {}
        for label in y:
            t = type(label).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
  
    
    return {
        "X_shape": X.shape,
        "y_len": len(y),
        "unique_label_types": list(str(t) for t in unique_types),
        "unique_label_count": len(unique_values) if 'unique_values' in locals() else 'unknown'
    }

def test_with_synthetic_data():
    """Test cross-validation using synthetic data"""
  
    
  
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    
  
    diagnose_data(X, y)
    
  
    try:
        analyzer = QualitativeAnalyzer(method='rf')
        analyzer.fit(X, y)
        result = analyzer.cross_validate(X, y)
  
    except Exception as e:
  
        logger.error(traceback.format_exc())

def test_with_mixed_types():
    """Test cross-validation using mixed type labels"""
  
    
  
    X, y_int = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    
  
    y_mixed = []
    for i, label in enumerate(y_int):
        if i % 3 == 0:
            y_mixed.append(str(label))  # string
        elif i % 3 == 1:
            y_mixed.append(float(label))  # float
        else:
            y_mixed.append(int(label))  # integer
    
  
    diagnose_data(X, y_mixed)
    
  
    try:
        analyzer = QualitativeAnalyzer(method='rf')
        analyzer.fit(X, y_mixed)
        result = analyzer.cross_validate(X, y_mixed)
  
    except Exception as e:
  
        logger.error(traceback.format_exc())

def test_direct_sklearn_metrics():
    """Directly test sklearn metric functions"""
  
    
  
    y_true = ['0', 0, 1.0, 1, '1']
    y_pred = [0.0, 0, 1, '1', 1.0]
    
  
  
    
  
    try:
  
        y_true_str = [str(x) for x in y_true]
        y_pred_str = [str(x) for x in y_pred]
        acc_str = accuracy_score(y_true_str, y_pred_str)
  
    except Exception as e:
        pass  # handle exception cases
    
    try:
  
        le = LabelEncoder()
        le.fit(y_true + y_pred)  # combine all unique values for fitting
        y_true_int = le.transform([str(x) for x in y_true])
        y_pred_int = le.transform([str(x) for x in y_pred])
        acc_int = accuracy_score(y_true_int, y_pred_int)
  
    except Exception as e:
        pass  # handle exception cases

if __name__ == "__main__":
  
    
  
    test_with_synthetic_data()
    test_with_mixed_types()
    test_direct_sklearn_metrics()
    
  