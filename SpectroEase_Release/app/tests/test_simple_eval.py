import numpy as np
import pandas as pd
from app.services.evaluation_service import EvaluationService
from sklearn.ensemble import RandomForestClassifier
from app.utils.label_processor import EnhancedLabelProcessor

def test_mixed_types():
    """测试混合类型labels的evaluation"""
    print("English text")
    
  
  
    X = np.random.rand(100, 5)
    y_raw = np.zeros(100, dtype=object)
    
  
    y_raw[:30] = np.random.randint(0, 3, 30)
    
  
    float_labels = [0.0, 1.0, 2.0]
    for i in range(30, 60):
        y_raw[i] = float_labels[np.random.randint(0, 3)]
    
  
    string_labels = ['类别0', '类别1', '类别2']
    for i in range(60, 80):
        y_raw[i] = string_labels[np.random.randint(0, 3)]
        
  
    bool_labels = [True, False]
    for i in range(80, 100):
        y_raw[i] = bool_labels[np.random.randint(0, 2)]
    
  
  
  
    train_indices = list(range(0, 60))
    test_indices = list(range(0, 100))  # 所有data用于测试
    
    X_train = X[train_indices]
    
    # **CRITICAL FIX: Do not force float conversion on classification labels**
    # Use enhanced label processor to detect task type first
    label_processor = EnhancedLabelProcessor()
    
    task_type = label_processor.detect_task_type(y_raw)
    print(f"🤖 Test task type detection: {task_type}")
    
    if task_type == 'classification':
        print("🔧 Classification test - preserving string labels")
        y_train = y_raw[train_indices]  # Keep original string labels
        y_test = y_raw[test_indices]
    else:
        print("🔧 Regression test - converting to float")
                # **CRITICAL FIX: Safe float conversion with string label detection**
        try:
            # Check for string labels before conversion
            sample_labels = [str(label) for label in y_raw[train_indices][:5]]
            has_string_labels = any(not label.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() 
                                   for label in sample_labels if label.strip())
            
            if has_string_labels:
                raise ValueError(f"String labels detected: {sample_labels[:3]}. Cannot convert to float for regression.")
            
            y_train = y_raw[train_indices].astype(float)  # Safe conversion to float for training
            y_test = y_raw[test_indices].astype(float)
        except Exception as e:
            print(f"❌ Float conversion failed: {e}")
            print("🔧 Keeping original labels for classification")
            y_train = y_raw[train_indices]
            y_test = y_raw[test_indices]
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    y_pred_mixed = np.array(y_pred, dtype=object)
    for i in range(len(y_pred_mixed)):
        if i % 4 == 0:
            y_pred_mixed[i] = int(y_pred_mixed[i])  # 整数
        elif i % 4 == 1:
            y_pred_mixed[i] = float(y_pred_mixed[i])  # 浮点数
        elif i % 4 == 2:
            y_pred_mixed[i] = string_labels[int(y_pred_mixed[i]) % 3]  # 字符串
        
    print("English text")
    print("English text")
    print("English text")
    print("English text")
    
    print("English text")
    for i in range(20):
        print(f"  [{i}] {y_test[i]} ({type(y_test[i]).__name__})")
    
    print("English text")
    for i in range(20):
        print(f"  [{i}] {y_pred_mixed[i]} ({type(y_pred_mixed[i]).__name__})")
    
    print("English text")
    y_test_types = {type(v).__name__: sum(1 for x in y_test if type(x).__name__ == type(v).__name__) 
                  for v in y_test}
    y_pred_types = {type(v).__name__: sum(1 for x in y_pred_mixed if type(x).__name__ == type(v).__name__) 
                   for v in y_pred_mixed}
    print("English text")
    print("English text")
    
    print("English text")
    evaluation_service = EvaluationService()
    
    try:
        results = evaluation_service.evaluate_model(clf, X_test, y_test, 'classification')
        print("English text")
        for key, value in results.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value}")
        
        results_mixed_pred = evaluation_service.evaluate_model(clf, X_test, y_test, 'classification')
        print("English text")
        for key, value in results_mixed_pred.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value}")
        
        print("English text")
    except Exception as e:
        print(f"\nevaluationfailed: {e}")
        print("English text")
        
if __name__ == "__main__":
    test_mixed_types() 