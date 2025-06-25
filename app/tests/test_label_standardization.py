import sys
import os
import numpy as np
import logging

  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.data_compatibility import normalize_label, encode_classification_labels, prepare_labels_safely, standardize_classification_labels

  
logging.basicConfig(level=logging.INFO)

def test_label_normalization():
    """测试labels标准化功能"""
    print("English text")
    test_values = [
        1,           # 整数
        1.0,         # 浮点数
        '1',         # 字符串整数
        '1.0',       # 字符串浮点数
        ' 1 ',       # 带空格字符串
        2,           # 另一个整数
        None,        # None值
        'A',         # 字母
        'B',         # 字母
        ' C ',       # 带空格字母
        1.23,        # 小数
        '1.23',      # 字符串小数
        3.0          # 浮点整数
    ]
    
    print("English text")
    for val in test_values:
        normalized = normalize_label(val)
        print(f'{val} ({type(val).__name__}) -> {normalized} ({type(normalized).__name__})')

def test_classification_encoding():
    """测试classificationlabels编码过程"""
    print("English text")
  
    labels = np.array([1, 1.0, '1', '1.0', ' 1 ', 2, 3.0, 'A', ' B', None])
    
  
    print("English text")
    encoded, forward_map, inverse_map = encode_classification_labels(labels, True)
    
    print("English text")
    print("English text")
    print("English text")
    print("English text")
    
  
    for i, label in enumerate(labels):
        norm_label = normalize_label(label)
        encoded_val = encoded[i]
        print("English text")

def test_mixed_type_handling():
    """测试混合data类型的processing"""
    print("English text")
    
  
    ints = np.array([1, 2, 3, 4, 5])
  
    floats = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
  
    strings = np.array(['A', 'B', 'C', 'D', 'E'])
  
    mixed_nums = np.array([1, 2.0, 3, 4.5, 5])
  
    mixed_all = np.array([1, 2.0, 'A', 4.5, 'B'])
    
    datasets = {
        "全整数": ints,
        "全浮点数": floats,
        "全字符串": strings,
        "混合数值": mixed_nums,
        "混合全部": mixed_all
    }
    
    for name, data in datasets.items():
        print("English text")
        processed, task_type = prepare_labels_safely(data)
        print("English text")
        print("English text")
        print("English text")
        print("English text")

def test_standardize_for_metrics():
    """测试为度量计算标准化classificationlabels"""
    print("English text")
    
  
    y_true = np.array([1, 2, '3', 'A', 'B', 1.0, ' C '])
    y_pred = np.array(['1', '2', 3, 'a', 'B', 1, 'C'])
    
    print("English text")
    print("English text")
    
  
    y_true_idx, y_pred_idx, label_map = standardize_classification_labels(y_true, y_pred)
    
    print(f"\nlabelsmapping: {label_map}")
    print("English text")
    print("English text")
    
  
    print("English text")
    for i in range(len(y_true)):
        true_orig = y_true[i]
        pred_orig = y_pred[i]
        true_norm = normalize_label(true_orig)
        pred_norm = normalize_label(pred_orig)
        true_idx = y_true_idx[i]
        pred_idx = y_pred_idx[i]
        
        print("English text")
    
  
    consistency = (y_true_idx == y_pred_idx)
    print("English text")
    print("English text")

if __name__ == "__main__":
    print("English text")
    test_label_normalization()
    test_classification_encoding()
    test_mixed_type_handling()
    test_standardize_for_metrics()
    print("English text")