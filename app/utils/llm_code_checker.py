#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM生成的代码静态检查模块
用于检测常见错误模式，避免之前遇到的问题
"""

import ast
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CodeIssue:
    """Code issue"""
    severity: str  # 'error' or 'warning'
    message: str
    line_number: int
    pattern: str = ""


class LLMCodeChecker:
    """Static checker for LLM-generated code"""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
    
    def check(self, code: str, algorithm_type: str) -> List[CodeIssue]:
        """
        Check code for common error patterns
        
        Args:
            code: Python code to check
            algorithm_type: Algorithm type ('Preprocessing', 'Feature Selection', 'Modeling', 'Data Partitioning')
        
        Returns:
            List of issues found
        """
        self.issues = []
        
        if not code or not code.strip():
            self.issues.append(CodeIssue(
                severity='error',
                message='Code is empty',
                line_number=0
            ))
            return self.issues
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                severity='error',
                message=f'Syntax error: {str(e)}',
                line_number=e.lineno or 0
            ))
            return self.issues
        
        # Algorithm-specific checks
        if algorithm_type == 'Feature Selection':
            self._check_feature_selection(code, tree)
        elif algorithm_type == 'Modeling':
            self._check_modeling(code, tree)
        elif algorithm_type == 'Data Partitioning':
            self._check_data_partitioning(code, tree)
        elif algorithm_type == 'Preprocessing':
            self._check_preprocessing(code, tree)
        
        # Common pattern checks
        self._check_common_patterns(code)
        
        return self.issues
    
    def _check_feature_selection(self, code: str, tree: ast.AST):
        """Check feature selection algorithms for common errors"""
        lines = code.split('\n')
        
        # Check 1: Whether returning X.index instead of X.columns
        for i, line in enumerate(lines, 1):
            if 'select_features' in line and 'return' in line:
                # Check if X.index is used
                if re.search(r'X\.index\[', line) or re.search(r'return.*X\.index', line):
                    self.issues.append(CodeIssue(
                        severity='error',
                        message='Feature selection should return X.columns (feature names), not X.index (sample names)',
                        line_number=i,
                        pattern='X.index[...]'
                    ))
        
        # Check 2: Whether return value contains X.columns
        # Directly check return statements in select_features method
        select_features_code = ""
        in_select_features = False
        for i, line in enumerate(lines, 1):
            if 'def select_features' in line:
                in_select_features = True
            if in_select_features:
                select_features_code += line + '\n'
                if line.strip().startswith('def ') and 'select_features' not in line:
                    break
        
        # Check return value format
        if 'select_features' in code:
            if 'X.columns' not in select_features_code and 'X.columns' not in code:
                # Check if there's a return statement
                if 'return' in select_features_code:
                    self.issues.append(CodeIssue(
                        severity='warning',
                        message='Return value may not use X.columns. Please confirm it returns feature column names, not indices',
                        line_number=0,
                        pattern='return statement'
                    ))
    
    def _check_modeling(self, code: str, tree: ast.AST):
        """Check modeling algorithms for common errors"""
        lines = code.split('\n')
        
        # Check 1: Whether CV hyperparameter selection uses wrong indexing pattern
        for i, line in enumerate(lines, 1):
            # Check for cv_scores[1:] pattern
            if re.search(r'cv_scores\[1:\]', line) or re.search(r'scores\[1:\]', line):
                self.issues.append(CodeIssue(
                    severity='error',
                    message='Detected cv_scores[1:] slicing. Should use np.argmin(cv_scores) + 1',
                    line_number=i,
                    pattern='cv_scores[1:]'
                ))
            
            # Check for complex indexing patterns
            if re.search(r'sel\[.*argmin.*\[1:\]', line) or re.search(r'range.*\[.*argmin.*\[1:\]', line):
                self.issues.append(CodeIssue(
                    severity='error',
                    message='Detected complex CV hyperparameter selection indexing. Should simplify to np.argmin(cv_scores) + 1',
                    line_number=i,
                    pattern='complex indexing'
                ))
        
        # Check 2: Whether correct selection pattern is used
        has_correct_pattern = re.search(r'np\.argmin\(cv_scores\)\s*\+\s*1', code) or \
                             re.search(r'np\.argmin\(scores\)\s*\+\s*1', code)
        
        if not has_correct_pattern and 'cv_scores' in code:
            self.issues.append(CodeIssue(
                severity='warning',
                message='Standard CV hyperparameter selection pattern (np.argmin(cv_scores) + 1) not detected. Please confirm selection logic is correct',
                line_number=0,
                pattern='CV selection'
            ))
    
    def _check_data_partitioning(self, code: str, tree: ast.AST):
        """Check data partitioning algorithms for common errors"""
        lines = code.split('\n')
        
        # Check 1: Whether X and y are extracted correctly
        has_correct_extraction = False
        for i, line in enumerate(lines, 1):
            if 'data.iloc[:, :-1]' in line and 'data.iloc[:, -1]' in line:
                has_correct_extraction = True
                break
        
        if not has_correct_extraction and 'partition' in code:
            self.issues.append(CodeIssue(
                severity='warning',
                message='Standard data extraction pattern (X = data.iloc[:, :-1], y = data.iloc[:, -1]) not detected',
                line_number=0,
                pattern='data extraction'
            ))
    
    def _check_preprocessing(self, code: str, tree: ast.AST):
        """Check preprocessing algorithms for common errors"""
        lines = code.split('\n')
        
        # Check 1: MSC polyfit coefficient order
        # Only check if code contains both 'polyfit' and 'msc'
        if 'polyfit' in code.lower() and 'msc' in code.lower():
            # Determine which polyfit function is used in the entire code
            uses_polynomial_polyfit = 'polynomial.polynomial.polyfit' in code or \
                                     'from numpy.polynomial.polynomial import polyfit' in code
            uses_np_polyfit = re.search(r'\bnp\.polyfit\b', code) or \
                             re.search(r'\bnumpy\.polyfit\b', code) or \
                             ('import numpy as np' in code and 'np.polyfit' in code)
            
            # Find polyfit lines and check coefficient usage
            for i, line in enumerate(lines, 1):
                if 'polyfit' in line.lower():
                    # Look for coefficient usage in following lines (check up to 10 lines for context)
                    if i < len(lines):
                        next_lines = '\n'.join(lines[i:min(i+10, len(lines))])
                        
                        # For numpy.polynomial.polynomial.polyfit: returns [intercept, slope]
                        # Correct MSC: (y - p[0]) / p[1] where p[0]=intercept, p[1]=slope
                        # Wrong pattern: (y - p[1]) / p[0] - using p[1] as intercept
                        if uses_polynomial_polyfit:
                            # Only flag if we see the clearly wrong pattern: subtracting p[1] and dividing by p[0]
                            wrong_pattern = re.search(r'\([^)]*-\s*[a-zA-Z_][a-zA-Z0-9_]*\[1\][^)]*\)\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*\[0\]', next_lines)
                            if wrong_pattern:
                                self.issues.append(CodeIssue(
                                    severity='error',
                                    message='MSC polyfit coefficient order may be wrong: numpy.polynomial.polynomial.polyfit returns [intercept, slope], should use (y - p[0]) / p[1] where p[0]=intercept, p[1]=slope',
                                    line_number=i,
                                    pattern='polyfit coefficient order'
                                ))
                        
                        # For np.polyfit: returns [slope, intercept] (high to low order)
                        # Correct MSC: (y - coeffs[1]) / coeffs[0] where coeffs[0]=slope, coeffs[1]=intercept
                        # Wrong pattern: (y - coeffs[0]) / coeffs[1] - using coeffs[0] as intercept
                        elif uses_np_polyfit:
                            # Only flag if we see the clearly wrong pattern: subtracting coeffs[0] and dividing by coeffs[1]
                            wrong_pattern = re.search(r'\([^)]*-\s*[a-zA-Z_][a-zA-Z0-9_]*\[0\][^)]*\)\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*\[1\]', next_lines)
                            if wrong_pattern:
                                self.issues.append(CodeIssue(
                                    severity='error',
                                    message='MSC polyfit coefficient order may be wrong: np.polyfit returns [slope, intercept], should use (y - coeffs[1]) / coeffs[0] where coeffs[0]=slope, coeffs[1]=intercept',
                                    line_number=i,
                                    pattern='polyfit coefficient order'
                                ))
                        
                        # If cannot determine which polyfit, downgrade to warning
                        elif not uses_polynomial_polyfit and not uses_np_polyfit:
                            # Only warn if we see suspicious patterns, don't error
                            suspicious = re.search(r'\([^)]*-\s*[a-zA-Z_][a-zA-Z0-9_]*\[1\][^)]*\)\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*\[0\]', next_lines)
                            if suspicious and 'polynomial' in line.lower():
                                self.issues.append(CodeIssue(
                                    severity='warning',
                                    message='MSC polyfit coefficient order: Please verify polyfit function type. If using numpy.polynomial.polynomial.polyfit, should use (y - p[0]) / p[1]',
                                    line_number=i,
                                    pattern='polyfit coefficient order'
                                ))
        
        # Check 2: Savitzky-Golay deriv parameter
        for i, line in enumerate(lines, 1):
            if 'savgol_filter' in line:
                # Check deriv parameter
                if 'deriv=0' in line and 'savgol' in code.lower():
                    self.issues.append(CodeIssue(
                        severity='warning',
                        message='Savitzky-Golay uses deriv=0 (smoothing only). If original code uses first derivative, should change to deriv=1',
                        line_number=i,
                        pattern='deriv=0'
                    ))
    
    def _check_common_patterns(self, code: str):
        """Check common error patterns"""
        lines = code.split('\n')
        
        # Check 1: Whether markdown code block markers are present
        if '```python' in code or '```' in code:
            self.issues.append(CodeIssue(
                severity='warning',
                message='Code contains markdown markers, should be removed',
                line_number=0,
                pattern='markdown code fence'
            ))
        
        # Check 2: Whether unnecessary explanatory text is present
        explanation_patterns = [
            r'^Here is',
            r"^Here's",
            r'^This is',
            r'^The following',
            r'^以下是',
            r'^这是'
        ]
        for i, line in enumerate(lines[:5], 1):  # Only check first 5 lines
            for pattern in explanation_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    self.issues.append(CodeIssue(
                        severity='warning',
                        message='Code starts with explanatory text, should be removed',
                        line_number=i,
                        pattern='explanatory text'
                    ))
                    break
    
    def get_report(self) -> str:
        """Generate check report"""
        if not self.issues:
            return "✅ Code check passed, no issues found"
        
        report = []
        errors = [i for i in self.issues if i.severity == 'error']
        warnings = [i for i in self.issues if i.severity == 'warning']
        
        if errors:
            report.append(f"❌ Found {len(errors)} error(s):")
            for i, issue in enumerate(errors, 1):
                report.append(f"  {i}. [{issue.pattern}] {issue.message}")
                if issue.line_number > 0:
                    report.append(f"     Line: {issue.line_number}")
        
        if warnings:
            report.append(f"\n⚠️  Found {len(warnings)} warning(s):")
            for i, issue in enumerate(warnings, 1):
                report.append(f"  {i}. [{issue.pattern}] {issue.message}")
                if issue.line_number > 0:
                    report.append(f"     Line: {issue.line_number}")
        
        return '\n'.join(report)
    
    def has_errors(self) -> bool:
        """Whether there are any errors"""
        return any(i.severity == 'error' for i in self.issues)


if __name__ == '__main__':
    # 测试代码
    test_code = """
import pandas as pd
import numpy as np

class TestSelector(FeatureSelectionAlgorithm):
    def select_features(self, X, y, params):
        # 错误的返回方式
        return X.index[:5]  # 应该返回X.columns
    """
    
    checker = LLMCodeChecker()
    issues = checker.check(test_code, 'Feature Selection')
    print(checker.get_report())

