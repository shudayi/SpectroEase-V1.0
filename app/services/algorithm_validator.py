"""
Algorithm Validator Service
Custom algorithm code validation service - Ensures LLM-generated code is safe and reliable
"""

import ast
import re
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self):
        if self.is_valid:
            msg = "✅ Validation passed"
            if self.warnings:
                msg += f"\n⚠️  Warnings ({len(self.warnings)}):\n" + "\n".join(f"  • {w}" for w in self.warnings)
            return msg
        else:
            msg = f"❌ Validation failed ({len(self.errors)} errors)"
            msg += "\n" + "\n".join(f"  • {e}" for e in self.errors)
            if self.warnings:
                msg += f"\n⚠️  Warnings ({len(self.warnings)}):\n" + "\n".join(f"  • {w}" for w in self.warnings)
            return msg


class AlgorithmValidator:
    """
    Custom algorithm code validator
    
    Validation checks:
    1. Python syntax correctness
    2. Interface method completeness
    3. Security check (dangerous imports/calls)
    4. Method signature matching
    """
    
    # Dangerous imports and calls
    DANGEROUS_PATTERNS = [
        'os.system',
        'subprocess.',
        'eval(',
        'exec(',
        '__import__',
        'open(',  # File operations need caution
        'compile(',
        'globals(',
        'locals(',
        'vars(',
        'dir(',
        'setattr(',
        'delattr(',
        'getattr(',  # Could be abused
    ]
    
    # Required methods for each type
    REQUIRED_METHODS = {
        'Preprocessing': ['get_name', 'get_params_info', 'apply'],
        'Feature Selection': ['get_name', 'get_params_info', 'select_features'],
        'Modeling': ['get_name', 'get_params_info', 'train', 'predict'],
        'Data Partitioning': ['get_name', 'get_params_info', 'partition'],
    }
    
    # Method signature specifications
    METHOD_SIGNATURES = {
        'Preprocessing': {
            'get_name': {'args': ['self'], 'returns': 'str'},
            'get_params_info': {'args': ['self'], 'returns': 'Dict'},
            'apply': {'args': ['self', 'data', 'params'], 'returns': 'pd.DataFrame'},
        },
        'Feature Selection': {
            'get_name': {'args': ['self'], 'returns': 'str'},
            'get_params_info': {'args': ['self'], 'returns': 'Dict'},
            'select_features': {'args': ['self', 'X', 'y', 'params'], 'returns': 'List[str]'},
        },
        'Modeling': {
            'get_name': {'args': ['self'], 'returns': 'str'},
            'get_params_info': {'args': ['self'], 'returns': 'Dict'},
            'train': {'args': ['self', 'X', 'y', 'params'], 'returns': 'Any'},
            'predict': {'args': ['self', 'model', 'X'], 'returns': 'np.ndarray'},
        },
        'Data Partitioning': {
            'get_name': {'args': ['self'], 'returns': 'str'},
            'get_params_info': {'args': ['self'], 'returns': 'Dict'},
            'partition': {'args': ['self', 'data', 'params'], 'returns': 'Tuple'},
        },
    }
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax
        
        Args:
            code: Code string
            
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error (line {e.lineno}): {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            return False, error_msg
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def validate_interface(self, code: str, algorithm_type: str) -> Tuple[bool, List[str]]:
        """
        Validate interface completeness
        V1.3.3: Added base class inheritance check
        
        Args:
            code: Code string
            algorithm_type: Algorithm type
            
        Returns:
            (is_valid, error_list)
        """
        errors = []
        
        if algorithm_type not in self.REQUIRED_METHODS:
            errors.append(f"Unknown algorithm type: {algorithm_type}")
            return False, errors
        
        # V1.3.3: Base class name mapping
        base_class_names = {
            'Preprocessing': 'PreprocessingAlgorithm',
            'Feature Selection': 'FeatureSelectionAlgorithm',
            'Modeling': 'ModelingAlgorithm',
            'Data Partitioning': 'DataPartitioningAlgorithm',
        }
        expected_base_class = base_class_names.get(algorithm_type)
        
        try:
            tree = ast.parse(code)
            
            # Find class definitions
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if not classes:
                errors.append("No class definition found in code")
                return False, errors
            
            # V1.3.3: Check if any class inherits from the correct base class
            has_valid_inheritance = False
            
            # Check each class (usually only one)
            for class_node in classes:
                # V1.3.3: Check base class inheritance
                base_names = []
                for base in class_node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                
                # Check if it inherits from the correct base class
                if expected_base_class and expected_base_class not in base_names:
                    errors.append(
                        f"❌ Class '{class_node.name}' MUST inherit from {expected_base_class}!\n"
                        f"   Correct format: class {class_node.name}({expected_base_class}):\n"
                        f"   Current definition: class {class_node.name}({', '.join(base_names) if base_names else ''}):"
                    )
                    continue  # Skip this class, check next one
                else:
                    has_valid_inheritance = True
                
                method_names = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]
                
                # Check required methods
                required = self.REQUIRED_METHODS[algorithm_type]
                missing = [m for m in required if m not in method_names]
                
                if missing:
                    errors.append(f"Missing required methods: {', '.join(missing)}")
                
                # Check method signatures (basic check)
                if algorithm_type in self.METHOD_SIGNATURES:
                    for method_node in class_node.body:
                        if isinstance(method_node, ast.FunctionDef) and method_node.name in self.METHOD_SIGNATURES[algorithm_type]:
                            expected_sig = self.METHOD_SIGNATURES[algorithm_type][method_node.name]
                            actual_args = [arg.arg for arg in method_node.args.args]
                            expected_args = expected_sig['args']
                            
                            if actual_args != expected_args:
                                errors.append(
                                    f"Method {method_node.name} signature mismatch: "
                                    f"Expected ({', '.join(expected_args)}), Actual ({', '.join(actual_args)})"
                                )
            
            # V1.3.3: If no valid inheritance found, add summary error
            if not has_valid_inheritance and expected_base_class:
                if not any("MUST inherit" in err for err in errors):
                    errors.append(
                        f"❌ No class found that inherits from {expected_base_class}!\n"
                        f"   Please ensure your class is defined as: class YourClassName({expected_base_class}):"
                    )
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Interface validation failed: {str(e)}")
            return False, errors
    
    def validate_security(self, code: str) -> Tuple[bool, List[str]]:
        """
        Security check - Detect dangerous code using AST parsing for accuracy
        
        Args:
            code: Code string
            
        Returns:
            (is_safe, dangerous_pattern_list)
        """
        dangerous_found = []
        
        try:
            # Parse code to AST for accurate detection
            tree = ast.parse(code)
            
            # Check for dangerous function calls using AST
            dangerous_calls = {
                'eval': 'eval(',
                'exec': 'exec(',
                'compile': 'compile(',
                '__import__': '__import__',
            }
            
            for node in ast.walk(tree):
                # Check for function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in dangerous_calls:
                            dangerous_found.append(f"Found dangerous code: {dangerous_calls[func_name]}")
                    elif isinstance(node.func, ast.Attribute):
                        # Check for os.system, subprocess calls, etc.
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            attr_name = node.func.attr
                            
                            # Check os.system, subprocess calls
                            if module_name == 'os' and attr_name == 'system':
                                dangerous_found.append("Found dangerous code: os.system")
                            elif module_name == 'subprocess':
                                dangerous_found.append(f"Found dangerous code: subprocess.{attr_name}")
            
            # Check for dangerous imports using AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess']:
                            # Check if dangerous functions are used (already checked above)
                            pass
                elif isinstance(node, ast.ImportFrom):
                    if node.module == 'os' and any(name.name == 'system' for name in node.names):
                        dangerous_found.append("Found dangerous import: from os import system")
                    elif node.module == 'subprocess':
                        dangerous_found.append(f"Found dangerous import: from subprocess import ...")
            
            # Check for file operations (open) - but allow if in safe context
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        # Check if it's in a safe context (e.g., with statement)
                        # For now, flag it but could be more lenient
                        dangerous_found.append("Found file operation: open( (use with caution)")
            
            # Check for network access using AST
            network_modules = ['requests', 'urllib', 'socket', 'http']
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name.startswith(mod) for mod in network_modules):
                            dangerous_found.append(f"Found network access: import {alias.name} (potentially unsafe)")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and any(node.module.startswith(mod) for mod in network_modules):
                        dangerous_found.append(f"Found network access: from {node.module} import ... (potentially unsafe)")
            
        except SyntaxError:
            # If code has syntax errors, fall back to simple string matching
            # but only check for critical patterns
            critical_patterns = ['eval(', 'exec(', 'os.system', '__import__']
            for pattern in critical_patterns:
                # Use regex to avoid matching in comments/strings
                if re.search(r'\b' + re.escape(pattern.replace('(', r'\(')) + r'\b', code):
                    # Check if it's in a comment or string
                    lines = code.split('\n')
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        # Skip comments
                        if stripped.startswith('#'):
                            continue
                        # Skip docstrings (basic check)
                        if '"""' in line or "'''" in line:
                            continue
                        # Check if pattern is in actual code
                        if pattern in line:
                            dangerous_found.append(f"Found dangerous code: {pattern} (line {i})")
                            break
        
        return len(dangerous_found) == 0, dangerous_found
    
    def validate_class_name(self, code: str) -> Tuple[bool, str]:
        """
        Validate class name (should be meaningful, not CustomAlgorithm)
        
        Args:
            code: Code string
            
        Returns:
            (is_valid, warning_message)
        """
        try:
            tree = ast.parse(code)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if not classes:
                return False, "No class definition found"
            
            class_name = classes[0].name
            
            # Check if it's a generic name
            generic_names = ['CustomAlgorithm', 'Algorithm', 'MyAlgorithm', 'Test', 'Custom']
            if class_name in generic_names:
                return True, f"Suggest using a more descriptive class name (current: {class_name})"
            
            return True, ""
            
        except Exception as e:
            return False, f"Class name check failed: {str(e)}"
    
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate import statements
        
        Args:
            code: Code string
            
        Returns:
            (is_valid, warning_list)
        """
        warnings = []
        
        try:
            tree = ast.parse(code)
            
            # Required imports
            required_imports = {
                'pandas': False,
                'numpy': False,
            }
            
            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in required_imports:
                            required_imports[alias.name] = True
                elif isinstance(node, ast.ImportFrom):
                    if node.module in required_imports:
                        required_imports[node.module] = True
            
            # Check for missing required imports
            missing = [name for name, imported in required_imports.items() if not imported]
            if missing:
                warnings.append(f"Possibly missing imports: {', '.join(missing)} (if code uses these libraries)")
            
            return True, warnings
            
        except Exception as e:
            warnings.append(f"Import check failed: {str(e)}")
            return True, warnings
    
    def validate_all(self, code: str, algorithm_type: str) -> ValidationResult:
        """
        Comprehensive validation
        
        Args:
            code: Code string
            algorithm_type: Algorithm type
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        
        # 1. Syntax check
        is_valid, error_msg = self.validate_syntax(code)
        if not is_valid:
            errors.append(f"[Syntax] {error_msg}")
            # Return directly on syntax error
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # 2. Interface check
        is_valid, interface_errors = self.validate_interface(code, algorithm_type)
        if not is_valid:
            errors.extend([f"[Interface] {e}" for e in interface_errors])
        
        # 3. Security check
        is_safe, security_issues = self.validate_security(code)
        if not is_safe:
            errors.extend([f"[Security] {s}" for s in security_issues])
        
        # 4. Class name check (warning level)
        is_valid, class_warning = self.validate_class_name(code)
        if class_warning:
            warnings.append(f"[Class Name] {class_warning}")
        
        # 5. Import check (warning level)
        is_valid, import_warnings = self.validate_imports(code)
        if import_warnings:
            warnings.extend([f"[Import] {w}" for w in import_warnings])
        
        # Final result
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get_validation_report(self, code: str, algorithm_type: str) -> str:
        """
        Get detailed validation report
        
        Args:
            code: Code string
            algorithm_type: Algorithm type
            
        Returns:
            Formatted validation report
        """
        result = self.validate_all(code, algorithm_type)
        
        report = "=" * 60 + "\n"
        report += "📋 Algorithm Code Validation Report\n"
        report += "=" * 60 + "\n\n"
        report += f"Algorithm Type: {algorithm_type}\n"
        report += f"Code Lines: {len(code.splitlines())}\n\n"
        report += str(result)
        report += "\n" + "=" * 60
        
        return report

