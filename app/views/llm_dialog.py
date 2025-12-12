from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                            QPushButton, QLabel, QComboBox, QMessageBox, QWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from app.views.responsive_dialog import ResponsiveDialog
from app.services.algorithm_validator import AlgorithmValidator  # V1.3.0: Add code validation
from app.utils.llm_code_checker import LLMCodeChecker  # V1.4.3: Add static pattern checker
import asyncio


class LLMWorker(QThread):
    """Background thread to handle LLM API calls without blocking UI"""
    finished = pyqtSignal(str)  # Success signal with result
    error = pyqtSignal(str)  # Error signal with error message
    
    def __init__(self, llm_service, prompt, source_code, algorithm_type):
        super().__init__()
        self.llm_service = llm_service
        self.prompt = prompt
        self.source_code = source_code
        self.algorithm_type = algorithm_type
    
    def run(self):
        """Run async task in background thread"""
        try:
            print("⏳ Background thread: Starting LLM API call...")
            # Create new event loop in background thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Execute async call
            result = loop.run_until_complete(
                self.llm_service.chat(
                    message=self.prompt,
                    code=self.source_code,
                    algorithm_type=self.algorithm_type
                )
            )
            loop.close()
            
            if result:
                print(f"✅ Background thread: LLM call succeeded, result length: {len(result)}")
                self.finished.emit(result)
            else:
                print("⚠️ Background thread: LLM returned None")
                self.error.emit("API call returned empty result")
                
        except Exception as e:
            print(f"❌ Background thread: Error occurred - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LLMDialog(ResponsiveDialog):
    def __init__(self, parent=None, llm_service=None):
        super().__init__(parent, base_width=1000, base_height=750)
        self.llm_service = llm_service
        self.parent = parent  # Save parent window reference
        self.validator = AlgorithmValidator()  # V1.3.0: Create validator instance
        self.code_checker = LLMCodeChecker()  # V1.4.3: Create static pattern checker
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Algorithm Conversion and Integration')
        
        layout = QVBoxLayout()
        
        # Target section selection
        target_layout = QHBoxLayout()
        target_label = QLabel('Select Target Module:')
        target_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.target_combo = QComboBox()
        self.target_combo.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        self.target_combo.addItems([
            'Preprocessing',
            'Feature Selection',
            'Modeling',
            'Data Partitioning'
        ])
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)
        
        # V1.3.2: Modeling task type selection (only shown when Modeling is selected)
        task_type_layout = QHBoxLayout()
        task_type_label = QLabel("Task Type:")
        task_type_label.setStyleSheet("font-weight: bold; color: #FF6B6B; font-size: 12px;")
        self.task_type_combo = QComboBox()
        self.task_type_combo.setStyleSheet("padding: 5px; border: 2px solid #FF6B6B; border-radius: 3px;")
        self.task_type_combo.addItems([
            'Qualitative (Classification)',
            'Quantitative (Regression)'
        ])
        task_type_layout.addWidget(task_type_label)
        task_type_layout.addWidget(self.task_type_combo)
        
        # Create a container widget to control show/hide
        self.task_type_widget = QWidget()
        self.task_type_widget.setLayout(task_type_layout)
        self.task_type_widget.setVisible(False)  # Hidden by default
        layout.addWidget(self.task_type_widget)
        
        # Template display
        template_label = QLabel('Standard Interface Template:')
        template_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.template_display = QTextEdit()
        self.template_display.setReadOnly(True)
        self.template_display.setMaximumHeight(200)
        self.template_display.setStyleSheet("font-family: Consolas, Monaco, monospace; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 3px;")
        layout.addWidget(template_label)
        layout.addWidget(self.template_display)
        
        # Source code input
        source_label = QLabel('Input Source Code:')
        source_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.source_code = QTextEdit()
        self.source_code.setPlaceholderText("Paste your algorithm code here...")
        self.source_code.setStyleSheet("font-family: Consolas, Monaco, monospace; border: 1px solid #ddd; border-radius: 3px;")
        layout.addWidget(source_label)
        layout.addWidget(self.source_code)
        
        # Converted code (editable)
        result_label = QLabel('Conversion Result (Editable):')
        result_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.result_code = QTextEdit()
        self.result_code.setReadOnly(False)  # Allow editing
        self.result_code.setPlaceholderText("Generated code will appear here. You can edit it before integrating to the system.")
        self.result_code.setStyleSheet("font-family: Consolas, Monaco, monospace; background-color: white; border: 1px solid #ddd; border-radius: 3px;")
        layout.addWidget(result_label)
        layout.addWidget(self.result_code)
        
        # Button area
        button_layout = QHBoxLayout()
        
        self.convert_button = QPushButton('Convert Code')
        self.convert_button.setStyleSheet("padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px;")
        self.convert_button.clicked.connect(self.convert_code)
        
        self.integrate_button = QPushButton('Integrate to System')
        self.integrate_button.setStyleSheet("padding: 8px 15px; background-color: #2196F3; color: white; border: none; border-radius: 4px;")
        self.integrate_button.clicked.connect(self.integrate_to_system)
        self.integrate_button.setEnabled(False)
        
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.integrate_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # V1.3.2: Check if task type selection needs to be shown on initialization
        self.on_target_changed(self.target_combo.currentIndex())
        
    def on_target_changed(self, index):
        """Callback when target algorithm type changes"""
        # Show corresponding template
        self.show_template(index)
        
        # V1.3.2: Only show task type selection when Modeling is selected
        current_text = self.target_combo.currentText()
        print(f"🔍 Target changed to: {current_text}")
        
        if current_text == 'Modeling':
            print(f"✅ Showing task type widget")
            self.task_type_widget.setVisible(True)
        else:
            print(f"⚠️ Hiding task type widget")
            self.task_type_widget.setVisible(False)
    
    def show_template(self, index):
        """Display the standard template for the selected section"""
        templates = {
            0: self.get_preprocessing_template(),
            1: self.get_feature_selection_template(),
            2: self.get_modeling_template(),
            3: self.get_data_partitioning_template()
        }
        self.template_display.setPlainText(templates.get(index, ""))
        
    def get_preprocessing_template(self):
        return """# Preprocessing Algorithm Interface Implementation Example
from interfaces.preprocessing_algorithm import PreprocessingAlgorithm
import pandas as pd

class CustomPreprocessor(PreprocessingAlgorithm):
    def get_name(self) -> str:
        # Return unique algorithm name
        return "Custom Preprocessing Algorithm"
    
    def get_parameter_info(self) -> dict:
        # Define algorithm parameters for UI generation
        return {
            'param1': {
                'type': 'int',
                'default': 5,
                'description': 'Parameter 1 description'
            },
            'param2': {
                'type': 'float',
                'default': 0.1,
                'description': 'Parameter 2 description'
            }
        }
    
    def apply(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        # Implement preprocessing logic
        # data: input data
        # params: parameter dictionary
        # return: processed data
        processed_data = data.copy()
        # Implement your preprocessing logic here
        return processed_data"""
    
    def get_feature_selection_template(self):
        return """# Feature Selection Algorithm Interface Implementation Example
from interfaces.feature_selection_algorithm import FeatureSelectionAlgorithm
import pandas as pd

class CustomFeatureSelector(FeatureSelectionAlgorithm):
    def get_name(self) -> str:
        # Return unique algorithm name
        return "Custom Feature Selection Algorithm"
    
    def get_parameter_info(self) -> dict:
        # Define algorithm parameters for UI generation
        return {
            'n_features': {
                'type': 'int',
                'default': 10,
                'description': 'Number of features to select'
            },
            'threshold': {
                'type': 'float',
                'default': 0.05,
                'description': 'Feature selection threshold'
            }
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, params: dict) -> list:
        # Implement feature selection logic
        # X: feature matrix
        # y: target variable
        # params: parameter dictionary
        # return: list of selected features
        
        # Implement your feature selection logic here
        selected_features = X.columns.tolist()[:params['n_features']]
        return selected_features"""
    
    def get_modeling_template(self):
        return """# Modeling Algorithm Interface Implementation Example
from interfaces.modeling_algorithm import ModelingAlgorithm
import pandas as pd

class CustomModel(ModelingAlgorithm):
    def get_name(self) -> str:
        # Return unique algorithm name
        return "Custom Model Algorithm"
    
    def get_parameter_info(self) -> dict:
        # Define algorithm parameters for UI generation
        return {
            'learning_rate': {
                'type': 'float',
                'default': 0.01,
                'description': 'Learning rate'
            },
            'max_depth': {
                'type': 'int',
                'default': 3,
                'description': 'Maximum depth'
            }
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, params: dict):
        # Implement model training logic
        # X: training features
        # y: training labels
        # params: parameter dictionary
        self.model = None  # Initialize and train your model here
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Implement prediction logic
        # X: prediction features
        # return: prediction results
        import numpy as np
        # Implement your prediction logic here
        return pd.Series(np.zeros(len(X)))"""
    
    def get_data_partitioning_template(self):
        return """# Data Partitioning Algorithm Interface Implementation Example
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
import pandas as pd

class CustomSplitter(DataPartitioningAlgorithm):
    def get_name(self) -> str:
        # Return unique algorithm name
        return "Custom Data Partitioning Algorithm"
    
    def get_parameter_info(self) -> dict:
        # Define algorithm parameters for UI generation
        return {
            'test_size': {
                'type': 'float',
                'default': 0.2,
                'description': 'Test set ratio'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random seed'
            }
        }
    
    def split_data(self, data: pd.DataFrame, params: dict) -> dict:
        # Implement data partitioning logic
        # data: input data
        # params: parameter dictionary
        # return: dictionary containing train and test sets
        
        # Implement your data partitioning logic here
        from sklearn.model_selection import train_test_split
        
        # Assume the last column is the target variable
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=params['random_state']
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }"""
    
    def convert_code(self):
        """Convert code - Using background thread to avoid blocking UI"""
        source_code = self.source_code.toPlainText()
        if not source_code.strip():
            QMessageBox.warning(self, 'Warning', 'Please input source code')
            return
            
        # Disable button to prevent duplicate clicks
        self.convert_button.setEnabled(False)
        self.convert_button.setText('Converting...')
        
        print("\n" + "="*60)
        print("🔧 V1.3.2: Starting convert_code (Non-blocking)...")
        print("="*60)
        
        # Get the current selected template
        template = self.template_display.toPlainText()
        target_type = self.target_combo.currentText()
        print(f"✅ Target type: {target_type}")
        
        # Save target_type for use in callback
        self.current_target_type = target_type
        
        # Build prompt message
        # V1.3.3: Add explicit base class inheritance requirement
        base_class_map = {
            'Preprocessing': 'PreprocessingAlgorithm',
            'Feature Selection': 'FeatureSelectionAlgorithm',
            'Modeling': 'ModelingAlgorithm',
            'Data Partitioning': 'DataPartitioningAlgorithm'
        }
        base_class = base_class_map.get(target_type, 'Algorithm')
        
        prompt = f"""Convert to {target_type} format.

SOURCE:
{source_code}

TEMPLATE:
{template}

⚠️ CRITICAL REQUIREMENTS ⚠️

1. **MANDATORY**: Class MUST inherit from {base_class}
   Example: class YourClassName({base_class}):
   
2. Output Python code only - NO markdown, NO explanations

3. Match method signatures EXACTLY

4. Return types must be EXACT:
   - PreprocessingAlgorithm.apply() -> pd.DataFrame (SAME shape, SAME index, SAME columns)
   - FeatureSelectionAlgorithm.select_features() -> List[str] (column names from X.columns)
   - ModelingAlgorithm.train() -> Any (model object, typically dict)
   - ModelingAlgorithm.predict() -> np.ndarray (1D array, length = X.shape[0])
   - DataPartitioningAlgorithm.partition() -> Tuple (4 elements: X_train, X_test, y_train, y_test)

5. ⚠️ **CRITICAL: DATA FLOW REQUIREMENTS**
   
   A) DataFrame Index and Columns:
      * MUST preserve index and columns in ALL operations
      * Preprocessing: return pd.DataFrame(result, index=data.index, columns=data.columns)
      * Feature Selection: return [X.columns[i] for i in selected_indices]  ← column names!
      * NEVER use reset_index(drop=True) unless absolutely necessary
   
   B) Data Format:
      * Input: Always pandas DataFrame/Series
      * Output: 
        - Preprocessing: DataFrame (not numpy array)
        - Feature Selection: List[str] (not indices, not numpy array)
        - Modeling predict: np.ndarray (not DataFrame, not Series)
   
   C) Shape Consistency:
      * Preprocessing: input.shape == output.shape
      * Feature Selection: len(output) == number of selected features
      * Modeling predict: len(output) == X.shape[0] (number of samples)
   
   D) Internal Transpose Handling:
      * If you transpose data internally for calculations, you MUST transpose back
      * Always restore original format: pd.DataFrame(array, index=original_index, columns=original_columns)

6. Include ALL imports at top (pandas as pd, numpy as np, etc.)

7. get_params_info() must use this format:
   {{'param': {{'type': 'int'|'float'|'str'|'bool', 'default': value, 'description': 'text'}}}}

8. Use descriptive class name (not CustomAlgorithm)

9. ⚠️ **MANDATORY: CODE ANALYSIS BEFORE CONVERSION**
   
   BEFORE writing code, analyze the SOURCE algorithm:
   
   STEP 1: Identify mathematical operations:
   - List ALL division operations → need numerical stability checks
   - List ALL matrix multiplications → verify dimensions and intent
   - List ALL decompositions (PCA, SVD, eigen) → verify what is computed
   - List ALL polynomial/regression fits → verify coefficient order
   - List ALL statistical operations → verify axis parameters
   
   STEP 2: Identify potential issues:
   - Division operations: Can denominator be zero/near-zero?
   - Matrix operations: Do dimensions match mathematical intent?
   - Transpose operations: Do they preserve mathematical meaning?
   - Index operations: Correct 0-based vs 1-based conversion?
   
   STEP 3: Map to Python equivalents:
   - For each operation, identify correct Python/NumPy function
   - Verify coefficient orders, matrix dimensions, axis parameters
   - Verify mathematical equivalence

10. ⚠️ **MANDATORY: UNIVERSAL NUMERICAL STABILITY CONSTRAINTS**
   
   These apply to ALL algorithms, regardless of type:
   
   A) Division Operations (MANDATORY):
      * For EVERY division (a / b), add stability check:
        denominator = <value>
        if abs(denominator) < 1e-8:
            result = <safe_fallback>
        else:
            result = numerator / denominator
      * Applies to: MSC, normalizations, ratios, inversions, etc.
   
   B) Matrix Operations (MANDATORY):
      * For EVERY matrix multiplication, verify:
        1. What mathematical quantity are you computing?
        2. What should output shape be?
        3. Does operation match mathematical intent?
      * Common pattern - Covariance (features × features):
        * If X shape (n_features, n_samples):
          CORRECT: Cov = (X @ X.T) / (n_samples - 1)  → (n_features, n_features)
          WRONG: Cov = (X.T @ X) / (n_samples - 1)  → (n_samples, n_samples) ❌
      * Add shape verification comments in code
   
   C) Polynomial Fitting (MANDATORY):
      * Identify function: numpy.polyfit vs numpy.polynomial.polynomial.polyfit
      * Verify coefficient order before using
      * Test reconstruction: y = p[0] + p[1]*x OR p[1] + p[0]*x?
   
   D) Statistical Operations (MANDATORY):
      * Verify axis parameter matches intent:
        - axis=0: across rows (samples)
        - axis=1: across columns (features)
      * Add comments explaining axis choice
   
   E) Decomposition (MANDATORY):
      * Verify: feature covariance vs sample covariance
      * Verify: which components/vectors selected
      * Add verification comments for matrix shapes
   
   F) Hyperparameter Selection (MANDATORY):
      * Use: best_k = int(np.argmin(cv_scores)) + 1
      * DO NOT use complex indexing or array slicing

11. ⚠️ **PRESERVE ORIGINAL ALGORITHM LOGIC**:
   - DO NOT modify the core algorithm implementation
   - Keep all mathematical operations, formulas, and calculations EXACTLY as in SOURCE code
   - Maintain the same data processing steps and order
   - Preserve all parameter names and their meanings
   - Only adapt code structure (class inheritance, method signatures) to match interface
   - The algorithm logic itself must remain UNCHANGED

12. Handle edge cases (empty data, None values)

13. No __init__ parameters (or use default values only)

OUTPUT FORMAT:
```python
import pandas as pd
import numpy as np
from typing import Dict, Any, ...

class YourDescriptiveName({base_class}):
    def get_name(self) -> str:
        ...
    def get_params_info(self) -> Dict[str, Any]:
        ...
    # ... other required methods
```
"""
        
        print("✅ Prompt built, starting background worker...")
        print(f"✅ LLM service: {self.llm_service}")
        
        # Create and start background worker thread
        self.worker = LLMWorker(
            llm_service=self.llm_service,
            prompt=prompt,
            source_code=source_code,
            algorithm_type=target_type
        )
        
        # Connect signals
        self.worker.finished.connect(self.on_llm_success)
        self.worker.error.connect(self.on_llm_error)
        
        # Start background thread
        self.worker.start()
        print("⏳ Background thread started, UI remains responsive...")
    
    def on_llm_success(self, result):
        """Handle LLM call success callback"""
        try:
            target_type = self.current_target_type
            print(f"✅ LLM call succeeded, result length: {len(result)}")
            
            if result:
                # V1.3.0: Validate generated code
                print("\n" + "="*60)
                print("🔍 Starting code validation...")
                print("="*60)
                
                validation_result = self.validator.validate_all(result, target_type)
                print(self.validator.get_validation_report(result, target_type))
                
                # V1.4.3: Additional static pattern checking
                print("\n" + "="*60)
                print("🔍 Starting static pattern checking...")
                print("="*60)
                checker_issues = self.code_checker.check(result, target_type)
                checker_report = self.code_checker.get_report()
                print(checker_report)
                
                # Combine validation and checker results
                has_validation_errors = not validation_result.is_valid
                has_checker_errors = self.code_checker.has_errors()
                
                if has_validation_errors or has_checker_errors:
                    # Show combined errors
                    error_msg = "LLM generated code has the following issues:\n\n"
                    
                    if has_validation_errors:
                        error_msg += "[Interface Validation Errors]\n"
                        for i, error in enumerate(validation_result.errors, 1):
                            error_msg += f"{i}. {error}\n"
                        error_msg += "\n"
                    
                    if has_checker_errors:
                        error_msg += "[Static Pattern Check Errors]\n"
                        for i, issue in enumerate([x for x in checker_issues if x.severity == 'error'], 1):
                            error_msg += f"{i}. [{issue.pattern}] {issue.message}"
                            if issue.line_number > 0:
                                error_msg += f" (line: {issue.line_number})"
                            error_msg += "\n"
                        error_msg += "\n"
                    
                    error_msg += "⚠️ You can edit the generated code in the 'Conversion Result' box above, or modify the source code and retry conversion."
                    
                    QMessageBox.critical(self, 'Code Validation Failed', error_msg)
                    # Still allow user to edit and integrate even if validation fails
                    self.result_code.setPlainText(result)
                    self.integrate_button.setEnabled(True)  # Enable integrate button so user can edit and integrate
                    self.convert_button.setEnabled(True)
                    return
                
                # If there are warnings, show but allow continue
                all_warnings = []
                if validation_result.warnings:
                    all_warnings.extend([f"Interface validation: {w}" for w in validation_result.warnings])
                checker_warnings = [x for x in checker_issues if x.severity == 'warning']
                if checker_warnings:
                    all_warnings.extend([f"Pattern check: {w.message}" for w in checker_warnings])
                
                if all_warnings:
                    warning_msg = "Code validation passed, but with the following suggestions:\n\n"
                    for i, warning in enumerate(all_warnings, 1):
                        warning_msg += f"{i}. {warning}\n"
                    warning_msg += "\n✅ You can continue using this code, or optimize based on suggestions."
                    
                    reply = QMessageBox.question(
                        self, 'Code Validation Warning', 
                        warning_msg + "\n\nContinue with this code?",
                        QMessageBox.Yes | QMessageBox.No, 
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.No:
                        self.convert_button.setEnabled(True)
                        return
                else:
                    # Perfect validation pass
                    print("✅ Code validation completely passed!")
  
                print("Generated Code:\n", result)
                
                # Always set the code in the editable text box
                try:
                    compile(result, '<string>', 'exec')
                    self.result_code.setPlainText(result)
                    self.integrate_button.setEnabled(True)
                    # Show info message that code is editable
                    QMessageBox.information(
                        self, 
                        'Code Generated', 
                        'Code has been generated and is ready for integration.\n\n'
                        '💡 Tip: You can edit the code in the "Conversion Result" box before integrating.'
                    )
                except SyntaxError as e:
                    # Even if syntax error, allow user to edit
                    self.result_code.setPlainText(result)
                    self.integrate_button.setEnabled(True)
                    QMessageBox.warning(
                        self, 
                        'Syntax Error', 
                        f'Generated code has syntax errors: {str(e)}\n\n'
                        'You can edit the code in the "Conversion Result" box to fix the errors.'
                    )
            else:
                QMessageBox.warning(self, 'Error', 'Conversion failed, please try again')
        
        except Exception as e:
            error_msg = f'Error during result processing: {type(e).__name__}: {str(e)}'
            print(f"\n❌ CRITICAL ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', error_msg)
        
        finally:
            # Restore button state
            self.convert_button.setEnabled(True)
            self.convert_button.setText('Convert Code')
    
    def on_llm_error(self, error_message):
        """Handle LLM call failure callback"""
        print(f"❌ LLM call failed: {error_message}")
        
        if "API call returned empty result" in error_message:
            QMessageBox.warning(
                self, 
                'LLM Conversion Failed',
                "LLM API call returned empty result.\n\n"
                "Possible reasons:\n"
                "1. Request timeout (120 seconds)\n"
                "2. API key invalid or expired\n"
                "3. Code too complex\n"
                "4. API service unavailable\n\n"
                "Please check console logs for details."
            )
        else:
            QMessageBox.warning(
                self, 
                'LLM Conversion Failed',
                f"Code conversion failed:\n\n{error_message}\n\n"
                "Possible reasons:\n"
                "1. Network timeout (please retry)\n"
                "2. API key invalid or expired\n"
                "3. Code too complex (try simplifying)\n"
                "4. API service unavailable"
            )
        
        # Restore button state
        self.convert_button.setEnabled(True)
        self.convert_button.setText('Convert Code')
    
    def integrate_to_system(self):
        """Integrate the converted code into the system"""
        try:
            code = self.result_code.toPlainText()
            if not code.strip():
                return
            
            # Show progress
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            progress = QProgressDialog("Integrating algorithm...", None, 0, 0, self)
            progress.setWindowTitle("Please Wait")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            
            try:
                # Get the current selected section
                target_type = self.target_combo.currentText()
                
                # V1.3.2: If Modeling, get task type
                task_type = None
                if target_type == 'Modeling':
                    task_type_text = self.task_type_combo.currentText()
                    print(f"🔍 LLM Dialog: Selected task type text: '{task_type_text}'")
                    # Support Qualitative/Classification and Quantitative/Regression
                    if 'Qualitative' in task_type_text or 'Classification' in task_type_text:
                        task_type = 'classification'
                    elif 'Quantitative' in task_type_text or 'Regression' in task_type_text:
                        task_type = 'regression'
                    else:
                        # Default to classification if unclear
                        task_type = 'classification'
                    print(f"✅ LLM Dialog: Determined task_type: '{task_type}'")
                
                # Call the parent window's method to add the algorithm
                # Use EXE-compatible loading method
                if hasattr(self.parent, 'add_converted_algorithm'):
                    print(f"📤 LLM Dialog: Calling add_converted_algorithm with task_type='{task_type}'")
                    self.parent.add_converted_algorithm(target_type, code, task_type=task_type)
                    progress.close()
                    self.accept()  # Close dialog
                else:
                    raise Exception("Parent window does not implement add_converted_algorithm method")
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error integrating algorithm: {str(e)}') 