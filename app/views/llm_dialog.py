from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                            QPushButton, QLabel, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt

class LLMDialog(QDialog):
    def __init__(self, parent=None, llm_service=None):
        super().__init__(parent)
        self.llm_service = llm_service
        self.parent = parent  # Save parent window reference
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Algorithm Conversion and Integration')
        self.setMinimumSize(1000, 700)
        
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
            'Data Splitting'
        ])
        self.target_combo.currentIndexChanged.connect(self.show_template)
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)
        
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
        
        # Converted code
        result_label = QLabel('Conversion Result:')
        result_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.result_code = QTextEdit()
        self.result_code.setReadOnly(True)
        self.result_code.setStyleSheet("font-family: Consolas, Monaco, monospace; background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 3px;")
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
        
        # Initially show the first template
        self.show_template(0)
        
    def show_template(self, index):
        """Display the standard template for the selected section"""
        templates = {
            0: self.get_preprocessing_template(),
            1: self.get_feature_selection_template(),
            2: self.get_modeling_template(),
            3: self.get_data_splitting_template()
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
    
    def get_data_splitting_template(self):
        return """# Data Splitting Algorithm Interface Implementation Example
from interfaces.data_partitioning_algorithm import DataPartitioningAlgorithm
import pandas as pd

class CustomSplitter(DataPartitioningAlgorithm):
    def get_name(self) -> str:
        # Return unique algorithm name
        return "Custom Data Splitting Algorithm"
    
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
        # Implement data splitting logic
        # data: input data
        # params: parameter dictionary
        # return: dictionary containing train and test sets
        
        # Implement your data splitting logic here
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
        """Convert code"""
        source_code = self.source_code.toPlainText()
        if not source_code.strip():
            QMessageBox.warning(self, 'Warning', 'Please input source code')
            return
            
        self.convert_button.setEnabled(False)
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get the current selected template
            template = self.template_display.toPlainText()
            target_type = self.target_combo.currentText()
            
            # Build prompt message
            prompt = f"""Please convert the following code to match the target format.

Target format requirements:
{template}

Source code:
{source_code}

Conversion requirements:
1. Provide only the converted Python code, no explanations
2. Code must be a valid Python class that fully matches the template interface
3. Preserve core algorithm logic while adapting to the required interface
4. Class must have a unique name identifier
5. Ensure all required methods from the template are implemented
6. Code must be syntactically correct Python code
7. Parameter information should reflect the original algorithm's actual parameters
8. Ensure all method return types match the interface requirements

Please ensure the converted code can be directly integrated into the system while maintaining the original algorithm's functionality.
"""
            
            # Call LLM service to convert code
            result = loop.run_until_complete(self.llm_service.chat(message=prompt))
            
            if result:
  
                print("Generated Code:\n", result)
                
  
                try:
                    compile(result, '<string>', 'exec')
                    self.result_code.setPlainText(result)
                    self.integrate_button.setEnabled(True)
                except SyntaxError as e:
                    QMessageBox.warning(self, 'Error', f'Generated code has syntax errors: {str(e)}')
            else:
                QMessageBox.warning(self, 'Error', 'Conversion failed, please try again')
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error during conversion: {str(e)}')
            
        finally:
            self.convert_button.setEnabled(True)
            loop.close()
    
    def integrate_to_system(self):
        """Integrate the converted code into the system"""
        try:
            code = self.result_code.toPlainText()
            if not code.strip():
                return
                
            # Get the current selected section
            target_type = self.target_combo.currentText()
            
            # Call the parent window's method to add the algorithm
            if hasattr(self.parent, 'add_converted_algorithm'):
                self.parent.add_converted_algorithm(target_type, code)
                self.accept()  # Close dialog
            else:
                raise Exception("Parent window does not implement add_converted_algorithm method")
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error integrating algorithm: {str(e)}') 