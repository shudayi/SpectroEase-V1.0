from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QComboBox, QMessageBox)
from config.llm_config import LLMConfig

class LLMConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        self.setWindowTitle('LLM Configuration')
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # API key configuration
        api_layout = QHBoxLayout()
        api_label = QLabel('API Key:')
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_key_input)
        layout.addLayout(api_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(['deepseek-coder', 'deepseek-chat', 'other'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # API base URL
        url_layout = QHBoxLayout()
        url_label = QLabel('API URL:')
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText('https://api.deepseek.com/v1')
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # Other parameters
        params_layout = QHBoxLayout()
        temp_label = QLabel('Temperature:')
        self.temp_input = QLineEdit()
        self.temp_input.setPlaceholderText('0.7')
        max_tokens_label = QLabel('Max Tokens:')
        self.max_tokens_input = QLineEdit()
        self.max_tokens_input.setPlaceholderText('2000')
        params_layout.addWidget(temp_label)
        params_layout.addWidget(self.temp_input)
        params_layout.addWidget(max_tokens_label)
        params_layout.addWidget(self.max_tokens_input)
        layout.addLayout(params_layout)
        
        # Test connection button
        self.test_button = QPushButton('Test Connection')
        self.test_button.clicked.connect(self.test_connection)
        layout.addWidget(self.test_button)
        
        # Save and cancel buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save_config)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        self.api_key_input.setText(LLMConfig.API_KEY)
        
    def load_config(self):
        """从配置文件加载设置"""
        self.api_key_input.setText(LLMConfig.API_KEY)
        self.model_combo.setCurrentText(LLMConfig.MODEL_NAME)
        self.url_input.setText(LLMConfig.API_BASE_URL)
        self.temp_input.setText(str(LLMConfig.TEMPERATURE))
        self.max_tokens_input.setText(str(LLMConfig.MAX_TOKENS))
        
    def save_config(self):
        """保存配置到配置文件"""
        try:
            api_key = self.api_key_input.text()
            model_name = self.model_combo.currentText()
            api_base_url = self.url_input.text()
            temperature = float(self.temp_input.text() or "0.7")
            max_tokens = int(self.max_tokens_input.text() or "2000")
            
            LLMConfig.save_config(
                api_key=api_key,
                model_name=model_name,
                api_base_url=api_base_url,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            QMessageBox.information(self, 'Success', 'Configuration saved')
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error saving configuration: {str(e)}')
            
    async def test_connection(self):
        """测试LLM API连接"""
        try:
            from app.services.llm_service import LLMService
            service = LLMService(self.api_key_input.text())
            
  
            test_code = "def hello(): print('hello')"
            result = await service.convert_algorithm(
                algorithm_type="测试",
                source_code=test_code,
                instruction="这是一个测试"
            )
            
            if result:
                QMessageBox.information(self, 'Success', 'API connection test successful!')
            else:
                QMessageBox.warning(self, 'Warning', 'API connection test failed, please check configuration.')
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error during connection test: {str(e)}')

    def accept(self):
        # Save the API key when the dialog is accepted
        api_key = self.api_key_input.text().strip()
        LLMConfig.save_config(api_key)
        super().accept() 