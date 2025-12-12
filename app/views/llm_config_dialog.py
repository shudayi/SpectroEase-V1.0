from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QComboBox, QMessageBox)
from config.llm_config import LLMConfig
from app.views.responsive_dialog import ResponsiveDialog
import asyncio

class LLMConfigDialog(ResponsiveDialog):
    def __init__(self, parent=None):
        super().__init__(parent, base_width=500, base_height=400)
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        self.setWindowTitle('LLM Configuration')
        
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
        self.model_combo.addItems(['deepseek-coder', 'deepseek-chat', 'deepseek-reasoner', 'other'])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Custom model name input (shown when "other" is selected)
        self.custom_model_layout = QHBoxLayout()
        custom_model_label = QLabel('Custom Model:')
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText('e.g., gpt-4, claude-3-opus, etc.')
        self.custom_model_layout.addWidget(custom_model_label)
        self.custom_model_layout.addWidget(self.custom_model_input)
        layout.addLayout(self.custom_model_layout)
        self.custom_model_input.setVisible(False)  # Hidden by default
        
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
        self.test_button.clicked.connect(self._on_test_connection_clicked)
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
        
    def _on_model_changed(self, text):
        """Show/hide custom model input based on selection"""
        self.custom_model_input.setVisible(text == 'other')
        
    def load_config(self):
        """Load settings from config file"""
        self.api_key_input.setText(LLMConfig.API_KEY)
        
        # Load model name - check if it's a custom model
        model_name = LLMConfig.MODEL_NAME
        if model_name and model_name not in ['deepseek-coder', 'deepseek-chat', 'deepseek-reasoner']:
            # Custom model
            self.model_combo.setCurrentText('other')
            self.custom_model_input.setText(model_name)
        else:
            # Predefined model
            self.model_combo.setCurrentText(model_name or 'deepseek-coder')
            self.custom_model_input.setText('')
        
        self.url_input.setText(LLMConfig.API_BASE_URL)
        self.temp_input.setText(str(LLMConfig.TEMPERATURE))
        self.max_tokens_input.setText(str(LLMConfig.MAX_TOKENS))
        
    def save_config(self):
        """Save configuration to config file"""
        try:
            api_key = self.api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(self, 'Warning', 'Please enter an API key')
                return
            
            # Get model name - use custom input if "other" is selected
            model_name = self.model_combo.currentText()
            if model_name == 'other':
                model_name = self.custom_model_input.text().strip()
                if not model_name:
                    QMessageBox.warning(self, 'Warning', 'Please enter a custom model name')
                    return
            
            # Get and normalize URL (remove trailing slash)
            api_base_url = self.url_input.text().strip().rstrip('/')
            if not api_base_url:
                api_base_url = "https://api.deepseek.com/v1"
            
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
        except ValueError as e:
            QMessageBox.critical(self, 'Error', f'Invalid input format: {str(e)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error saving configuration: {str(e)}')
            
    def _on_test_connection_clicked(self):
        """Synchronous wrapper for async test_connection"""
        # 🔧 FIX: Run async function in a new event loop
        # This prevents the RuntimeWarning about unawaited coroutine
        try:
            # Disable button during test
            self.test_button.setEnabled(False)
            self.test_button.setText('Testing...')
            
            # Create new event loop for this async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.test_connection())
            finally:
                loop.close()
            
            # Re-enable button
            self.test_button.setEnabled(True)
            self.test_button.setText('Test Connection')
            
        except Exception as e:
            # Re-enable button on error
            self.test_button.setEnabled(True)
            self.test_button.setText('Test Connection')
            QMessageBox.critical(self, 'Error', f'Error during connection test: {str(e)}')
    
    async def test_connection(self):
        """Test LLM API connection (async implementation)"""
        try:
            from app.services.llm_service import LLMService
            
            # Get configuration values
            api_key = self.api_key_input.text().strip()
            
            # Validate API key
            if not api_key:
                QMessageBox.warning(self, 'Warning', 'Please enter an API key')
                return
            
            # Get and normalize URL (remove trailing slash)
            base_url = self.url_input.text().strip().rstrip('/') or None
            
            # Get model name - use custom input if "other" is selected
            model_name = self.model_combo.currentText()
            if model_name == 'other':
                model_name = self.custom_model_input.text().strip()
                if not model_name:
                    QMessageBox.warning(self, 'Warning', 'Please enter a custom model name')
                    return
            
            # 🔧 FIX: Use configured URL and model for testing
            service = LLMService(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )
            
            # Simple test: just try to make a minimal API call
            # Use a very simple test message to avoid long waits
            test_message = "Hello, this is a connection test. Please respond with 'OK'."
            result = await service.chat(
                message=test_message,
                code=None,
                algorithm_type=None
            )
            
            if result:
                QMessageBox.information(self, 'Success', 'API connection test successful!')
            else:
                QMessageBox.warning(self, 'Warning', 'API connection test failed, please check configuration.')
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error during connection test: {str(e)}')

    def accept(self):
        # 🔧 FIX: accept() is called from save_config(), so we don't need to save again here
        # Just call parent accept to close the dialog
        super().accept() 