import json
import os

class LLMConfig:
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
    
    API_KEY = ""
    MODEL_NAME = "deepseek-coder"
    API_BASE_URL = "https://api.deepseek.com/v1"
    MAX_TOKENS = 4000  # Increased for complex algorithm conversions
    TEMPERATURE = 0.3  # Lower for deterministic code generation

    @classmethod
    def load_config(cls):
        if os.path.exists(cls.CONFIG_PATH):
            with open(cls.CONFIG_PATH, 'r') as file:
                config = json.load(file)
                cls.API_KEY = config.get('api_key', '')
                cls.MODEL_NAME = config.get('model_name', 'deepseek-coder')
                cls.API_BASE_URL = config.get('api_base_url', 'https://api.deepseek.com/v1')
                cls.MAX_TOKENS = config.get('max_tokens', 2000)
                cls.TEMPERATURE = config.get('temperature', 0.3)  # Lower for code generation
                return cls.API_KEY
        return ''

    @classmethod
    def save_config(cls, api_key, model_name=None, api_base_url=None, max_tokens=None, temperature=None):
        config = {
            'api_key': api_key,
            'model_name': model_name or cls.MODEL_NAME,
            'api_base_url': api_base_url or cls.API_BASE_URL,
            'max_tokens': max_tokens or cls.MAX_TOKENS,
            'temperature': temperature or cls.TEMPERATURE
        }
        with open(cls.CONFIG_PATH, 'w') as file:
            json.dump(config, file, indent=4)
        
 
        cls.API_KEY = api_key
        if model_name:
            cls.MODEL_NAME = model_name
        if api_base_url:
            cls.API_BASE_URL = api_base_url
        if max_tokens:
            cls.MAX_TOKENS = max_tokens
        if temperature:
            cls.TEMPERATURE = temperature


LLMConfig.load_config() 