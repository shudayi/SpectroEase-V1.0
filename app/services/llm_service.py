from typing import Dict, Any
import requests
import json
from config.llm_config import LLMConfig

class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, message: str, code: str = None) -> str:
        """
        Chat with the LLM
        """
        # Build prompt
        if code:
            prompt = f"""User message: {message}

Related code:
```python
{code}
```

Please provide a professional response, ensuring the code meets system requirements. If code modification is needed, ensure:
1. Code follows Python syntax standards
2. Code adheres to interface definition requirements
3. Maintains the core logic of the original algorithm
4. Provides complete code implementation without omitting any parts
5. Ensures code can be directly integrated into the system
"""
        else:
            prompt = f"""User message: {message}

Please provide a professional response. If code is involved, ensure the code meets system requirements:
1. Code follows Python syntax standards
2. Code adheres to interface definition requirements
3. Provides complete code implementation without omitting any parts
4. Ensures code can be directly integrated into the system
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": LLMConfig.MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": """You are a professional machine learning algorithm expert skilled in code conversion and optimization.
Your main task is to help users convert various machine learning algorithm code into system-compatible formats.

The system supports the following types of algorithms:
1. Preprocessing Algorithms (PreprocessingAlgorithm): For data cleaning, transformation, and standardization
2. Feature Selection Algorithms (FeatureSelectionAlgorithm): For selecting the most relevant features
3. Modeling Algorithms (ModelingAlgorithm): For training and predicting models
4. Data Partitioning Algorithms (DataPartitioningAlgorithm): For splitting data into training and test sets

Each algorithm has specific interface requirements. You need to ensure the converted code fully complies with these interfaces.
When converting, preserve the core logic of the original algorithm while adapting to the system interface.
"""},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": LLMConfig.TEMPERATURE,
                    "max_tokens": LLMConfig.MAX_TOKENS
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error calling Deepseek API: {str(e)}")
            return None 