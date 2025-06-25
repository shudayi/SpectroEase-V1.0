# config/settings.py
import json
import os

class Settings:
    def __init__(self):
        self.settings_file = 'config/settings.json'
        self.theme = 'light'
        self.language = 'en'
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                data = json.load(f)
                self.theme = data.get('theme', 'light')
                self.language = data.get('language', 'en')

    def save_settings(self):
        data = {
            'theme': self.theme,
            'language': self.language
        }
        with open(self.settings_file, 'w') as f:
            json.dump(data, f, indent=4)
