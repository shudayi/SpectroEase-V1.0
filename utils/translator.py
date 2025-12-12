# utils/translator.py

import json
import os

class Translator:
    def __init__(self, language='en'):
        self.language = language
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's absolute path
        translation_file = os.path.join(base_dir, '..', 'translations', f'{self.language}.json')
        translation_file = os.path.normpath(translation_file)  # Normalize path
        if os.path.exists(translation_file):
            try:
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
            except json.JSONDecodeError:
                self.translations = {}
                print(f"Error decoding JSON in translation file for language '{self.language}'.")
        else:
            self.translations = {}
            print(f"Translation file for language '{self.language}' not found.")

    def set_language(self, language):
        self.language = language
        self.load_translations()

    def translate(self, key):
        return self.translations.get(key, key)
