# config/settings.py
import json
import os

class Settings:
    def __init__(self):
        self.settings_file = 'config/settings.json'
        self.theme = 'light'
        self.language = 'en'
        # 数据导入保守规则阈值（需求D）
        self.numeric_conversion_threshold = 0.7  # 70%数值转换阈值
        self.missing_rate_warning_threshold = 0.3  # 30%缺失率警戒线
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                data = json.load(f)
                self.theme = data.get('theme', 'light')
                self.language = data.get('language', 'en')
                # 加载数据导入阈值设置
                self.numeric_conversion_threshold = data.get('numeric_conversion_threshold', 0.7)
                self.missing_rate_warning_threshold = data.get('missing_rate_warning_threshold', 0.3)

    def save_settings(self):
        data = {
            'theme': self.theme,
            'language': self.language,
            # 保存数据导入阈值设置
            'numeric_conversion_threshold': self.numeric_conversion_threshold,
            'missing_rate_warning_threshold': self.missing_rate_warning_threshold
        }
        with open(self.settings_file, 'w') as f:
            json.dump(data, f, indent=4)
