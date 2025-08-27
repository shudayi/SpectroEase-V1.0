import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon
from app.views.main_window import MainWindow
from app.utils.translator import Translator
from plugin_loader import load_plugins
from app.services.llm_service import LLMService
from config.llm_config import LLMConfig
from PyQt5.QtCore import Qt

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
     
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
    app.setWindowIcon(QIcon(icon_path))

    font = QFont("Arial", 10)
    app.setFont(font)

    translator = Translator(language='en')
    
    llm_service = LLMService(api_key=LLMConfig.API_KEY)

    plugin_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugins')
    preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins = load_plugins(
        plugin_base_dir)

    main_window = MainWindow(
        translator, 
        preprocessing_plugins, 
        feature_selection_plugins, 
        modeling_plugins,
        data_partitioning_plugins,
        llm_service
    )
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
