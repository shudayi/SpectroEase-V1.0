import sys
import os
import PyQt5

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from app.views.main_window import MainWindow
from app.utils.translator import Translator
from plugin_loader import load_plugins
from app.services.llm_service import LLMService
from config.llm_config import LLMConfig

def main():

    # Set QT_QPA_PLATFORM_PLUGIN_PATH to fix plugin loading issue
    pyqt_path = os.path.dirname(PyQt5.__file__)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(pyqt_path, 'Qt5', 'plugins')

    # import io
    # sys.stdout = io.StringIO()
    # sys.stderr = io.StringIO()
    
    # Enable High DPI Scaling
    # CRITICAL FIX: Attributes must be set BEFORE the application object is created.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
    app.setWindowIcon(QIcon(icon_path))

    # Set application font
    font = QFont("Arial", 9)
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
