import sys
import os


def get_base_path():
    """获取应用基础路径，兼容开发环境和打包环境"""
    if getattr(sys, 'frozen', False):
        # 打包后的EXE环境
        # sys._MEIPASS 是PyInstaller创建的临时目录，包含所有资源文件
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS
        return os.path.dirname(sys.executable)
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))


def get_resource_path(relative_path):
    """获取资源文件路径，兼容开发环境和打包环境"""
    return os.path.join(get_base_path(), relative_path)


project_root = get_base_path()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from app.views.main_window import MainWindow
from plugin_loader import load_plugins
from app.services.llm_service import LLMService
from config.llm_config import LLMConfig

def main():


    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Qt 5.14+: Smoother scaling policy
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # Older Qt versions don't support this, skip
    
    app = QApplication(sys.argv)

    icon_path = get_resource_path('logo.png')
    app.setWindowIcon(QIcon(icon_path))

    # V1.2.1: Dynamically adjust font size based on screen DPI
    screen = app.primaryScreen()
    if screen and screen.devicePixelRatio() >= 1.5:
        # High resolution screen uses slightly larger font
        base_font_size = 10
    else:
        # Standard resolution screen
        base_font_size = 9
    
    font = QFont("Arial", base_font_size)
    app.setFont(font)
    
    print(f"Application started: font size={base_font_size}pt, screen pixel ratio={screen.devicePixelRatio() if screen else 'N/A'}")
    
    # Initialize LLM service with proper error handling
    try:
        api_key = LLMConfig.API_KEY
        if not api_key or api_key.strip() == "":
            print("AI Assistant not configured (no API key) - feature disabled")
            print("You can still use all core features without AI Assistant")
            llm_service = None
        else:
            llm_service = LLMService(
                api_key=api_key,
                base_url=LLMConfig.API_BASE_URL,
                model_name=LLMConfig.MODEL_NAME
            )
            print("AI Assistant enabled")
    except Exception as e:
        print(f"Failed to initialize AI Assistant: {e}")
        llm_service = None

    plugin_base_dir = get_resource_path('plugins')
    preprocessing_plugins, feature_selection_plugins, modeling_plugins, data_partitioning_plugins = load_plugins(
        plugin_base_dir)

    main_window = MainWindow(
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
