from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt
from app.utils.ui_scaling import ui_scaling_manager


class ResponsiveDialog(QDialog):
    def __init__(self, parent=None, base_width=800, base_height=600):
        super().__init__(parent)
        self.base_width = base_width
        self.base_height = base_height
        self.setup_responsive_size()
    
    def setup_responsive_size(self):
        """设置响应式尺寸和位置"""
        width, height = ui_scaling_manager.get_responsive_dialog_size(
            self.base_width, self.base_height
        )
        
        min_width = max(400, int(self.base_width * 0.5))
        min_height = max(300, int(self.base_height * 0.5))
        self.setMinimumSize(min_width, min_height)
        
        max_width = int(self.base_width * 2.0)
        max_height = int(self.base_height * 2.0)
        self.setMaximumSize(max_width, max_height)
        
        self.resize(width, height)
        
        self.center_on_screen()
    
    def center_on_screen(self):
        """在屏幕中央显示对话框"""
        screen_geometry = ui_scaling_manager.screen_info["available_geometry"]
        
        x = screen_geometry.x() + (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
        
        x = max(screen_geometry.x(), x)
        y = max(screen_geometry.y(), y)
        
        self.move(x, y)
    
    def showEvent(self, event):
        """对话框显示时重新居中（处理多显示器情况）"""
        super().showEvent(event)
        
        screen_geometry = ui_scaling_manager.screen_info["available_geometry"]
        dialog_geometry = self.geometry()
        
        if not screen_geometry.contains(dialog_geometry.center()):
            self.center_on_screen()

