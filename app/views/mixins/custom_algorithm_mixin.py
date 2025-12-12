"""
Custom Algorithm Mixin
自定义算法UI集成混入类 - 统一所有view的自定义算法添加机制
"""

from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QCheckBox, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from typing import Callable, Optional


class CustomAlgorithmMixin:
    """
    统一的自定义算法UI集成Mixin
    
    所有需要支持自定义算法的View都应该继承此Mixin
    
    使用方法:
    ```python
    class MyView(QWidget, CustomAlgorithmMixin):
        def __init__(self):
            super().__init__()
            self.plugins = {}  # 必须有plugins字典
            self.method_checkboxes = {}  # 必须有复选框字典
            # ... 其他初始化
    ```
    """
    
    def add_custom_algorithm_ui(self, 
                                algorithm_name: str, 
                                layout_container,
                                on_checkbox_clicked: Optional[Callable] = None,
                                on_params_clicked: Optional[Callable] = None,
                                is_custom: bool = True):
        """
        统一的UI添加方法 - 添加自定义算法到UI
        
        Args:
            algorithm_name: 算法名称
            layout_container: 要添加到的布局容器
            on_checkbox_clicked: 复选框点击回调函数 (method_name, checked)
            on_params_clicked: 参数按钮点击回调函数 (method_name)
            is_custom: 是否标记为自定义算法
        """
        # 创建方法框架
        method_frame = QFrame()
        method_frame.setFrameShape(QFrame.StyledPanel)
        
        # 自定义算法使用特殊样式
        if is_custom:
            method_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 #E3F2FD,
                        stop:1 #f8f8f8
                    );
                    border-left: 3px solid #2196F3;
                    border-radius: 3px;
                    padding: 4px;
                }
                QFrame:hover {
                    background: #E3F2FD;
                    border-left: 3px solid #1976D2;
                }
            """)
        else:
            method_frame.setStyleSheet("""
                QFrame {
                    background-color: #f8f8f8;
                    border-radius: 3px;
                    padding: 4px;
                }
            """)
        
        method_layout = QHBoxLayout(method_frame)
        method_layout.setContentsMargins(3, 3, 3, 3)
        method_layout.setSpacing(4)
        
        # 创建复选框
        checkbox = QCheckBox(algorithm_name)
        checkbox.setStyleSheet("font-size: 11px;")
        
        # 自定义算法标识（已移除图标）
        if is_custom:
            checkbox.setText(algorithm_name)
            checkbox.setToolTip(f"自定义算法: {algorithm_name}\n✅ 由LLM转换或用户定义\n⚙️ 点击'Parameters'配置参数")
        else:
            checkbox.setToolTip(f"内置算法: {algorithm_name}")
        
        # 连接复选框点击事件
        if on_checkbox_clicked:
            checkbox.clicked.connect(lambda checked: on_checkbox_clicked(algorithm_name, checked))
        
        # 存储复选框引用
        if hasattr(self, 'method_checkboxes'):
            self.method_checkboxes[algorithm_name] = checkbox
        
        method_layout.addWidget(checkbox)
        
        # 创建参数按钮
        params_button = QPushButton("Parameters")
        params_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: none;
                border-radius: 2px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        params_button.setFixedWidth(70)
        
        # 连接参数按钮点击事件
        if on_params_clicked:
            params_button.clicked.connect(lambda: on_params_clicked(algorithm_name))
        
        method_layout.addWidget(params_button)
        
        # 添加到容器
        layout_container.addWidget(method_frame)
        
        return method_frame, checkbox, params_button
    
    def mark_algorithm_as_custom(self, algorithm_name: str):
        """
        将已存在的算法标记为自定义
        
        Args:
            algorithm_name: 算法名称
        """
        if hasattr(self, 'method_checkboxes') and algorithm_name in self.method_checkboxes:
            checkbox = self.method_checkboxes[algorithm_name]
            
            # 更新文本（图标已移除）
            checkbox.setText(algorithm_name)
            
            # 更新工具提示
            checkbox.setToolTip(f"自定义算法: {algorithm_name}\n✅ 由LLM转换或用户定义\n⚙️ 点击'Parameters'配置参数")
            
            # 更新父框架样式
            parent_frame = checkbox.parent()
            if parent_frame and isinstance(parent_frame, QFrame):
                parent_frame.setStyleSheet("""
                    QFrame {
                        background: qlineargradient(
                            x1:0, y1:0, x2:1, y2:0,
                            stop:0 #E3F2FD,
                            stop:1 #f8f8f8
                        );
                        border-left: 3px solid #2196F3;
                        border-radius: 3px;
                        padding: 4px;
                    }
                    QFrame:hover {
                        background: #E3F2FD;
                        border-left: 3px solid #1976D2;
                    }
                """)
    
    def refresh_algorithm_list(self):
        """
        刷新算法列表 - 重新构建UI
        
        子类应该重写此方法来实现具体的刷新逻辑
        """
        pass
    
    def remove_algorithm_from_ui(self, algorithm_name: str):
        """
        从UI中移除算法
        
        Args:
            algorithm_name: 算法名称
        """
        if hasattr(self, 'method_checkboxes') and algorithm_name in self.method_checkboxes:
            checkbox = self.method_checkboxes[algorithm_name]
            
            # 获取父框架并移除
            parent_frame = checkbox.parent()
            if parent_frame:
                parent_frame.deleteLater()
            
            # 从字典中删除
            del self.method_checkboxes[algorithm_name]
        
        # 从plugins中删除
        if hasattr(self, 'plugins') and algorithm_name in self.plugins:
            del self.plugins[algorithm_name]
    
    def get_custom_algorithms_count(self) -> int:
        """
        获取自定义算法数量
        
        Returns:
            自定义算法数量
        """
        if not hasattr(self, 'method_checkboxes'):
            return 0
        
        # 由于移除了图标，现在所有在method_checkboxes中的都是自定义算法
        # 或者可以通过其他方式判断（比如检查plugins字典）
        return len(self.method_checkboxes)
    
    def list_custom_algorithms(self) -> list:
        """
        列出所有自定义算法名称
        
        Returns:
            自定义算法名称列表
        """
        if not hasattr(self, 'method_checkboxes'):
            return []
        
        # 由于移除了图标，返回所有算法名称
        # 或者可以通过其他方式判断（比如检查plugins字典）
        return list(self.method_checkboxes.keys())
    
    def show_custom_algorithm_info(self):
        """
        显示自定义算法信息对话框
        """
        count = self.get_custom_algorithms_count()
        algos = self.list_custom_algorithms()
        
        if count == 0:
            msg = "当前没有自定义算法。\n\n您可以通过以下方式添加:\n"
            msg += "1. Tools → Algorithm Conversion (LLM) - 使用AI转换现有算法\n"
            msg += "2. Tools → Custom Algorithm Manager - 管理已有的自定义算法"
        else:
            msg = f"当前有 {count} 个自定义算法:\n\n"
            for i, name in enumerate(algos, 1):
                msg += f"{i}. {name}\n"
            msg += "\n💡 提示: 自定义算法已集成到系统中"
        
        QMessageBox.information(self, "自定义算法信息", msg)

