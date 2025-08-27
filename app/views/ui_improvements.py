#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Professional UI/UX Improvements for SpectroEase
专业的SpectroEase界面optimization方案
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class ImprovedLayoutManager:
    """改进的布局管理器"""
    
    @staticmethod
    def create_form_layout_with_alignment():
        """创建对齐良好的表单布局"""
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)  # labels右对齐，垂直居中
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)        # 表单左对齐，顶部对齐
        layout.setVerticalSpacing(8)    # 垂直间距8px
        layout.setHorizontalSpacing(12) # 水平间距12px
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow) # 字段可扩展
        return layout
    
    @staticmethod  
    def create_compact_group_box(title, content_layout, min_height=None):
        """创建紧凑的分组框"""
        group = QGroupBox(title)
        group.setLayout(content_layout)
        
  
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(6)
        
        if min_height:
            group.setMinimumHeight(min_height)
            
        return group

class ModernStylesheet:
    """现代化样式表"""
    
    @staticmethod
    def get_improved_styles():
        """获取改进的样式表"""
        return """
        /* === 主要布局样式 === */
        QMainWindow {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', 'Microsoft YaHei UI', Arial, sans-serif;
            font-size: 9pt;
        }
        
        /* === 分组框optimization === */
        QGroupBox {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-top: 12px;
            background-color: #ffffff;
            padding-top: 8px;
            font-weight: 600;
            color: #495057;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            background-color: #ffffff;
            color: #343a40;
            font-size: 9pt;
            font-weight: 600;
        }
        
        /* === 表单控件对齐 === */
        QLabel {
            color: #495057;
            font-size: 9pt;
            padding: 2px;
            min-height: 20px;
        }
        
        QLabel[role="form-label"] {
            font-weight: 500;
            min-width: 100px;
            text-align: right;
        }
        
        /* === 输入控件统一样式 === */
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 6px 8px;
            background-color: #ffffff;
            font-size: 9pt;
            min-height: 20px;
            selection-background-color: #007bff;
        }
        
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
            border-color: #80bdff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.1);
        }
        
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
            border-color: #007bff;
            outline: none;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.15);
        }
        
        /* === 按钮现代化 === */
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 9pt;
            min-height: 24px;
        }
        
        QPushButton:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
        }
        
        QPushButton:pressed {
            background-color: #004085;
            transform: translateY(0);
        }
        
        QPushButton:disabled {
            background-color: #6c757d;
            color: #adb5bd;
        }
        
        QPushButton[role="secondary"] {
            background-color: #6c757d;
        }
        
        QPushButton[role="secondary"]:hover {
            background-color: #545b62;
        }
        
        QPushButton[role="success"] {
            background-color: #28a745;
        }
        
        QPushButton[role="success"]:hover {
            background-color: #1e7e34;
        }
        
        /* === 复选框optimization === */
        QCheckBox {
            spacing: 8px;
            font-size: 9pt;
            color: #495057;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 2px solid #ced4da;
            border-radius: 3px;
            background-color: #ffffff;
        }
        
        QCheckBox::indicator:hover {
            border-color: #007bff;
        }
        
        QCheckBox::indicator:checked {
            background-color: #007bff;
            border-color: #007bff;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjg1NCA0LjE0NkwxNC4xNDYgNC44NTRMNi41IDEyLjVMMS44NTQgNy44NTRMMi4xNDYgNy4xNDZMNi41IDExLjVMMTMuODU0IDQuMTQ2WiIgZmlsbD0id2hpdGUiLz4KPHN2Zz4K);
        }
        
        /* === 工具栏现代化 === */
        QToolBar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            spacing: 4px;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 4px;
        }
        
        QToolBar QToolButton {
            background-color: transparent;
            border: none;
            color: white;
            padding: 8px 12px;
            margin: 2px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 9pt;
            min-width: 80px;
        }
        
        QToolBar QToolButton:hover {
            background-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }
        
        QToolBar QToolButton:pressed {
            background-color: rgba(255, 255, 255, 0.25);
            transform: translateY(0);
        }
        
        /* === 表格现代化 === */
        QTableWidget {
            gridline-color: #e9ecef;
            background-color: #ffffff;
            selection-background-color: #e3f2fd;
            selection-color: #1976d2;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            font-size: 9pt;
        }
        
        QTableWidget QHeaderView::section {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 10px 8px;
            border: none;
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
            color: #495057;
            text-align: left;
        }
        
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #f1f3f4;
        }
        
        QTableWidget::item:selected {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        
        QTableWidget::item:hover {
            background-color: #f5f5f5;
        }
        
        /* === labels页optimization === */
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: #ffffff;
            margin-top: 2px;
        }
        
        QTabBar::tab {
            background: #f8f9fa;
            border: none;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            color: #6c757d;
            font-weight: 500;
            min-width: 100px;
        }
        
        QTabBar::tab:selected {
            background: #ffffff;
            color: #495057;
            font-weight: 600;
            border-bottom: 3px solid #007bff;
        }
        
        QTabBar::tab:hover:!selected {
            background: #e9ecef;
            color: #495057;
        }
        
        /* === 滚动条现代化 === */
        QScrollBar:vertical {
            border: none;
            background: #f8f9fa;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: #ced4da;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #adb5bd;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        
        /* === 状态栏optimization === */
        QStatusBar {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            font-size: 8pt;
            padding: 4px 8px;
        }
        
        QStatusBar QLabel {
            color: #6c757d;
            font-size: 8pt;
        }
        
        /* === 进度条optimization === */
        QProgressBar {
            border: none;
            border-radius: 8px;
            background-color: #e9ecef;
            text-align: center;
            color: #495057;
            height: 16px;
            font-size: 8pt;
            font-weight: 500;
        }
        
        QProgressBar::chunk {
            background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
            border-radius: 8px;
        }
        
        /* === 分割器optimization === */
        QSplitter::handle {
            background-color: #dee2e6;
        }
        
        QSplitter::handle:horizontal {
            width: 2px;
        }
        
        QSplitter::handle:vertical {
            height: 2px;
        }
        
        QSplitter::handle:hover {
            background-color: #007bff;
        }
        """

class ResponsiveLayoutHandler:
    """响应式布局processing器"""
    
    @staticmethod
    def setup_responsive_splitter(main_window):
        """设置响应式分割器"""
        splitter = main_window.centralWidget()
        if isinstance(splitter, QSplitter):
  
            splitter.setSizes([600, 1400])
            
  
            splitter.setChildrenCollapsible(False)
            
  
            splitter.setHandleWidth(3)
            splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #007bff;
                    border-radius: 1px;
                }
                QSplitter::handle:hover {
                    background-color: #0056b3;
                }
            """)
    
    @staticmethod
    def setup_auto_resize_table(table_widget):
        """设置表格自动调整列宽"""
        header = table_widget.horizontalHeader()
        
  
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)    # 第一列适应内容
        header.setSectionResizeMode(1, QHeaderView.Stretch)             # 第二列拉伸填充
        
  
        for i in range(2, table_widget.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Interactive)     # 其他列可交互调整
        
  
        for i in range(table_widget.columnCount()):
            table_widget.setColumnWidth(i, max(100, table_widget.columnWidth(i)))

class ToolbarEnhancer:
    """工具栏增强器"""
    
    @staticmethod
    def create_enhanced_toolbar_action(text, icon_path=None, tooltip=None, shortcut=None):
        """创建增强的工具栏动作"""
        action = QAction(text)
        
        if icon_path:
            action.setIcon(QIcon(icon_path))
        
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        
  
        action.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        return action
    
    @staticmethod
    def setup_toolbar_with_text_and_icons(toolbar):
        """设置带文字和图标的工具栏"""
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(20, 20))
        
  
        toolbar.setStyleSheet("""
            QToolBar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                spacing: 6px;
                padding: 10px 15px;
                border-radius: 10px;
                margin: 6px;
            }
            
            QToolBar QToolButton {
                background-color: transparent;
                border: none;
                color: white;
                padding: 10px 15px;
                margin: 2px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 9pt;
                min-width: 100px;
                text-align: left;
            }
            
            QToolBar QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.15);
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            QToolBar QToolButton:pressed {
                background-color: rgba(255, 255, 255, 0.25);
                transform: translateY(0);
            }
        """)

class TypographyManager:
    """字体排版管理器"""
    
    @staticmethod
    def get_typography_styles():
        """获取统一的字体排版样式"""
        return {
            'heading1': QFont('Segoe UI', 14, QFont.Bold),      # 主标题
            'heading2': QFont('Segoe UI', 12, QFont.DemiBold),  # 副标题  
            'heading3': QFont('Segoe UI', 10, QFont.DemiBold),  # 小标题
            'body': QFont('Segoe UI', 9, QFont.Normal),         # 正文
            'small': QFont('Segoe UI', 8, QFont.Normal),        # 小字
            'button': QFont('Segoe UI', 9, QFont.Medium),       # 按钮
            'input': QFont('Segoe UI', 9, QFont.Normal),        # 输入框
        }
    
    @staticmethod
    def apply_typography_to_widget(widget, style_name):
        """为控件applying字体样式"""
        fonts = TypographyManager.get_typography_styles()
        if style_name in fonts:
            widget.setFont(fonts[style_name])
    
    @staticmethod
    def get_spacing_standards():
        """获取间距标准"""
        return {
            'xs': 4,    # 极小间距
            'sm': 8,    # 小间距
            'md': 12,   # 中等间距  
            'lg': 16,   # 大间距
            'xl': 24,   # 超大间距
        }

def apply_professional_improvements(main_window):
    """applying专业的界面改进"""
    
  
    main_window.setStyleSheet(ModernStylesheet.get_improved_styles())
    
  
    ResponsiveLayoutHandler.setup_responsive_splitter(main_window)
    
  
    toolbar = main_window.findChild(QToolBar)
    if toolbar:
        ToolbarEnhancer.setup_toolbar_with_text_and_icons(toolbar)
    
  
    for table in main_window.findChildren(QTableWidget):
        ResponsiveLayoutHandler.setup_auto_resize_table(table)
    
  
    fonts = TypographyManager.get_typography_styles()
    main_window.setFont(fonts['body'])
    
    print("✅ Professional UI/UX improvements applied successfully!")

if __name__ == "__main__":
    print("UI/UX Improvement Module Ready")
    print("Use apply_professional_improvements(main_window) to apply all improvements") 