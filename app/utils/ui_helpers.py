"""
UI Helper Functions for SpectroEase
UI工具函数库 - 简化常用UI组件的创建和配置
"""

from PyQt5.QtWidgets import (
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
    QLabel, QCheckBox, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox
)
from PyQt5.QtCore import Qt

try:
    from app.config.ui_design_tokens import UIDesignTokens as DT
except ImportError:
    # 如果设计令牌文件不存在，使用默认值
    class DT:
        HEIGHT_BUTTON_PRIMARY = 28
        HEIGHT_BUTTON_SECONDARY = 24
        HEIGHT_INPUT_CONTROL = 24
        HEIGHT_LABEL_INLINE = 24
        WIDTH_LABEL_STANDARD = 100
        WIDTH_CONTROL_STANDARD = 150
        WIDTH_CONTROL_WIDE = 200
        WIDTH_BUTTON_COMPACT = 80
        SPACING_STANDARD = 8
        SPACING_TIGHT = 4
        SPACING_RELAXED = 12
        MARGIN_STANDARD = (8, 8, 8, 8)


# ========== 按钮创建函数 ==========

def create_primary_button(text, width=None, callback=None):
    """
    创建主要操作按钮（高度28px）
    
    Args:
        text (str): 按钮文字
        width (int, optional): 按钮宽度，None为自适应
        callback (function, optional): 点击回调函数
        
    Returns:
        QPushButton: 配置好的按钮
        
    示例:
        apply_btn = create_primary_button("Apply", callback=self.apply_preprocessing)
    """
    button = QPushButton(text)
    button.setFixedHeight(DT.HEIGHT_BUTTON_PRIMARY)
    if width:
        button.setFixedWidth(width)
    if callback:
        button.clicked.connect(callback)
    return button


def create_secondary_button(text, width=None, callback=None):
    """
    创建次要操作按钮（高度24px）
    
    Args:
        text (str): 按钮文字
        width (int, optional): 按钮宽度
        callback (function, optional): 点击回调函数
        
    Returns:
        QPushButton: 配置好的按钮
    """
    button = QPushButton(text)
    button.setFixedHeight(DT.HEIGHT_BUTTON_SECONDARY)
    if width:
        button.setFixedWidth(width)
    if callback:
        button.clicked.connect(callback)
    return button


def create_compact_button(text, width=None, callback=None):
    """
    创建紧凑按钮（如Settings按钮，宽度80px）
    
    Args:
        text (str): 按钮文字
        width (int, optional): 按钮宽度，默认80px
        callback (function, optional): 点击回调函数
        
    Returns:
        QPushButton: 配置好的按钮
    """
    button = QPushButton(text)
    button.setFixedHeight(DT.HEIGHT_BUTTON_SECONDARY)
    button.setFixedWidth(width or DT.WIDTH_BUTTON_COMPACT)
    if callback:
        button.clicked.connect(callback)
    return button


# ========== 输入控件创建函数 ==========

def create_combobox(items, width='standard', current_index=0, callback=None):
    """
    创建标准ComboBox
    
    Args:
        items (list): 下拉选项列表
        width (str/int): 'standard' (180px), 'wide' (220px), 或具体像素值
        current_index (int): 默认选中索引
        callback (function, optional): 选项改变回调
        
    Returns:
        QComboBox: 配置好的下拉框
        
    示例:
        method_combo = create_combobox(['PCA', 'PLS', 'SVM'], width='standard')
    """
    combo = QComboBox()
    combo.addItems(items)
    combo.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)
    
    # 🎨 使用最小宽度而不是固定宽度，避免文字被截断
    if width == 'standard':
        combo.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)
    elif width == 'wide':
        combo.setMinimumWidth(DT.WIDTH_CONTROL_WIDE)
    elif isinstance(width, int):
        combo.setMinimumWidth(width)
    
    combo.setCurrentIndex(current_index)
    
    if callback:
        combo.currentTextChanged.connect(callback)
    
    return combo


def create_spinbox(min_val=0, max_val=100, default_val=0, step=1, width=None, callback=None):
    """
    创建标准SpinBox
    
    Args:
        min_val (int): 最小值
        max_val (int): 最大值
        default_val (int): 默认值
        step (int): 步长
        width (int, optional): 宽度，默认使用标准宽度
        callback (function, optional): 值改变回调
        
    Returns:
        QSpinBox: 配置好的数值输入框
    """
    spinbox = QSpinBox()
    spinbox.setRange(min_val, max_val)
    spinbox.setValue(default_val)
    spinbox.setSingleStep(step)
    spinbox.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)
    
    # 🎨 使用最小宽度，避免数值显示不全
    if width:
        spinbox.setMinimumWidth(width)
    else:
        spinbox.setMinimumWidth(DT.WIDTH_CONTROL_STANDARD)
    
    if callback:
        spinbox.valueChanged.connect(callback)
    
    return spinbox


def create_double_spinbox(min_val=0.0, max_val=100.0, default_val=0.0, 
                          step=0.1, decimals=2, width=None, callback=None):
    """
    创建标准DoubleSpinBox
    
    Args:
        min_val (float): 最小值
        max_val (float): 最大值
        default_val (float): 默认值
        step (float): 步长
        decimals (int): 小数位数
        width (int, optional): 宽度
        callback (function, optional): 值改变回调
        
    Returns:
        QDoubleSpinBox: 配置好的小数输入框
    """
    spinbox = QDoubleSpinBox()
    spinbox.setRange(min_val, max_val)
    spinbox.setValue(default_val)
    spinbox.setSingleStep(step)
    spinbox.setDecimals(decimals)
    spinbox.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)
    
    if width:
        spinbox.setFixedWidth(width)
    else:
        spinbox.setFixedWidth(DT.WIDTH_CONTROL_STANDARD)
    
    if callback:
        spinbox.valueChanged.connect(callback)
    
    return spinbox


def create_checkbox(text, checked=False, callback=None):
    """
    创建标准CheckBox
    
    Args:
        text (str): 复选框文字
        checked (bool): 是否默认选中
        callback (function, optional): 状态改变回调
        
    Returns:
        QCheckBox: 配置好的复选框
    """
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    checkbox.setFixedHeight(DT.HEIGHT_CHECKBOX)
    
    if callback:
        checkbox.toggled.connect(callback)
    
    return checkbox


# ========== 标签创建函数 ==========

def create_label(text, width='standard', align='right'):
    """
    创建标准标签（用于表单）
    
    Args:
        text (str): 标签文字
        width (str/int): 'standard' (110px), 'wide' (150px), 或具体像素值
        align (str): 对齐方式 'left', 'right', 'center'
        
    Returns:
        QLabel: 配置好的标签
    """
    label = QLabel(text)
    label.setFixedHeight(DT.HEIGHT_LABEL_INLINE)
    
    # 🎨 使用最小宽度，允许文字较长时自动扩展
    if width == 'standard':
        label.setMinimumWidth(DT.WIDTH_LABEL_STANDARD)
    elif width == 'wide':
        label.setMinimumWidth(DT.WIDTH_LABEL_WIDE)
    elif isinstance(width, int):
        label.setMinimumWidth(width)
    
    # 设置对齐
    align_map = {
        'left': Qt.AlignLeft | Qt.AlignVCenter,
        'right': Qt.AlignRight | Qt.AlignVCenter,
        'center': Qt.AlignCenter,
    }
    label.setAlignment(align_map.get(align, Qt.AlignRight | Qt.AlignVCenter))
    
    return label


# ========== 布局创建函数 ==========

def create_form_row(label_text, widget, label_width='standard'):
    """
    创建标准表单行（标签 + 控件）
    
    Args:
        label_text (str): 标签文字
        widget (QWidget): 控件
        label_width (str/int): 标签宽度
        
    Returns:
        QHBoxLayout: 包含标签和控件的水平布局
        
    示例:
        method_row = create_form_row("Method:", create_combobox(['PCA', 'PLS']))
        layout.addLayout(method_row)
    """
    layout = QHBoxLayout()
    layout.setSpacing(DT.SPACING_TIGHT)
    layout.setContentsMargins(0, 0, 0, 0)
    
    label = create_label(label_text, width=label_width, align='right')
    
    layout.addWidget(label)
    layout.addWidget(widget)
    layout.addStretch()
    
    return layout


def create_button_row(buttons, spacing='standard'):
    """
    创建按钮行
    
    Args:
        buttons (list): QPushButton列表
        spacing (str): 'tight' (4px), 'standard' (8px), 'relaxed' (12px)
        
    Returns:
        QHBoxLayout: 包含所有按钮的水平布局
        
    示例:
        btn_row = create_button_row([apply_btn, reset_btn, help_btn])
    """
    layout = QHBoxLayout()
    
    # 设置间距
    spacing_map = {
        'tight': DT.SPACING_TIGHT,
        'standard': DT.SPACING_STANDARD,
        'relaxed': DT.SPACING_RELAXED,
    }
    layout.setSpacing(spacing_map.get(spacing, DT.SPACING_STANDARD))
    layout.setContentsMargins(0, 0, 0, 0)
    
    for button in buttons:
        layout.addWidget(button)
    
    return layout


def create_standard_groupbox(title, height=None, margin='standard'):
    """
    创建标准组框
    
    Args:
        title (str): 组框标题
        height (int, optional): 固定高度，None为自适应
        margin (str): 边距类型 'tight', 'standard', 'relaxed'
        
    Returns:
        tuple: (QGroupBox, QVBoxLayout) 组框和其内部布局
        
    示例:
        group, layout = create_standard_groupbox("Preprocessing", height=200)
        layout.addWidget(some_widget)
    """
    groupbox = QGroupBox(title)
    layout = QVBoxLayout()
    
    # 设置边距
    margin_map = {
        'tight': DT.MARGIN_TIGHT,
        'standard': DT.MARGIN_STANDARD,
        'relaxed': DT.MARGIN_RELAXED,
    }
    layout.setContentsMargins(*margin_map.get(margin, DT.MARGIN_STANDARD))
    layout.setSpacing(DT.SPACING_STANDARD)
    
    groupbox.setLayout(layout)
    
    if height:
        groupbox.setFixedHeight(height)
    
    return groupbox, layout


# ========== 复合组件创建函数 ==========

def create_checkbox_with_params(checkbox_text, param_widgets, checked=False):
    """
    创建带参数容器的复选框组（复选框 + 可启用/禁用的参数区域）
    
    Args:
        checkbox_text (str): 复选框文字
        param_widgets (list): 参数控件列表 [(label, widget), ...]
        checked (bool): 是否默认选中
        
    Returns:
        tuple: (QVBoxLayout, QCheckBox, QWidget) 主布局、复选框、参数容器
        
    示例:
        layout, checkbox, container = create_checkbox_with_params(
            "Enable Smoothing",
            [
                ("Method:", method_combo),
                ("Window:", window_spin)
            ]
        )
    """
    main_layout = QVBoxLayout()
    main_layout.setSpacing(DT.SPACING_TIGHT)
    
    # 创建复选框
    checkbox = create_checkbox(checkbox_text, checked=checked)
    main_layout.addWidget(checkbox)
    
    # 创建参数容器
    param_container = QWidget()
    param_layout = QFormLayout(param_container)
    param_layout.setSpacing(DT.SPACING_TIGHT)
    param_layout.setContentsMargins(*DT.MARGIN_FORM_FIELD)
    param_layout.setLabelAlignment(Qt.AlignLeft)
    
    # 添加参数控件
    for label_text, widget in param_widgets:
        if isinstance(label_text, str):
            label = QLabel(label_text)
            param_layout.addRow(label, widget)
        else:
            param_layout.addRow(label_text, widget)
    
    main_layout.addWidget(param_container)
    
    # 连接复选框状态到参数容器的启用/禁用
    param_container.setEnabled(checked)
    checkbox.toggled.connect(param_container.setEnabled)
    
    return main_layout, checkbox, param_container


# ========== 应用设计令牌的便捷函数 ==========

def apply_design_tokens_to_button(button, button_type='primary', width=None):
    """
    为现有按钮应用设计令牌
    
    Args:
        button (QPushButton): 按钮对象
        button_type (str): 'primary', 'secondary', 'compact'
        width (int, optional): 宽度
    """
    height_map = {
        'primary': DT.HEIGHT_BUTTON_PRIMARY,
        'secondary': DT.HEIGHT_BUTTON_SECONDARY,
        'compact': DT.HEIGHT_BUTTON_SECONDARY,
    }
    button.setFixedHeight(height_map.get(button_type, DT.HEIGHT_BUTTON_PRIMARY))
    
    if width:
        button.setFixedWidth(width)
    elif button_type == 'compact':
        button.setFixedWidth(DT.WIDTH_BUTTON_COMPACT)


def apply_design_tokens_to_layout(layout, spacing='standard', margins='standard'):
    """
    为现有布局应用设计令牌
    
    Args:
        layout: QLayout对象
        spacing (str): 'tight', 'standard', 'relaxed'
        margins (str): 'tight', 'standard', 'relaxed', 'none'
    """
    spacing_map = {
        'tight': DT.SPACING_TIGHT,
        'standard': DT.SPACING_STANDARD,
        'relaxed': DT.SPACING_RELAXED,
        'none': DT.SPACING_NONE,
    }
    layout.setSpacing(spacing_map.get(spacing, DT.SPACING_STANDARD))
    
    margin_map = {
        'none': DT.MARGIN_NONE,
        'tight': DT.MARGIN_TIGHT,
        'standard': DT.MARGIN_STANDARD,
        'relaxed': DT.MARGIN_RELAXED,
    }
    layout.setContentsMargins(*margin_map.get(margins, DT.MARGIN_STANDARD))


# ========== 使用示例 ==========
"""
示例 1: 创建标准表单
    from app.utils.ui_helpers import *
    
    # 创建组框
    group, layout = create_standard_groupbox("Settings", height=200)
    
    # 添加表单行
    method_combo = create_combobox(['Method A', 'Method B'])
    layout.addLayout(create_form_row("Method:", method_combo))
    
    # 添加按钮行
    apply_btn = create_primary_button("Apply", callback=self.on_apply)
    reset_btn = create_secondary_button("Reset", callback=self.on_reset)
    layout.addLayout(create_button_row([apply_btn, reset_btn]))

示例 2: 创建带参数的复选框组
    smooth_layout, smooth_check, smooth_params = create_checkbox_with_params(
        "Enable Smoothing",
        [
            ("Method:", create_combobox(['S-Golay', 'Moving Avg'])),
            ("Window:", create_spinbox(3, 51, 11, 2))
        ]
    )
    parent_layout.addLayout(smooth_layout)

示例 3: 更新现有控件
    # 为现有按钮应用设计令牌
    apply_design_tokens_to_button(self.old_button, 'primary')
    
    # 为现有布局应用设计令牌
    apply_design_tokens_to_layout(self.old_layout, spacing='standard', margins='relaxed')
"""

