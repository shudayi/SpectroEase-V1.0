# -*- coding: utf-8 -*-
"""
Responsive Matplotlib Figure Utilities (V1.2.2)
为matplotlib图表提供响应式支持

使用方法：
    # 旧代码：
    # self.figure = Figure(figsize=(10, 6), dpi=100)
    
    # 新代码：
    from app.utils.responsive_matplotlib import create_responsive_figure
    self.figure = create_responsive_figure(base_width=10, base_height=6)
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from app.utils.ui_scaling import ui_scaling_manager


def create_responsive_figure(base_width=10, base_height=6):
    """
    创建响应式matplotlib图表
    
    Args:
        base_width: 基准宽度（inch），1920x1080下的理想宽度
        base_height: 基准高度（inch）
        
    Returns:
        Figure: 响应式配置的Figure对象
        
    Note:
        这个函数会根据屏幕分辨率自动调整：
        - figsize: 低分辨率缩小，4K放大
        - DPI: 低分辨率80-90，4K 144-240
    
    Example:
        >>> # 替换固定DPI的代码
        >>> # 旧: self.figure = Figure(figsize=(10, 6), dpi=100)
        >>> # 新:
        >>> self.figure = create_responsive_figure(base_width=10, base_height=6)
    """
    figsize, dpi = ui_scaling_manager.get_responsive_figsize(base_width, base_height)
    return Figure(figsize=figsize, dpi=dpi)


def apply_responsive_font_sizes():
    """
    应用响应式matplotlib字体大小
    
    这个函数会根据DPI自动调整所有matplotlib字体大小，
    确保在所有分辨率下都清晰可读。
    
    Note:
        - 在创建图表前调用
        - 会修改全局plt.rcParams
        - 字体大小基于DPI自动缩放
    
    Example:
        >>> from app.utils.responsive_matplotlib import apply_responsive_font_sizes
        >>> apply_responsive_font_sizes()
        >>> # 然后再创建图表
        >>> fig = create_responsive_figure(10, 6)
    """
    font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
    for key, value in font_sizes.items():
        plt.rcParams[key] = value


def get_responsive_font_sizes():
    """
    获取响应式字体大小字典（不自动应用）
    
    Returns:
        dict: matplotlib字体配置字典
        
    Note:
        如果您不想修改全局plt.rcParams，可以使用这个函数
        获取字体大小，然后手动应用到特定的axes或figure
    
    Example:
        >>> font_sizes = get_responsive_font_sizes()
        >>> ax.tick_params(labelsize=font_sizes['xtick.labelsize'])
    """
    return ui_scaling_manager.get_matplotlib_font_sizes()
