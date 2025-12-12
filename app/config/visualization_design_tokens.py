"""
可视化设计令牌 - Matplotlib图表的响应式参数配置
根据屏幕分辨率和DPI自动调整字体、线宽、标记大小等
"""
from app.utils.ui_scaling import ui_scaling_manager


class VisualizationDesignTokens:
    """
    可视化设计令牌 - 统一管理所有matplotlib图表的视觉参数
    
    根据屏幕分辨率自动缩放以下参数：
    - 字体大小（标题、标签、刻度、图例、文本注释）
    - 线宽（数据线、网格线、边框线）
    - 标记大小（散点图、数据点）
    - 图形尺寸和DPI
    """
    
    def __init__(self):
        """初始化时获取当前DPI缩放系数"""
        self._update_scale()
    
    def _update_scale(self):
        """更新缩放系数（基于ui_scaling_manager）"""
        # 获取matplotlib字体大小配置（已经根据DPI调整）
        self.font_config = ui_scaling_manager.get_matplotlib_font_sizes()
        
        # 计算通用缩放系数（相对于96 DPI的基准）
        self.scale = ui_scaling_manager.font_scale
        
    # ========== 字体大小系统 (Font Sizes) ==========
    
    @property
    def FONT_TITLE_LARGE(self):
        """大标题字体 (16px @ 96dpi)"""
        return max(14, min(int(16 * self.scale), 20))
    
    @property
    def FONT_TITLE_MEDIUM(self):
        """中标题字体 (14px @ 96dpi)"""
        return max(12, min(int(14 * self.scale), 18))
    
    @property
    def FONT_TITLE_SMALL(self):
        """小标题字体 (13px @ 96dpi)"""
        return max(11, min(int(13 * self.scale), 17))
    
    @property
    def FONT_LABEL_LARGE(self):
        """大标签字体 (14px @ 96dpi) - 用于主轴标签"""
        return max(12, min(int(14 * self.scale), 18))
    
    @property
    def FONT_LABEL_MEDIUM(self):
        """中标签字体 (12px @ 96dpi) - 用于次要轴标签"""
        return max(10, min(int(12 * self.scale), 16))
    
    @property
    def FONT_LABEL_SMALL(self):
        """小标签字体 (11px @ 96dpi)"""
        return max(9, min(int(11 * self.scale), 15))
    
    @property
    def FONT_TICK_LARGE(self):
        """大刻度字体 (12px @ 96dpi)"""
        return self.font_config.get('xtick.labelsize', 12)
    
    @property
    def FONT_TICK_MEDIUM(self):
        """中刻度字体 (10px @ 96dpi)"""
        return max(8, min(int(10 * self.scale), 14))
    
    @property
    def FONT_TICK_SMALL(self):
        """小刻度字体 (9px @ 96dpi)"""
        return max(7, min(int(9 * self.scale), 13))
    
    @property
    def FONT_TICK_TINY(self):
        """微小刻度字体 (7px @ 96dpi) - 用于大型混淆矩阵"""
        return max(6, min(int(7 * self.scale), 11))
    
    @property
    def FONT_LEGEND(self):
        """图例字体"""
        return self.font_config.get('legend.fontsize', 11)
    
    @property
    def FONT_TEXT_LARGE(self):
        """大文本注释 (11px @ 96dpi)"""
        return max(9, min(int(11 * self.scale), 15))
    
    @property
    def FONT_TEXT_MEDIUM(self):
        """中文本注释 (9px @ 96dpi)"""
        return max(7, min(int(9 * self.scale), 13))
    
    @property
    def FONT_TEXT_SMALL(self):
        """小文本注释 (8px @ 96dpi)"""
        return max(6, min(int(8 * self.scale), 12))
    
    @property
    def FONT_CONFUSION_MATRIX_LARGE(self):
        """混淆矩阵单元格文字 - 大 (8px @ 96dpi)"""
        return max(7, min(int(8 * self.scale), 12))
    
    @property
    def FONT_CONFUSION_MATRIX_MEDIUM(self):
        """混淆矩阵单元格文字 - 中 (6px @ 96dpi)"""
        return max(5, min(int(6 * self.scale), 10))
    
    @property
    def FONT_CONFUSION_MATRIX_SMALL(self):
        """混淆矩阵单元格文字 - 小 (5px @ 96dpi)"""
        return max(4, min(int(5 * self.scale), 9))
    
    # ========== 线宽系统 (Line Widths) ==========
    
    @property
    def LINE_DATA_THICK(self):
        """粗数据线 (2.5px @ 96dpi) - 用于平均线"""
        return max(2.0, min(2.5 * self.scale, 3.5))
    
    @property
    def LINE_DATA_MEDIUM(self):
        """中等数据线 (2.0px @ 96dpi) - 用于主要数据"""
        return max(1.5, min(2.0 * self.scale, 3.0))
    
    @property
    def LINE_DATA_STANDARD(self):
        """标准数据线 (1.8px @ 96dpi)"""
        return max(1.4, min(1.8 * self.scale, 2.6))
    
    @property
    def LINE_DATA_THIN(self):
        """细数据线 (1.5px @ 96dpi) - 用于对比线"""
        return max(1.2, min(1.5 * self.scale, 2.2))
    
    @property
    def LINE_DATA_LIGHT(self):
        """轻数据线 (0.8px @ 96dpi) - 用于背景线"""
        return max(0.6, min(0.8 * self.scale, 1.2))
    
    @property
    def LINE_GRID_STANDARD(self):
        """标准网格线 (0.8px @ 96dpi)"""
        return max(0.6, min(0.8 * self.scale, 1.2))
    
    @property
    def LINE_GRID_LIGHT(self):
        """轻网格线 (0.5px @ 96dpi)"""
        return max(0.4, min(0.5 * self.scale, 0.8))
    
    @property
    def LINE_GRID_MINOR(self):
        """次要网格线 (0.2px @ 96dpi)"""
        return max(0.15, min(0.2 * self.scale, 0.4))
    
    @property
    def LINE_SPINE(self):
        """边框线宽 (1.2px @ 96dpi)"""
        return max(1.0, min(1.2 * self.scale, 1.8))
    
    # ========== 标记大小系统 (Marker Sizes) ==========
    
    @property
    def MARKER_LARGE(self):
        """大标记 (50 @ 96dpi) - 用于散点图"""
        return max(40, min(int(50 * self.scale), 70))
    
    @property
    def MARKER_MEDIUM(self):
        """中标记 (30 @ 96dpi)"""
        return max(25, min(int(30 * self.scale), 45))
    
    @property
    def MARKER_SMALL(self):
        """小标记 (20 @ 96dpi)"""
        return max(15, min(int(20 * self.scale), 30))
    
    # ========== 图形尺寸系统 (Figure Sizes) ==========
    
    def get_figsize_and_dpi(self, base_width=10, base_height=7):
        """
        获取响应式图形尺寸和DPI
        
        Args:
            base_width: 基准宽度（英寸，96 DPI下）
            base_height: 基准高度（英寸，96 DPI下）
            
        Returns:
            tuple: ((width, height), dpi)
        """
        return ui_scaling_manager.get_responsive_figsize(base_width, base_height)
    
    # ========== 辅助方法 (Helper Methods) ==========
    
    def apply_to_matplotlib(self):
        """
        应用字体配置到matplotlib全局设置
        建议在创建图形前调用一次
        """
        import matplotlib.pyplot as plt
        plt.rcParams.update(self.font_config)
    
    def get_confusion_matrix_params(self, num_classes):
        """
        根据类别数量获取混淆矩阵的合适参数
        
        Args:
            num_classes: 类别数量
            
        Returns:
            dict: 包含title_fontsize, label_fontsize, tick_fontsize, cell_fontsize
        """
        if num_classes <= 5:
            return {
                'title_fontsize': self.FONT_TITLE_MEDIUM,
                'label_fontsize': self.FONT_LABEL_MEDIUM,
                'tick_fontsize': self.FONT_TICK_MEDIUM,
                'cell_fontsize': self.FONT_CONFUSION_MATRIX_LARGE,
            }
        elif num_classes <= 15:
            return {
                'title_fontsize': self.FONT_TITLE_SMALL,
                'label_fontsize': self.FONT_LABEL_SMALL,
                'tick_fontsize': self.FONT_TICK_SMALL,
                'cell_fontsize': self.FONT_CONFUSION_MATRIX_MEDIUM,
            }
        else:
            return {
                'title_fontsize': self.FONT_TITLE_SMALL,
                'label_fontsize': self.FONT_LABEL_SMALL,
                'tick_fontsize': self.FONT_TICK_TINY,
                'cell_fontsize': self.FONT_CONFUSION_MATRIX_SMALL,
            }
    
    def refresh(self):
        """刷新缩放参数（当DPI或分辨率改变时调用）"""
        self._update_scale()


# 创建全局单例实例
VDT = VisualizationDesignTokens()


# ========== 使用示例 (Usage Examples) ==========
"""
示例 1: 基本图形标题和标签
    from app.config.visualization_design_tokens import VDT
    
    ax.set_title("My Plot", fontsize=VDT.FONT_TITLE_LARGE, fontweight='bold')
    ax.set_xlabel("X Axis", fontsize=VDT.FONT_LABEL_LARGE)
    ax.set_ylabel("Y Axis", fontsize=VDT.FONT_LABEL_LARGE)

示例 2: 数据线
    ax.plot(x, y, linewidth=VDT.LINE_DATA_MEDIUM, label="Data")
    ax.plot(x, y_mean, linewidth=VDT.LINE_DATA_THICK, label="Mean")
    ax.plot(x, y_bg, linewidth=VDT.LINE_DATA_LIGHT, alpha=0.3, label="Background")

示例 3: 图例
    ax.legend(fontsize=VDT.FONT_LEGEND)

示例 4: 网格
    ax.grid(True, linewidth=VDT.LINE_GRID_LIGHT, alpha=0.3)

示例 5: 混淆矩阵
    params = VDT.get_confusion_matrix_params(num_classes)
    ax.set_title("Confusion Matrix", fontsize=params['title_fontsize'])
    ax.set_xlabel("Predicted", fontsize=params['label_fontsize'])
    ax.set_ylabel("True", fontsize=params['label_fontsize'])

示例 6: 创建响应式图形
    figsize, dpi = VDT.get_figsize_and_dpi(10, 7)
    fig = plt.figure(figsize=figsize, dpi=dpi)

示例 7: 应用全局matplotlib配置
    VDT.apply_to_matplotlib()  # 在创建图形前调用一次
"""

