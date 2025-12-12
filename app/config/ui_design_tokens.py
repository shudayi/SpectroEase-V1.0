"""
UI Design Tokens for SpectroEase
统一的设计系统配置 - 所有UI尺寸和间距的单一真实来源
"""

class UIDesignTokens:
    """
    UI设计令牌配置类
    
    这个类定义了整个应用程序的视觉设计系统，包括：
    - 控件尺寸（高度、宽度）
    - 间距系统（垂直和水平间距）
    - 边距系统（容器内外边距）
    - 字体系统
    - 图标尺寸
    - 响应式缩放
    
    使用方法:
        from app.config.ui_design_tokens import UIDesignTokens as DT
        
        button = QPushButton("Click Me")
        button.setFixedHeight(DT.HEIGHT_BUTTON_PRIMARY)
    """
    
    # ========== 尺寸系统 (Sizing System) ==========
    
    ## 控件高度 (Widget Heights)
    HEIGHT_BUTTON_PRIMARY = 28        # 主要操作按钮（Apply, Start Analysis等）
    HEIGHT_BUTTON_SECONDARY = 24      # 次要按钮（Preview, Settings等）
    HEIGHT_BUTTON_COMPACT = 20        # 紧凑按钮（内联操作）
    HEIGHT_INPUT_CONTROL = 24         # 输入控件（ComboBox, SpinBox等）
    HEIGHT_CHECKBOX = 24              # 复选框
    HEIGHT_LABEL_INLINE = 24          # 行内标签
    HEIGHT_TOOLBAR = 40               # 工具栏高度
    HEIGHT_STATUSBAR = 28             # 状态栏高度
    
    ## 组框高度 (GroupBox Heights)
    HEIGHT_GROUP_COMPACT = 60         # 紧凑型组框（单行控件）
    HEIGHT_GROUP_SMALL = 100          # 小型组框（2-3行控件）
    HEIGHT_GROUP_MEDIUM = 160         # 中型组框（多行控件）
    HEIGHT_GROUP_LARGE = 280          # 大型组框（复杂控件组）
    HEIGHT_GROUP_AUTO = None          # 自适应高度（使用最小高度约束）
    HEIGHT_PREVIEW_GROUP = 80         # 预览组固定高度
    HEIGHT_RESULT_TABLE_MIN = 180     # 结果表格最小高度
    HEIGHT_RESULT_TABLE_MAX = 300     # 结果表格最大高度
    HEIGHT_VISUALIZATION_MIN = 400    # 可视化组件最小高度
    
    ## 宽度 (Widths)
    WIDTH_LABEL_STANDARD = 110        # 标准标签宽度（增加10px避免文字截断）
    WIDTH_LABEL_WIDE = 150            # 宽标签（增加10px）
    WIDTH_CONTROL_STANDARD = 180      # 标准控件宽度（从150增加到180，避免文字截断）
    WIDTH_CONTROL_WIDE = 220          # 宽控件（从200增加到220）
    WIDTH_CONTROL_COMPACT = 90        # 紧凑控件（从80增加到90）
    WIDTH_BUTTON_COMPACT = 90         # 紧凑按钮（从80增加到90）
    WIDTH_LEFT_PANEL_MIN = 600        # 左侧面板最小宽度（从580增加到600）
    WIDTH_PROGRESS_BAR = 300          # 进度条宽度
    
    ## 图标尺寸 (Icon Sizes)
    SIZE_TOOLBAR_ICON = 24            # 工具栏图标尺寸
    SIZE_STATUSBAR_ICON = 16          # 状态栏图标尺寸
    SIZE_BUTTON_ICON = 16             # 按钮内图标尺寸
    
    # ========== 间距系统 (Spacing System) ==========
    
    ## 垂直/水平间距 (Vertical/Horizontal Spacing)
    SPACING_NONE = 0                  # 无间距（特殊情况）
    SPACING_TIGHT = 4                 # 紧凑间距（同组相关元素，表单行）
    SPACING_STANDARD = 8              # 标准间距（常规组件间，组框内）
    SPACING_RELAXED = 12              # 宽松间距（主要组框间）
    SPACING_LOOSE = 16                # 松散间距（大模块间，很少使用）
    
    # ========== 边距系统 (Margins/Padding) ==========
    
    ## 常用边距组合 (Common Margin Combinations)
    MARGIN_NONE = (0, 0, 0, 0)        # 无边距
    MARGIN_TIGHT = (4, 4, 4, 4)       # 紧凑边距（内部容器）
    MARGIN_STANDARD = (8, 8, 8, 8)    # 标准边距（组框内）
    MARGIN_RELAXED = (10, 10, 10, 10) # 宽松边距（主容器）
    
    ## 特殊用途边距 (Special Purpose Margins)
    MARGIN_FORM_FIELD = (0, 4, 0, 0)  # 表单字段上边距
    MARGIN_CONTAINER = (10, 10, 10, 10) # 主容器边距
    
    # ========== 字体系统 (Typography System) ==========
    
    FONT_FAMILY = "Arial"             # 默认字体系列
    FONT_SIZE_SMALL = 8               # 辅助文字
    FONT_SIZE_STANDARD = 9            # 标准文字
    FONT_SIZE_LARGE = 10              # 标题文字
    FONT_SIZE_XLARGE = 11             # 主标题
    
    # Matplotlib 字体基准（将被 ui_scaling_manager 动态缩放）
    MPL_FONT_BASE = 12                # 基础字体
    MPL_FONT_LABEL = 14               # 轴标签
    MPL_FONT_TITLE = 16               # 标题
    MPL_FONT_LEGEND = 11              # 图例
    MPL_FONT_TICK = 12                # 刻度标签
    
    # ========== 分割器系统 (Splitter System) ==========
    
    SPLITTER_LEFT_RATIO = 0.25        # 左侧面板占比（25%）
    SPLITTER_RIGHT_RATIO = 0.75       # 右侧面板占比（75%）
    SPLITTER_HANDLE_WIDTH = 4         # 分割器手柄宽度
    SPLITTER_VISUALIZATION_RATIO = 0.8  # 可视化区域占右侧面板的比例
    SPLITTER_TABLE_RATIO = 0.2          # 结果表格占右侧面板的比例
    
    # ========== 响应式系统 (Responsive System) ==========
    
    @staticmethod
    def get_responsive_multiplier(screen_width):
        """
        根据屏幕宽度返回缩放系数
        
        Args:
            screen_width (int): 屏幕宽度（像素）
            
        Returns:
            float: 缩放系数
            
        示例:
            1366x768  -> 0.9x
            1920x1080 -> 1.0x
            2560x1440 -> 1.1x
            3840x2160 -> 1.2x
        """
        if screen_width < 1366:
            return 0.85
        elif screen_width < 1920:
            return 0.9
        elif screen_width < 2560:
            return 1.0
        elif screen_width < 3840:
            return 1.1
        else:
            return 1.2
    
    @staticmethod
    def get_responsive_height(base_height, screen_height):
        """
        根据屏幕高度返回响应式高度
        
        Args:
            base_height (int): 基础高度（1080p标准）
            screen_height (int): 屏幕高度（像素）
            
        Returns:
            int: 响应式高度
        """
        if screen_height >= 1440:  # 2K及以上
            return int(base_height * 1.15)
        elif screen_height >= 1080:  # 1080p
            return base_height
        elif screen_height >= 900:  # 900p
            return int(base_height * 0.92)
        else:  # 720p或更小
            return int(base_height * 0.85)
    
    @classmethod
    def apply_responsive_sizing(cls, base_value, screen_width):
        """
        应用响应式缩放到给定的基础值
        
        Args:
            base_value (int/float): 基础尺寸值
            screen_width (int): 屏幕宽度
            
        Returns:
            int: 缩放后的尺寸值
            
        示例:
            DT.apply_responsive_sizing(100, 1920) -> 100
            DT.apply_responsive_sizing(100, 2560) -> 110
        """
        multiplier = cls.get_responsive_multiplier(screen_width)
        return int(base_value * multiplier)
    
    # ========== 便捷方法 (Convenience Methods) ==========
    
    @classmethod
    def get_spacing_for_level(cls, level):
        """
        根据层级获取推荐的间距值
        
        Args:
            level (int): 层级 (1-5)
                1: 主要模块间
                2: 组框间
                3: 组框内组件间
                4: 表单行间
                5: 相关元素间
                
        Returns:
            int: 间距像素值
        """
        spacing_map = {
            1: cls.SPACING_LOOSE,      # 16px - 主要模块间
            2: cls.SPACING_RELAXED,    # 12px - 组框间
            3: cls.SPACING_STANDARD,   # 8px - 组框内
            4: cls.SPACING_TIGHT,      # 4px - 表单行
            5: cls.SPACING_TIGHT,      # 4px - 相关元素
        }
        return spacing_map.get(level, cls.SPACING_STANDARD)
    
    @classmethod
    def get_margin_tuple(cls, margin_type='standard'):
        """
        获取标准的边距元组
        
        Args:
            margin_type (str): 边距类型
                'none': (0,0,0,0)
                'tight': (4,4,4,4)
                'standard': (8,8,8,8)
                'relaxed': (10,10,10,10)
                'form': (0,4,0,0)
                
        Returns:
            tuple: (left, top, right, bottom)
        """
        margin_map = {
            'none': cls.MARGIN_NONE,
            'tight': cls.MARGIN_TIGHT,
            'standard': cls.MARGIN_STANDARD,
            'relaxed': cls.MARGIN_RELAXED,
            'form': cls.MARGIN_FORM_FIELD,
            'container': cls.MARGIN_CONTAINER,
        }
        return margin_map.get(margin_type, cls.MARGIN_STANDARD)


# 创建全局单例实例（可选，方便使用）
DT = UIDesignTokens()


# ========== 使用示例 (Usage Examples) ==========
"""
示例 1: 创建标准按钮
    from app.config.ui_design_tokens import UIDesignTokens as DT
    
    button = QPushButton("Apply")
    button.setFixedHeight(DT.HEIGHT_BUTTON_PRIMARY)

示例 2: 设置布局间距
    layout = QVBoxLayout()
    layout.setSpacing(DT.SPACING_STANDARD)
    layout.setContentsMargins(*DT.MARGIN_STANDARD)

示例 3: 创建标准ComboBox
    combo = QComboBox()
    combo.setFixedHeight(DT.HEIGHT_INPUT_CONTROL)
    combo.setFixedWidth(DT.WIDTH_CONTROL_STANDARD)

示例 4: 响应式尺寸
    screen_width = QApplication.desktop().screenGeometry().width()
    button_height = DT.apply_responsive_sizing(DT.HEIGHT_BUTTON_PRIMARY, screen_width)
    button.setFixedHeight(button_height)

示例 5: 使用层级间距
    main_layout.setSpacing(DT.get_spacing_for_level(1))  # 主模块间
    group_layout.setSpacing(DT.get_spacing_for_level(3))  # 组框内
"""

