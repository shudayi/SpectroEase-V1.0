# -*- coding: utf-8 -*-
"""
UI Scaling and DPI Awareness Utility
Addresses Editor Comment 2: UI scaling issues on different resolutions
"""

import sys
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import platform

class UIScalingManager:
    """Manages UI scaling and DPI awareness across different screen resolutions"""
    
    def __init__(self):
        self.screen_info = self._get_screen_info()
        self.scale_factor = self._calculate_scale_factor()
        self.font_scale = self._calculate_font_scale()
        
    def _get_screen_info(self):
        """Get current screen information"""
        app = QApplication.instance()
        if app is None:
            # Fallback values when no QApplication exists
            from PyQt5.QtCore import QRect
            return {
                "width": 1920, 
                "height": 1080, 
                "dpi": 96,
                "available_geometry": QRect(0, 0, 1920, 1080)
            }
            
        desktop = app.desktop()
        screen_rect = desktop.screenGeometry()
        
        # Get DPI information
        screen = app.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen else 96
        
        return {
            "width": screen_rect.width(),
            "height": screen_rect.height(), 
            "dpi": dpi,
            "available_geometry": desktop.availableGeometry()
        }
    
    def _calculate_scale_factor(self):
        """Calculate UI scale factor based on screen resolution"""
        width = self.screen_info["width"]
        height = self.screen_info["height"]
        
        # Base resolution: 1920x1080
        base_width, base_height = 1920, 1080
        
        # Calculate scale factors
        width_scale = width / base_width
        height_scale = height / base_height
        
        # Use the smaller scale to ensure everything fits
        scale_factor = min(width_scale, height_scale)
        
        # Clamp scale factor to reasonable range
        return max(0.7, min(scale_factor, 2.0))
    
    def _calculate_font_scale(self):
        """Calculate font scale factor based on DPI"""
        dpi = self.screen_info["dpi"]
        
        # Standard DPI is 96
        base_dpi = 96
        font_scale = dpi / base_dpi
        
        # Clamp font scale to reasonable range
        return max(0.8, min(font_scale, 1.5))
    
    def get_scaled_window_size(self, base_width=2000, base_height=1200):
        """Get scaled window size based on screen resolution"""
        available = self.screen_info["available_geometry"]
        
        # For 1080p and smaller, use 80% of screen size as specified in response
        if self.screen_info["height"] <= 1080:
            target_width = int(available.width() * 0.8)
            target_height = int(available.height() * 0.8)
        else:
            # For larger screens, use base size with scaling
            target_width = int(base_width * self.scale_factor)
            target_height = int(base_height * self.scale_factor)
        
        # Ensure minimum size
        target_width = max(target_width, 1200)
        target_height = max(target_height, 800)
        
        # Ensure it fits on screen
        target_width = min(target_width, available.width() - 100)
        target_height = min(target_height, available.height() - 100)
        
        return target_width, target_height
    
    def get_scaled_font(self, base_font_name="Arial", base_size=9):
        """Get scaled font based on DPI"""
        scaled_size = int(base_size * self.font_scale)
        scaled_size = max(8, min(scaled_size, 14))  # Reasonable range
        
        return QFont(base_font_name, scaled_size)
    
    def get_scaled_value(self, base_value):
        """Scale any value based on current scale factor"""
        return int(base_value * self.scale_factor)
    
    def apply_dpi_awareness(self):
        """Apply DPI awareness settings to the application"""
        app = QApplication.instance()
        if app is None:
            return
            
        # Enable high DPI scaling
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Set DPI awareness policy for Windows
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            except:
                pass
    
    def get_scaling_info(self):
        """Get detailed scaling information for debugging"""
        return {
            "screen_resolution": f"{self.screen_info['width']}x{self.screen_info['height']}",
            "screen_dpi": self.screen_info["dpi"],
            "ui_scale_factor": self.scale_factor,
            "font_scale_factor": self.font_scale,
            "recommended_window_size": self.get_scaled_window_size()
        }
    
    def get_responsive_dialog_size(self, base_width, base_height):
        """
        为对话框计算响应式尺寸
        
        Args:
            base_width: 基准宽度（1920x1080下的理想宽度）
            base_height: 基准高度
            
        Returns:
            (width, height): 响应式计算后的尺寸
        """
        try:
            screen_width = self.screen_info["width"]
            screen_height = self.screen_info["height"]
            
            # 根据分辨率计算占屏比例
            if screen_width <= 1366:
                # 低分辨率：对话框占屏幕90%
                width_ratio = 0.90
                height_ratio = 0.85
            elif screen_width <= 1920:
                # FHD：对话框占屏幕60-70%
                width_ratio = 0.65
                height_ratio = 0.70
            elif screen_width <= 2560:
                # 2K：对话框占屏幕50-60%
                width_ratio = 0.55
                height_ratio = 0.60
            else:
                # 4K及以上：对话框占屏幕40-50%
                width_ratio = 0.45
                height_ratio = 0.55
            
            # 计算实际尺寸
            calc_width = int(screen_width * width_ratio)
            calc_height = int(screen_height * height_ratio)
            
            # 限制在合理范围内（基于基准尺寸）
            min_width = int(base_width * 0.6)
            max_width = int(base_width * 1.8)
            min_height = int(base_height * 0.6)
            max_height = int(base_height * 1.8)
            
            calc_width = max(min_width, min(calc_width, max_width))
            calc_height = max(min_height, min(calc_height, max_height))
            
            return calc_width, calc_height
        except Exception as e:
            # Fallback: 返回基准尺寸
            print(f"⚠️ Dialog size calculation failed: {e}, using base size")
            return base_width, base_height
    
    def get_responsive_figure_dpi(self):
        """为matplotlib图表计算响应式DPI"""
        screen_width = self.screen_info["width"]
        screen_dpi = self.screen_info["dpi"]
        
        # 基于屏幕宽度和DPI综合判断
        if screen_width >= 3840:  # 4K
            base_dpi = max(screen_dpi, 120)
            return int(base_dpi * 1.2)  # 144-240
        elif screen_width >= 2560:  # 2K
            base_dpi = max(screen_dpi, 100)
            return int(base_dpi * 1.1)  # 110-121
        elif screen_width >= 1920:  # FHD
            return max(screen_dpi, 96)  # 96-110
        else:  # 低分辨率
            return max(int(screen_dpi * 0.9), 80)  # 80-90
    
    def get_responsive_figsize(self, base_width, base_height):
        """
        为matplotlib图表计算响应式figsize
        
        Args:
            base_width: 基准宽度（inch）
            base_height: 基准高度（inch）
            
        Returns:
            ((width, height), dpi): figsize元组和DPI
        """
        screen_width = self.screen_info["width"]
        
        # 计算尺寸缩放因子
        if screen_width <= 1366:
            size_scale = 0.75  # 低分辨率缩小
        elif screen_width <= 1920:
            size_scale = 1.0   # FHD保持
        elif screen_width <= 2560:
            size_scale = 1.15  # 2K略微放大
        else:
            size_scale = 1.35  # 4K明显放大
        
        figsize = (base_width * size_scale, base_height * size_scale)
        dpi = self.get_responsive_figure_dpi()
        
        return figsize, dpi
    
    def get_matplotlib_font_sizes(self):
        """获取响应式matplotlib字体大小"""
        font_scale = self.font_scale
        
        # 基于DPI缩放字体
        return {
            'font.size': max(8, min(int(12 * font_scale), 16)),
            'axes.labelsize': max(10, min(int(14 * font_scale), 18)),
            'axes.titlesize': max(12, min(int(16 * font_scale), 20)),
            'xtick.labelsize': max(8, min(int(12 * font_scale), 16)),
            'ytick.labelsize': max(8, min(int(12 * font_scale), 16)),
            'legend.fontsize': max(8, min(int(11 * font_scale), 15)),
        }
    
    def get_dynamic_font(self, base_name="Arial", base_size=9):
        """
        获取动态字体（考虑DPI）
        
        Args:
            base_name: 字体名称
            base_size: 基准大小（96 DPI下的大小）
            
        Returns:
            QFont: 响应式字体对象
        """
        scaled_size = int(base_size * self.font_scale)
        # 限制在合理范围
        scaled_size = max(7, min(scaled_size, 16))
        
        return QFont(base_name, scaled_size)

# Global UI scaling manager instance
ui_scaling_manager = UIScalingManager()

def apply_responsive_sizing(window, base_width=2000, base_height=1200):
    """Apply responsive sizing to a window"""
    width, height = ui_scaling_manager.get_scaled_window_size(base_width, base_height)
    window.resize(width, height)
    
    # Center the window on screen
    screen_geometry = ui_scaling_manager.screen_info["available_geometry"]
    x = (screen_geometry.width() - width) // 2
    y = (screen_geometry.height() - height) // 2
    window.move(x, y)
    
    print(f"🖥️ Applied responsive sizing: {width}x{height} (scale: {ui_scaling_manager.scale_factor:.2f})")

def get_responsive_font(base_name="Arial", base_size=9):
    """Get responsive font"""
    return ui_scaling_manager.get_scaled_font(base_name, base_size)
