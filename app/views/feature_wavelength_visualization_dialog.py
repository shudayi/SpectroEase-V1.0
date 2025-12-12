# -*- coding: utf-8 -*-
"""
特征波段可视化弹窗
在光谱图上用透明柱形标记选中的特征波段
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from app.views.responsive_dialog import ResponsiveDialog

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager


class FeatureWavelengthVisualizationDialog(ResponsiveDialog):
    """特征波段可视化弹窗"""
    
    def __init__(self, data_model, wavelengths_array, selected_features, parent=None):
        super().__init__(parent, base_width=1200, base_height=900)
        self.data_model = data_model
        self.wavelengths_array = np.array(wavelengths_array)
        self.selected_features = selected_features
        
        self.setWindowTitle("Selected Feature Wavelengths Visualization")
        
        self.init_ui()
        self.plot_spectra_with_highlights()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("📊 Selected Feature Wavelengths on Processed Spectra")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 说明文字
        info_label = QLabel(
            f"金色透明柱形标记表示选中的 {len(self.selected_features)} 个特征波段 | "
            f"Wavelength range: {self.wavelengths_array.min():.1f} - {self.wavelengths_array.max():.1f} nm"
        )
        info_label.setStyleSheet("font-size: 11px; color: #666; padding: 5px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Matplotlib 画布（响应式）
        from app.utils.responsive_matplotlib import create_responsive_figure
        self.figure = create_responsive_figure(base_width=14, base_height=9)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setStyleSheet("padding: 8px 20px; font-size: 12px;")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def plot_spectra_with_highlights(self):
        """绘制光谱并标记选中的特征波段"""
        self.ax.clear()
        
        try:
            # 提取选中的波长值
            selected_wavelengths = []
            for feature in self.selected_features:
                if isinstance(feature, str) and feature.startswith('X_'):
                    try:
                        wl = float(feature.split('_')[1])
                        selected_wavelengths.append(wl)
                    except:
                        pass
                elif isinstance(feature, (int, float)):
                    selected_wavelengths.append(float(feature))
            
            if len(selected_wavelengths) == 0:
                self.ax.text(0.5, 0.5, 'No valid wavelength features found', 
                           ha='center', va='center', fontsize=14, color='red')
                self.canvas.draw()
                return
            
            # 获取光谱数据
            if self.data_model.X_processed is not None:
                spectra = self.data_model.X_processed.values
                title_suffix = "(Processed Spectra)"
            elif self.data_model.X is not None:
                spectra = self.data_model.X.values
                title_suffix = "(Original Spectra)"
            else:
                self.ax.text(0.5, 0.5, 'No spectral data available', 
                           ha='center', va='center', fontsize=14, color='red')
                self.canvas.draw()
                return
            
            # 判断是否为分类任务
            y_labels = None
            is_categorical = False
            if hasattr(self.data_model, 'y') and self.data_model.y is not None:
                y_labels = self.data_model.y.values if hasattr(self.data_model.y, 'values') else self.data_model.y
                unique_labels = np.unique(y_labels)
                if len(unique_labels) < len(y_labels) * 0.3:
                    is_categorical = True
            
            # 选择显示的样本
            selected_indices = []
            if is_categorical and y_labels is not None:
                unique_labels = np.unique(y_labels)
                samples_per_category = 3 if len(unique_labels) < 5 else 1
                for label in unique_labels:
                    label_indices = np.where(y_labels == label)[0]
                    samples_to_select = min(len(label_indices), samples_per_category)
                    selected_indices.extend(label_indices[:samples_to_select].tolist())
            else:
                selected_indices = list(range(min(5, spectra.shape[0])))
            
            # 计算波长间距，用于确定标记宽度
            wl_spacing = np.median(np.diff(np.sort(self.wavelengths_array))) if len(self.wavelengths_array) > 1 else 1.0
            marker_width = wl_spacing * 0.8
            
            # 先绘制透明柱形标记（置于底层）
            for wl in selected_wavelengths:
                self.ax.axvspan(
                    wl - marker_width/2, 
                    wl + marker_width/2,
                    alpha=0.25,
                    color='gold',
                    zorder=0
                )
            
            # 绘制光谱线
            cmap = plt.get_cmap("tab10")
            for i, idx in enumerate(selected_indices):
                if is_categorical and y_labels is not None:
                    label = f"{y_labels[idx]}"
                else:
                    label = f"Sample {i+1}"
                
                self.ax.plot(
                    self.wavelengths_array, 
                    spectra[idx], 
                    color=cmap(i % 10), 
                    linewidth=1.2, 
                    alpha=0.8, 
                    label=label,
                    zorder=2
                )
            
            # 绘制均值线
            mean_spectra = np.mean(spectra, axis=0)
            std_spectra = np.std(spectra, axis=0)
            self.ax.plot(
                self.wavelengths_array, 
                mean_spectra, 
                color='black', 
                linewidth=2.5, 
                alpha=0.7, 
                label='Mean',
                zorder=3
            )
            self.ax.fill_between(
                self.wavelengths_array, 
                mean_spectra - std_spectra, 
                mean_spectra + std_spectra, 
                color='gray', 
                alpha=0.15,
                zorder=1
            )
            
            # 图表设置
            # Get responsive font sizes
            font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
            
            self.ax.set_title(
                f"Selected Feature Wavelengths {title_suffix}\n{len(selected_wavelengths)} features highlighted in gold", 
                fontsize=font_sizes['axes.titlesize'], 
                fontweight='bold',
                pad=15
            )
            self.ax.set_xlabel("Wavelength (nm)", fontsize=font_sizes['axes.labelsize'])
            self.ax.set_ylabel("Intensity", fontsize=font_sizes['axes.labelsize'])
            self.ax.grid(True, linestyle='--', alpha=0.3)
            
            # 添加图例
            handles, labels = self.ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='gold', alpha=0.25, label=f'Selected Features ({len(selected_wavelengths)})'))
            self.ax.legend(handles=handles, loc='best', fontsize=font_sizes['legend.fontsize'], framealpha=0.9)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            print(f"✅ Feature wavelength visualization displayed, marked {len(selected_wavelengths)} selected bands")
            
        except Exception as e:
            print(f"❌ Error plotting feature wavelength visualization: {e}")
            import traceback
            traceback.print_exc()
            
            self.ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', fontsize=12, color='red')
            self.canvas.draw()

