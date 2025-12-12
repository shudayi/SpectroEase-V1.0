# -*- coding: utf-8 -*-
"""
Dialog for PCA Exploratory Analysis Visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QScrollArea, QWidget, QDialogButtonBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import UI scaling manager for responsive fonts
from app.utils.ui_scaling import ui_scaling_manager

class PCAExplorerDialog(QDialog):
    """
    A comprehensive dialog to display PCA exploratory analysis results, 
    including scree, cumulative variance, scores, and loadings plots.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCA Exploratory Analysis")
        
        # 🔧 V1.2.1: 动态适配不同分辨率
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.size()
            screen_dpi = screen.logicalDotsPerInch()
            
            # 计算窗口大小（占屏幕的75-80%）
            window_width = int(screen_size.width() * 0.75)
            window_height = int(screen_size.height() * 0.80)
            
            # 最小尺寸限制（防止太小）
            window_width = max(900, min(window_width, 1600))
            window_height = max(700, min(window_height, 1200))
            
            self.setMinimumSize(window_width, window_height)
            
            # 计算图表DPI（基于屏幕DPI）
            base_dpi = 96
            self.figure_dpi = int(screen_dpi * 100 / base_dpi)
            self.figure_dpi = max(80, min(self.figure_dpi, 150))
            
            print(f"📺 PCA Dialog adaptation: window={window_width}x{window_height}, DPI={self.figure_dpi}")
        else:
            self.setMinimumSize(1000, 900)
            self.figure_dpi = 100

        self.pca = None
        self.loadings = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        self.X_scores = None
        self.wavelengths = None
        self.y = None
        self.is_modeling_mode = False

        # --- UI Setup ---
        # Main layout with Scroll Area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        
        container_widget = QWidget()
        self.main_layout = QVBoxLayout(container_widget)
        
        scroll_area.setWidget(container_widget)
        
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(scroll_area)

        # OK Button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        dialog_layout.addWidget(button_box)

        self.setLayout(dialog_layout)

    def set_data(self, pca_instance, loadings, explained_variance_ratio, cumulative_variance, X_scores, wavelengths, y=None, is_modeling_mode=False):
        """Set the data to be plotted and render the plots."""
        self.pca = pca_instance
        self.loadings = loadings
        self.explained_variance_ratio = explained_variance_ratio
        self.cumulative_variance = cumulative_variance
        self.X_scores = X_scores
        self.wavelengths = wavelengths
        self.y = y
        self.is_modeling_mode = is_modeling_mode
        self.plot()

    def plot(self):
        """Renders all PCA plots."""
        # Clear previous plots if any
        for i in reversed(range(self.main_layout.count())): 
            widgetToRemove = self.main_layout.itemAt(i).widget()
            self.main_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

        self._plot_scree()
        self._plot_scores()
        self._plot_loadings()

    def _plot_scree(self):
        """Plots the scree and cumulative variance."""
        if self.explained_variance_ratio is None: return

        # 🔧 V1.2.1修复: 动态计算figsize（基于窗口大小）
        window_width = self.width()
        window_height = self.height()
        
        # 每个图占窗口宽度的90%，高度的25%
        fig_width = (window_width * 0.9) / self.figure_dpi
        fig_height = (window_height * 0.25) / self.figure_dpi
        
        scree_fig = Figure(figsize=(fig_width, fig_height), dpi=self.figure_dpi)
        ax1 = scree_fig.add_subplot(111)
        
        components = np.arange(1, len(self.explained_variance_ratio) + 1)
        
        # Get responsive font sizes
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
        
        ax1.bar(components, self.explained_variance_ratio * 100, alpha=0.7, color='g', label='Individual Explained Variance')
        ax1.set_xlabel('Principal Component', fontsize=font_sizes['axes.labelsize'])
        ax1.set_ylabel('Explained Variance (%)', color='g', fontsize=font_sizes['axes.labelsize'])
        ax1.tick_params(axis='y', labelcolor='g', labelsize=font_sizes['ytick.labelsize'])
        ax1.tick_params(axis='x', labelsize=font_sizes['xtick.labelsize'])
        ax1.set_xticks(components)
        ax1.set_xlim(0.5, len(components) + 0.5)

        ax2 = ax1.twinx()
        ax2.plot(components, self.cumulative_variance * 100, 'b-', marker='o', label='Cumulative Explained Variance')
        ax2.set_ylabel('Cumulative Explained Variance (%)', color='b', fontsize=font_sizes['axes.labelsize'])
        ax2.tick_params(axis='y', labelcolor='b', labelsize=font_sizes['ytick.labelsize'])
        ax2.set_ylim(0, 105)
        
        scree_fig.suptitle('Scree Plot & Cumulative Variance', fontsize=font_sizes['axes.titlesize'])
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=font_sizes['legend.fontsize'])
        
        scree_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        canvas = FigureCanvas(scree_fig)
        groupbox = QGroupBox("Variance Explained")
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        groupbox.setLayout(layout)
        self.main_layout.addWidget(groupbox)

    def _plot_scores(self):
        """Plots the scores of PC1 vs PC2."""
        if self.X_scores is None or self.X_scores.shape[1] < 2: return

        # 🔧 V1.2.1修复: 动态计算figsize
        window_width = self.width()
        window_height = self.height()
        
        fig_width = (window_width * 0.9) / self.figure_dpi
        fig_height = (window_height * 0.35) / self.figure_dpi
        
        scores_fig = Figure(figsize=(fig_width, fig_height), dpi=self.figure_dpi)
        ax = scores_fig.add_subplot(111)
        
        title = "Scores Plot (PC2 vs. PC1)"
        if self.is_modeling_mode:
            title += " - Training Set"

        scatter = ax.scatter(self.X_scores[:, 0], self.X_scores[:, 1], c=self.y, cmap='viridis', alpha=0.7, edgecolors='k')
        
        # Get responsive font sizes
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
        
        ax.set_xlabel(f'Principal Component 1 ({self.explained_variance_ratio[0]:.1%} Var)', fontsize=font_sizes['axes.labelsize'])
        ax.set_ylabel(f'Principal Component 2 ({self.explained_variance_ratio[1]:.1%} Var)', fontsize=font_sizes['axes.labelsize'])
        ax.set_title(title, fontsize=font_sizes['axes.titlesize'])
        ax.tick_params(labelsize=font_sizes['xtick.labelsize'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        if self.y is not None and len(np.unique(self.y)) > 1:
            try:
                legend = ax.legend(*scatter.legend_elements(), title="Classes", fontsize=font_sizes['legend.fontsize'])
                ax.add_artist(legend)
            except ValueError:
                pass
        
        scores_fig.tight_layout()
        canvas = FigureCanvas(scores_fig)
        groupbox = QGroupBox("Scores Plot")
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        groupbox.setLayout(layout)
        self.main_layout.addWidget(groupbox)

    def _plot_loadings(self):
        """Plots the PCA loadings for the first few components."""
        if self.loadings is None or self.wavelengths is None: return

        # 🔧 V1.2.1修复: 动态计算figsize
        window_width = self.width()
        window_height = self.height()
        
        fig_width = (window_width * 0.9) / self.figure_dpi
        fig_height = (window_height * 0.30) / self.figure_dpi
        
        loadings_fig = Figure(figsize=(fig_width, fig_height), dpi=self.figure_dpi)
        ax = loadings_fig.add_subplot(111)
        
        num_pcs_to_plot = min(self.loadings.shape[0], 3)
        for i in range(num_pcs_to_plot):
            ax.plot(self.wavelengths, self.loadings[i, :], label=f'PC-{i+1}', linewidth=1.5)
        
        # Get responsive font sizes
        font_sizes = ui_scaling_manager.get_matplotlib_font_sizes()
            
        ax.set_title('PCA Loadings', fontsize=font_sizes['axes.titlesize'])
        ax.set_xlabel('Wavelength / Wavenumber', fontsize=font_sizes['axes.labelsize'])
        ax.set_ylabel('Loading Value', fontsize=font_sizes['axes.labelsize'])
        ax.tick_params(labelsize=font_sizes['xtick.labelsize'])
        ax.legend(fontsize=font_sizes['legend.fontsize'])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        loadings_fig.tight_layout()
        canvas = FigureCanvas(loadings_fig)
        
        groupbox = QGroupBox("Loadings Plot")
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        groupbox.setLayout(layout)
        self.main_layout.addWidget(groupbox)