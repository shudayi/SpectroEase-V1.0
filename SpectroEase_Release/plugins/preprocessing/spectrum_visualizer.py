import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

class SpectrumVisualizer:
    """Spectrum data visualization class"""
    
    @staticmethod
    def plot_spectra(wavelengths, spectra, 
                    title: str = "Spectrum Plot",
                    xlabel: str = "Wavelength (nm)",
                    ylabel: str = "Absorbance",
                    legend=None) -> plt.Figure:
        """
        Plot spectrum chart
        
        Args:
            wavelengths: Wavelength array
            spectra: Spectrum data array  
            title: Chart title
            xlabel: x-axis labels
            ylabel: y-axis labels
            legend: Legend labels list
            
        Returns:
            plt.Figure: matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(spectra.shape) == 1:
            spectra = spectra.reshape(1, -1)
            
        for i in range(spectra.shape[0]):
            label = legend[i] if legend and i < len(legend) else f"Spectrum {i+1}"
            ax.plot(wavelengths, spectra[i], label=label)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    @staticmethod
    def plot_comparison(wavelengths, spectrum1, spectrum2,
                       title: str = "Spectrum Comparison",
                       label1: str = "Spectrum 1",
                       label2: str = "Spectrum 2") -> plt.Figure:
        """
        Plot comparison chart of two spectrum groups
        
        Args:
            wavelengths: Wavelength array
            spectrum1: First spectrum data
            spectrum2: Second spectrum data
            title: Chart title
            label1: First spectrum label
            label2: Second spectrum label
            
        Returns:
            plt.Figure: matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        for i in range(spectrum1.shape[0]):
            ax1.plot(wavelengths, spectrum1[i], label=f"{label1} {i+1}")
        for i in range(spectrum2.shape[0]):
            ax1.plot(wavelengths, spectrum2[i], label=f"{label2} {i+1}")
            
        ax1.set_title(title)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Absorbance")
        ax1.grid(True)
        ax1.legend()
        
        diff = np.mean(spectrum1, axis=0) - np.mean(spectrum2, axis=0)
        ax2.plot(wavelengths, diff, label="Difference")
        ax2.set_title("Spectrum Difference")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Difference")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    @staticmethod
    def plot_spectra_heatmap(spectra: np.ndarray, wavelengths: np.ndarray,
                           sample_labels: Optional[List[str]] = None,
                           title: str = "Spectrum Heatmap") -> plt.Figure:
        """
        Plot spectrum heatmap
        
        Args:
            spectra: Spectrum data array
            wavelengths: Wavelength array
            sample_labels: Samples labels list
            title: Chart title
            
        Returns:
            plt.Figure: matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(spectra, aspect='auto',
                      extent=[wavelengths[0], wavelengths[-1],
                             0, spectra.shape[0]])
        
        plt.colorbar(im, ax=ax, label="Absorbance")
        
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Samples")
        
        if sample_labels:
            ax.set_yticks(range(len(sample_labels)))
            ax.set_yticklabels(sample_labels)
            
        return fig 