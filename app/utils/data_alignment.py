# -*- coding: utf-8 -*-
"""
数据维度对齐工具
用于解决光谱数据和波长数组维度不匹配的问题
"""

import numpy as np
import pandas as pd

class DataDimensionAligner:
    """数据维度对齐工具类"""
    
    @staticmethod
    def align_spectral_data(wavelengths, *spectra_arrays, **kwargs):
        """
        对齐光谱数据的维度
        
        Args:
            wavelengths: 波长数组
            *spectra_arrays: 可变数量的光谱数据数组
            **kwargs: 
                - strict_mode: 严格模式，如果维度差异太大则抛出异常
                - min_overlap_ratio: 最小重叠比例，默认0.8
                
        Returns:
            tuple: (aligned_wavelengths, *aligned_spectra_arrays)
            
        Raises:
            ValueError: 当数据维度不兼容时
        """
        strict_mode = kwargs.get('strict_mode', False)
        min_overlap_ratio = kwargs.get('min_overlap_ratio', 0.8)
        
        try:
            # 转换为numpy数组
            wavelengths = np.array(wavelengths, dtype=float)
            converted_spectra = []
            
            for i, spectra in enumerate(spectra_arrays):
                if spectra is None:
                    converted_spectra.append(None)
                    continue
                    
                spectra = np.array(spectra, dtype=float)
                
                # 确保是2D数组
                if spectra.ndim == 1:
                    spectra = spectra.reshape(1, -1)
                elif spectra.ndim > 2:
                    raise ValueError(f"Spectra array {i} has too many dimensions: {spectra.ndim}")
                
                converted_spectra.append(spectra)
            
            # 过滤掉None值
            valid_spectra = [s for s in converted_spectra if s is not None]
            
            if not valid_spectra:
                return wavelengths, *converted_spectra
            
            # 找到所有数组的最小长度
            all_lengths = [len(wavelengths)] + [s.shape[1] for s in valid_spectra]
            min_length = min(all_lengths)
            max_length = max(all_lengths)
            
            # 检查重叠比例
            overlap_ratio = min_length / max_length
            if strict_mode and overlap_ratio < min_overlap_ratio:
                raise ValueError(f"Dimension mismatch too large. Overlap ratio: {overlap_ratio:.2f}, minimum required: {min_overlap_ratio}")
            
            # 对齐所有数组
            aligned_wavelengths = wavelengths[:min_length]
            aligned_spectra = []
            
            for spectra in converted_spectra:
                if spectra is None:
                    aligned_spectra.append(None)
                else:
                    aligned_spectra.append(spectra[:, :min_length])
            
            return aligned_wavelengths, *aligned_spectra
            
        except Exception as e:
            raise ValueError(f"Failed to align spectral data dimensions: {str(e)}")
    
    @staticmethod
    def validate_spectral_data(wavelengths, spectra, name="spectra"):
        """
        验证光谱数据的有效性
        
        Args:
            wavelengths: 波长数组
            spectra: 光谱数据
            name: 数据名称，用于错误信息
            
        Returns:
            dict: 验证结果信息
        """
        info = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            if wavelengths is None or spectra is None:
                info['valid'] = False
                info['errors'].append(f"{name}: Data is None")
                return info
            
            wavelengths = np.array(wavelengths)
            spectra = np.array(spectra)
            
            # 基本统计信息
            info['stats'] = {
                'wavelength_count': len(wavelengths),
                'spectra_shape': spectra.shape,
                'wavelength_range': (float(wavelengths.min()), float(wavelengths.max())),
                'spectra_range': (float(spectra.min()), float(spectra.max()))
            }
            
            # 检查维度
            if spectra.ndim == 1:
                info['warnings'].append(f"{name}: 1D array detected, will be reshaped to 2D")
            elif spectra.ndim > 2:
                info['valid'] = False
                info['errors'].append(f"{name}: Too many dimensions ({spectra.ndim})")
                return info
            
            # 检查长度匹配
            expected_length = spectra.shape[1] if spectra.ndim == 2 else len(spectra)
            if len(wavelengths) != expected_length:
                diff = abs(len(wavelengths) - expected_length)
                diff_ratio = diff / max(len(wavelengths), expected_length)
                
                if diff_ratio > 0.1:  # 超过10%的差异
                    info['warnings'].append(f"{name}: Large dimension mismatch - wavelengths: {len(wavelengths)}, spectra: {expected_length}")
                else:
                    info['warnings'].append(f"{name}: Minor dimension mismatch - will be aligned automatically")
            
            # 检查数据质量
            if np.any(np.isnan(wavelengths)):
                info['warnings'].append(f"{name}: Wavelengths contain NaN values")
            
            if np.any(np.isnan(spectra)):
                info['warnings'].append(f"{name}: Spectra contain NaN values")
            
            if np.any(np.isinf(spectra)):
                info['warnings'].append(f"{name}: Spectra contain infinite values")
                
        except Exception as e:
            info['valid'] = False
            info['errors'].append(f"{name}: Validation error - {str(e)}")
        
        return info
    
    @staticmethod
    def create_error_plot(ax, error_message, title="Data Error"):
        """
        创建错误显示图表
        
        Args:
            ax: matplotlib轴对象
            error_message: 错误信息
            title: 图表标题
        """
        ax.clear()
        ax.text(0.5, 0.5, f'数据维度错误:\n{error_message}\n\n请检查您的数据格式', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
               fontsize=12, wrap=True)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
    @staticmethod
    def get_alignment_info(wavelengths, *spectra_arrays):
        """
        获取数据对齐信息，用于调试
        
        Args:
            wavelengths: 波长数组
            *spectra_arrays: 光谱数据数组
            
        Returns:
            dict: 对齐信息
        """
        info = {
            'wavelength_length': len(wavelengths) if wavelengths is not None else 0,
            'spectra_info': [],
            'alignment_needed': False,
            'min_length': 0,
            'max_length': 0
        }
        
        lengths = [info['wavelength_length']]
        
        for i, spectra in enumerate(spectra_arrays):
            if spectra is None:
                spectra_info = {'index': i, 'shape': None, 'length': 0}
            else:
                spectra = np.array(spectra)
                if spectra.ndim == 1:
                    length = len(spectra)
                    shape = (1, length)
                else:
                    length = spectra.shape[1]
                    shape = spectra.shape
                
                spectra_info = {'index': i, 'shape': shape, 'length': length}
                lengths.append(length)
            
            info['spectra_info'].append(spectra_info)
        
        valid_lengths = [l for l in lengths if l > 0]
        if valid_lengths:
            info['min_length'] = min(valid_lengths)
            info['max_length'] = max(valid_lengths)
            info['alignment_needed'] = len(set(valid_lengths)) > 1
        
        return info
