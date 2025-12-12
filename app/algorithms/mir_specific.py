# app/algorithms/mir_specific.py
"""
MIR/FTIR光谱专用算法
包含大气补偿（CO₂、H₂O干扰去除）等MIR光谱分析必需的算法
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class AtmosphericCompensation:
    """
    MIR大气补偿算法
    
    MIR/FTIR光谱受大气中CO₂和H₂O强烈吸收，必须进行大气补偿。
    
    主要干扰波段:
    - CO₂: 2349 cm⁻¹ (主峰), 2280-2400 cm⁻¹ (吸收带)
    - H₂O: 1595 cm⁻¹ (弯曲振动), 3400-3900 cm⁻¹ (伸缩振动)
    - H₂O: 1300-1900 cm⁻¹ (广泛吸收)
    
    补偿方法:
    1. Background subtraction: 使用背景光谱相减
    2. Reference scaling: 使用参考光谱缩放减除
    3. Interpolation: 插值替换干扰区域
    
    这是审稿人#2暗示的"Atmospheric Correction"功能！
    
    参考文献:
    Griffiths, P. R., & De Haseth, J. A. (2007).
    Fourier Transform Infrared Spectrometry (2nd ed.).
    John Wiley & Sons.
    """
    
    # CO₂和H₂O的标准吸收波段 (cm⁻¹)
    CO2_REGIONS = [
        (2280, 2400),  # CO₂主吸收带
        (3500, 3800),  # CO₂弱吸收带
        (660, 680),    # CO₂弱吸收带
    ]
    
    H2O_REGIONS = [
        (1300, 1900),  # H₂O弯曲振动
        (3200, 3900),  # H₂O伸缩振动
    ]
    
    def __init__(self, 
                 method: str = 'interpolation',
                 compensate_co2: bool = True,
                 compensate_h2o: bool = True,
                 background_spectrum: Optional[np.ndarray] = None,
                 scale_factor: float = 1.0):
        """
        初始化大气补偿算法
        
        Parameters:
        -----------
        method : str
            补偿方法: 'interpolation', 'background', 'reference'
        compensate_co2 : bool
            是否补偿CO₂
        compensate_h2o : bool
            是否补偿H₂O
        background_spectrum : ndarray, optional
            背景光谱（用于background方法）
        scale_factor : float
            缩放因子（用于reference方法）
        """
        self.method = method.lower()
        self.compensate_co2 = compensate_co2
        self.compensate_h2o = compensate_h2o
        self.background_spectrum = background_spectrum
        self.scale_factor = scale_factor
        
        if self.method not in ['interpolation', 'background', 'reference']:
            raise ValueError(f"未知的补偿方法: {method}")
    
    def fit_transform(self, 
                      wavenumbers: np.ndarray, 
                      X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用大气补偿
        
        Parameters:
        -----------
        wavenumbers : ndarray
            波数轴 (cm⁻¹)
        X : ndarray, shape (n_samples, n_wavelengths)
            原始MIR光谱
            
        Returns:
        --------
        wavenumbers : ndarray
            波数轴（不变）
        X_compensated : ndarray
            补偿后的光谱
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X.copy()
        
        n_samples = X_values.shape[0]
        
        print(f"🔧 MIR atmospheric compensation: {self.method} method")
        if self.compensate_co2:
            print(f"  Compensating CO₂: {len(self.CO2_REGIONS)} bands")
        if self.compensate_h2o:
            print(f"  Compensating H₂O: {len(self.H2O_REGIONS)} bands")
        
        # 确定需要补偿的区域
        regions_to_compensate = []
        if self.compensate_co2:
            regions_to_compensate.extend(self.CO2_REGIONS)
        if self.compensate_h2o:
            regions_to_compensate.extend(self.H2O_REGIONS)
        
        if self.method == 'interpolation':
            X_compensated = self._interpolation_compensation(wavenumbers, X_values, regions_to_compensate)
        elif self.method == 'background':
            X_compensated = self._background_compensation(wavenumbers, X_values, regions_to_compensate)
        elif self.method == 'reference':
            X_compensated = self._reference_compensation(wavenumbers, X_values, regions_to_compensate)
        
        print(f"✅ Atmospheric compensation completed")
        
        if isinstance(X, pd.DataFrame):
            X_compensated = pd.DataFrame(X_compensated, columns=X.columns, index=X.index)
        
        return wavenumbers, X_compensated
    
    def _interpolation_compensation(self, 
                                   wavenumbers: np.ndarray, 
                                   X: np.ndarray, 
                                   regions: List[Tuple[float, float]]) -> np.ndarray:
        """
        插值法大气补偿
        
        在干扰区域的边界进行插值，替换受干扰的数据点
        """
        X_compensated = X.copy()
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            spectrum = X[i, :]
            
            for region_start, region_end in regions:
                # 找到区域索引
                mask = (wavenumbers >= region_start) & (wavenumbers <= region_end)
                
                if not np.any(mask):
                    continue
                
                # 找到边界点
                region_indices = np.where(mask)[0]
                if len(region_indices) == 0:
                    continue
                
                start_idx = region_indices[0]
                end_idx = region_indices[-1]
                
                # 使用边界点进行线性插值
                if start_idx > 0 and end_idx < len(wavenumbers) - 1:
                    # 取边界外5个点用于插值
                    left_idx = max(0, start_idx - 5)
                    right_idx = min(len(wavenumbers) - 1, end_idx + 5)
                    
                    # 不包含干扰区域的点
                    good_mask = np.ones(len(wavenumbers), dtype=bool)
                    good_mask[start_idx:end_idx+1] = False
                    good_mask[:left_idx] = False
                    good_mask[right_idx+1:] = False
                    
                    if np.sum(good_mask) >= 2:
                        # 插值
                        interpolator = interp1d(
                            wavenumbers[good_mask], 
                            spectrum[good_mask],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        spectrum[mask] = interpolator(wavenumbers[mask])
            
            X_compensated[i, :] = spectrum
        
        return X_compensated
    
    def _background_compensation(self, 
                                wavenumbers: np.ndarray, 
                                X: np.ndarray, 
                                regions: List[Tuple[float, float]]) -> np.ndarray:
        """
        背景减除法大气补偿
        
        使用背景光谱相减
        """
        if self.background_spectrum is None:
            print("  ⚠ No background spectrum provided, using first spectrum as background")
            background = X[0, :]
        else:
            background = self.background_spectrum
        
        X_compensated = X - background
        return X_compensated
    
    def _reference_compensation(self, 
                               wavenumbers: np.ndarray, 
                               X: np.ndarray, 
                               regions: List[Tuple[float, float]]) -> np.ndarray:
        """
        参考光谱缩放法大气补偿
        
        使用标准大气光谱缩放减除
        """
        # 如果有背景光谱，用作参考
        if self.background_spectrum is not None:
            reference = self.background_spectrum
        else:
            # 使用数据集的平均光谱作为参考
            reference = np.mean(X, axis=0)
        
        X_compensated = X.copy()
        
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            
            for region_start, region_end in regions:
                mask = (wavenumbers >= region_start) & (wavenumbers <= region_end)
                
                if not np.any(mask):
                    continue
                
                # 在干扰区域估计最佳缩放因子
                region_spectrum = spectrum[mask]
                region_reference = reference[mask]
                
                if np.std(region_reference) > 0:
                    # 最小二乘估计缩放因子
                    scale = np.sum(region_spectrum * region_reference) / np.sum(region_reference ** 2)
                    scale = scale * self.scale_factor
                    
                    # 减除缩放后的参考光谱
                    spectrum[mask] -= scale * region_reference
            
            X_compensated[i, :] = spectrum
        
        return X_compensated
    
    @staticmethod
    def detect_atmospheric_interference(wavenumbers: np.ndarray, 
                                       spectrum: np.ndarray) -> Dict[str, bool]:
        """
        检测光谱中是否存在大气干扰
        
        Parameters:
        -----------
        wavenumbers : ndarray
            波数轴
        spectrum : ndarray
            光谱数据
            
        Returns:
        --------
        interference : dict
            {'co2': bool, 'h2o': bool}
        """
        interference = {'co2': False, 'h2o': False}
        
        # 检测CO₂
        co2_mask = (wavenumbers >= 2280) & (wavenumbers <= 2400)
        if np.any(co2_mask):
            co2_region = spectrum[co2_mask]
            # 如果该区域有异常强吸收（负值或非常低的值）
            if np.min(co2_region) < np.percentile(spectrum, 5):
                interference['co2'] = True
        
        # 检测H₂O
        h2o_mask = (wavenumbers >= 1300) & (wavenumbers <= 1900)
        if np.any(h2o_mask):
            h2o_region = spectrum[h2o_mask]
            if np.min(h2o_region) < np.percentile(spectrum, 5):
                interference['h2o'] = True
        
        return interference


def atmospheric_compensation(wavenumbers: np.ndarray, 
                            X: np.ndarray,
                            method: str = 'interpolation',
                            compensate_co2: bool = True,
                            compensate_h2o: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    大气补偿便捷函数
    
    Parameters:
    -----------
    wavenumbers : ndarray
        波数轴
    X : ndarray
        光谱数据
    method : str
        补偿方法
    compensate_co2 : bool
        是否补偿CO₂
    compensate_h2o : bool
        是否补偿H₂O
        
    Returns:
    --------
    wavenumbers : ndarray
        波数轴
    X_compensated : ndarray
        补偿后的光谱
    """
    compensator = AtmosphericCompensation(
        method=method,
        compensate_co2=compensate_co2,
        compensate_h2o=compensate_h2o
    )
    return compensator.fit_transform(wavenumbers, X)

