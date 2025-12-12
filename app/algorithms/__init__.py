# app/algorithms/__init__.py
"""
Core spectral analysis algorithms module
Contains professional algorithms for wavelength selection, spectrum-type-specific processing, model transfer, etc.
"""

from .wavelength_selection import CARS, SPA
from .raman_specific import ModPolyBaseline, RamanFluorescenceRemoval, RamanShiftCalibration
from .mir_specific import AtmosphericCompensation
from .nir_specific import WaterPeakRemoval
from .model_transfer import PDS, SBC
from .baseline_correction import SNIPBaseline

__all__ = [
    # Wavelength Selection
    'CARS',
    'SPA',
    # Raman Specific
    'ModPolyBaseline',
    'RamanFluorescenceRemoval', 
    'RamanShiftCalibration',
    # MIR Specific
    'AtmosphericCompensation',
    # NIR Specific
    'WaterPeakRemoval',
    # Model Transfer
    'PDS',
    'SBC',
    # Baseline Correction
    'SNIPBaseline',
]

