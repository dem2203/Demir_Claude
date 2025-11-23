"""
DEMIR AI v8.0 - Core Components
Signal generation, validation, and data verification
"""

from .signal_generator import SignalGenerator
from .signal_validator import SignalValidator
from .data_validator import DataValidator, RealDataVerifier, MockDataDetector

__all__ = [
    'SignalGenerator',
    'SignalValidator', 
    'DataValidator',
    'RealDataVerifier',
    'MockDataDetector'
]

# Version
__version__ = '8.0'
