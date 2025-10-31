"""
Subpackage containing model implememtation.
"""

from .regression import *
from .logit import *
__all__ = ['LinearRegression', 'CauchyRegression', 'LogisticRegression']