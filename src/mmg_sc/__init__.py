from .model import RMGE        
from .integration import CrossSpeciesAligner,HomologyMapper
from .mapping import cell_mappings   

__version__ = "0.1.0"
__author__ = "Yongkang Sun"
__all__ = [
    "RMGE", 
    "CrossSpeciesAligner", 
    "HomologyMapper",
    "cell_mappings",
]


import sys
sys.modules['mmg'] = sys.modules[__name__]

