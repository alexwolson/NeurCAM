"""
NeurCAM: Neural Clustering Additive Model

A Python package for interpretable clustering using neural networks.
"""

from neurcam.model import NeurCAM, NeurCAMModel
from neurcam.loss import FuzzyCMeansLoss

__version__ = "0.1.0"
__all__ = ["NeurCAM", "NeurCAMModel", "FuzzyCMeansLoss"]
