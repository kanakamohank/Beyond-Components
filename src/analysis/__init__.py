"""
Analysis package for arithmetic circuit discovery.

Modules:
    fourier_discovery: DFT-based bottom-up analysis of residual stream
                       activations to find periodic number representations.
    fourier_plots: Visualization utilities for Fourier discovery results.
    experiment_history: Persistent append-only experiment results logging.
"""

from .fourier_discovery import (
    FourierDiscovery,
    FourierResult,
    LayerFourierResult,
)
from .fourier_plots import plot_all
from .experiment_history import (
    append_experiment_result,
    load_experiment_history,
    print_experiment_history,
)

__all__ = [
    "FourierDiscovery",
    "FourierResult",
    "LayerFourierResult",
    "plot_all",
    "append_experiment_result",
    "load_experiment_history",
    "print_experiment_history",
]
