"""
MibiLoader Package

This package provides functionality for loading and processing MIBI data.
"""
from .mibi_loader_utils import mibi_loader, segmentation_grouper , find_files_ending # Adjust the import as needed
from .mibi_eroder_utils import mibi_eroder, mibi_eroder_parallel
from .data_explore_utils import data_exploration_plots


__all__ = [
    "mibi_loader",
    "segmentation_grouper",
    "find_files_ending",
    "mibi_eroder",
    "mibi_eroder_parallel"
    "data_exploration_plots"
]


__version__ = "1.0.0"