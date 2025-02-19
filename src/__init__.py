import logging
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

# # Import the model, data module, and helper functions
# from .SoftOrdering1DCNN import SoftOrdering1DCNN
# from .RNASeqData import RNASeqData
# from .func import *  # Import all helper functions
# from .utils import *  # Import all helper functions

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("Soft Ordered CNN for Omics: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Prevent double outputs
logger.propagate = False

# # Define exports for the package
# __all__ = ["SoftOrdering1DCNN", "RNASeqData"]

# # Set version information
# # __version__ = version("Soft-Ordered-CNN")
from .utils import SimpleMLP, SimpleCNN_3CH, SimpleCNN, PCamDataset, load_training_data, get_dimensions

__all__ = [
    "SimpleMLP",
    "SimpleMLP_3CH",
    "SimpleCNN_3CH",
    "SimpleCNN",
    "PCamDataset",
    "load_training_data",
    "get_dimensions"
]
