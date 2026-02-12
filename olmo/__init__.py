from .config import *
from .custom_hf_model import *
from .custom_hf_model_config import *
from .model import *
from .tokenizer import *


def check_install(cuda: bool = False):
    import torch

    from .version import VERSION

    if cuda:
        assert torch.cuda.is_available(), "CUDA is not available!"
        print("CUDA available")

    print(f"OLMo v{VERSION} installed")
