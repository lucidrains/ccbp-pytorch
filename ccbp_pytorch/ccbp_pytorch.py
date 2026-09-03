import torch
from torch import nn
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

# classes

class CCBP(Module):
    def __init__(
        self
    ):
        super().__init__()
