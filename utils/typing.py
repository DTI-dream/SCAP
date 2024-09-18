"""
@ author: neo
@ date: 2023-06-10  20:54 
@ file_name: typing.PY
@ github: https://github.com/Underson888/
"""
from typing import Union, Sequence, Tuple
import torch

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]
