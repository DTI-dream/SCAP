"""
@ author: neo
@ date: 2023-05-18  10:43
@ file_name: __init__.py.PY
@ github: https://github.com/Underson888/
"""
from .field import RawField, Merge, ImageDetectionsField, TextField
from .dataset import COCO
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
