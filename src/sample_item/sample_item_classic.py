import torch

from typing import Sequence, Dict

from .sample_item import SampleItem
from ..utils import Comparable, Vector


class ClassicSampleItem(SampleItem):
    def __init__(self,
                 input: Sequence[Vector],
                 target: Sequence[Comparable],
                 dictionary: Dict[Comparable, int],
                 device: torch.device
                 ):
        super().__init__(input, target, dictionary, device)

    def preprocess_target(self,
                          target: Sequence[Comparable],
                          dictionary: Dict[Comparable, int]
                          ):
        self.p = [dictionary[el] for el in target]
