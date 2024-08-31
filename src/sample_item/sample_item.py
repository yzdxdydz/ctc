import torch

from typing import Sequence, Dict
from abc import ABC, abstractmethod

from ..utils import Comparable, Vector


class SampleItem(ABC):
    def __init__(self,
                 input: Sequence[Vector],
                 target: Sequence[Comparable],
                 dictionary: Dict[Comparable, int],
                 device: torch.device
                 ):
        self.device = device
        self.x = torch.tensor(input,
                              dtype=torch.float,
                              device=self.device,
                              requires_grad=False)
        p = torch.tensor([dictionary[el] for el in target],
                         dtype=torch.long,
                         device=self.device,
                         requires_grad=False)
        self.preprocess_target(p, len(dictionary)+1)

    @abstractmethod
    def preprocess_target(self,
                          p: torch.tensor,
                          y_size: int
                          ):
        pass
