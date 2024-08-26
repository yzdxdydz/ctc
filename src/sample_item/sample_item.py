from typing import Sequence, Dict
from abc import ABC, abstractmethod

from ..utils import Comparable, Vector


class SampleItem(ABC):
    def __init__(self,
                 input: Sequence[Vector],
                 target: Sequence[Comparable],
                 dictionary: Dict[Comparable, int]
                 ):
        self.x = input
        self.preprocess_target(target, dictionary)

    @abstractmethod
    def preprocess_target(self,
                          target: Sequence[Comparable],
                          dictionary: Dict[Comparable, int]
                          ):
        pass
