from typing import Sequence, Dict

from .sample_item import SampleItem
from ..utils import Comparable


class ClassicSampleItem(SampleItem):
    def __init__(self,
                 input: Sequence[float],
                 target: Sequence[Comparable],
                 dictionary: Dict[Comparable, int]
                 ):
        super().__init__(input, target, dictionary)

    def preprocess_target(self,
                          target: Sequence[Comparable],
                          dictionary: Dict[Comparable, int]
                          ):
        self.p = []
        for el in target:
            self.p.append(dictionary[el])
