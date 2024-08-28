from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from ..sample_item import SampleItem
from ..utils import Vector, Comparable


class SampleGenerator(ABC):
    @abstractmethod
    def generate_sample(self) -> Sequence[
                  Tuple[Sequence[Vector], Sequence[Comparable]]
                 ]:
        pass

