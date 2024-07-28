from typing import Sequence, Dict

from .sample_item import SampleItem
from ..utils import Comparable


class LogmatrixSampleItem(SampleItem):
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
        p = []
        for el in target:
            p.append(dictionary[el])
        self.mat_p = [[float('-inf')] * (2 * len(p) + 1)
                      for _ in range(len(dictionary)+1)
                      ]
        self.mat_a = [[float('-inf')] * (2 * len(p) + 1)
                      for _ in range(2 * len(p) + 1)
                      ]
        self.mat_b = [[float('-inf')] * (2 * len(p) + 1)
                      for _ in range(2 * len(p) + 1)
                      ]
        for s in range(2 * len(p) + 1):
            if s % 2 == 0:
                self.mat_p[-1][s] = 0.
            else:
                self.mat_p[p[s // 2]][s] = 0.

            if s == 0:
                self.mat_a[s][s] = 0.
            elif s == 1 or \
                    s % 2 == 0 or \
                    p[s // 2] == p[(s - 2) // 2]:
                self.mat_a[s][s] = 0.
                self.mat_a[s][s - 1] = 0.
            else:
                self.mat_a[s][s] = 0.
                self.mat_a[s][s - 1] = 0.
                self.mat_a[s][s - 2] = 0.

            if s == 2 * len(p):
                self.mat_b[s][s] = 0.
            elif s == 2 * len(p) - 1 or \
                    s % 2 == 0 or \
                    p[s // 2] == p[(s + 2) // 2]:
                self.mat_b[s][s] = 0.
                self.mat_b[s][s + 1] = 0.
            else:
                self.mat_b[s][s] = 0.
                self.mat_b[s][s + 1] = 0.
                self.mat_b[s][s + 2] = 0.