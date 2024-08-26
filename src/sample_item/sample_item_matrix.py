from typing import Sequence, Dict

from .sample_item import SampleItem
from ..utils import Comparable, Vector


class MatrixSampleItem(SampleItem):
    def __init__(self,
                 input: Sequence[Vector],
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
        self.mat_p = [[0.] * (2 * len(p) + 1)
                      for _ in range(len(dictionary)+1)
                      ]
        self.mat_a = [[0.] * (2 * len(p) + 1)
                      for _ in range(2 * len(p) + 1)
                      ]
        self.mat_b = [[0.] * (2 * len(p) + 1)
                      for _ in range(2 * len(p) + 1)
                      ]
        for s in range(2 * len(p) + 1):
            if s % 2 == 0:
                self.mat_p[-1][s] = 1.
            else:
                self.mat_p[p[s // 2]][s] = 1.

            if s == 0:
                self.mat_a[s][s] = 1.
            elif s == 1 or \
                    s % 2 == 0 or \
                    p[s // 2] == p[(s - 2) // 2]:
                self.mat_a[s][s] = 1.
                self.mat_a[s][s - 1] = 1.
            else:
                self.mat_a[s][s] = 1.
                self.mat_a[s][s - 1] = 1.
                self.mat_a[s][s - 2] = 1.

            if s == 2 * len(p):
                self.mat_b[s][s] = 1.
            elif s == 2 * len(p) - 1 or \
                    s % 2 == 0 or \
                    p[s // 2] == p[(s + 2) // 2]:
                self.mat_b[s][s] = 1.
                self.mat_b[s][s + 1] = 1.
            else:
                self.mat_b[s][s] = 1.
                self.mat_b[s][s + 1] = 1.
                self.mat_b[s][s + 2] = 1.
