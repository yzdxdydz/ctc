import torch

from typing import Sequence, Dict

from .sample_item import SampleItem
from ..utils import Comparable, Vector


class MatrixSampleItem(SampleItem):
    def __init__(self,
                 input: Sequence[Vector],
                 target: Sequence[Comparable],
                 dictionary: Dict[Comparable, int],
                 device: torch.device
                 ):
        super().__init__(input, target, dictionary, device)

    def preprocess_target(self,
                          p: torch.tensor,
                          y_size: int
                          ):
        size = 2 * p.shape[0] + 1

        self.mat_p = torch.zeros((y_size, size),
                                 dtype=torch.float,
                                 device=self.device,
                                 requires_grad=False)
        self.mat_a = torch.zeros((size, size),
                                 dtype=torch.float,
                                 device=self.device,
                                 requires_grad=False)
        self.mat_b = torch.zeros((size, size),
                                 dtype=torch.float,
                                 device=self.device,
                                 requires_grad=False)

        indices = torch.arange(size, device=self.device)
        odd_indices = indices[indices % 2 == 1]

        self.mat_p[-1, indices[indices % 2 == 0]] = 1.0
        self.mat_p[p[odd_indices // 2], odd_indices] = 1.0

        self.mat_a[indices, indices] = 1.0
        self.mat_a[indices[1:], indices[1:] - 1] = 1.0
        condition_a = (p[odd_indices[1:] // 2] !=
                       p[(odd_indices[1:] - 2) // 2])
        self.mat_a[odd_indices[1:], odd_indices[1:] - 2] = \
            torch.where(condition_a,
                        1.0,
                        0.0
                        )

        self.mat_b[indices, indices] = 1.0
        self.mat_b[indices[:-1], indices[:-1] + 1] = 1.0
        condition_b = (p[odd_indices[:-1] // 2] !=
                       p[(odd_indices[:-1] + 2) // 2])
        self.mat_b[odd_indices[:-1], odd_indices[:-1] + 2] = \
            torch.where(condition_b,
                        1.0,
                        0.0
                        )
