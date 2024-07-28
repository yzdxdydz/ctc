from typing import Tuple, Sequence, Dict, Type
import torch

from src.ctc import SampleItem, CTC
from src.comparable import Comparable


class MatrixSampleItem(SampleItem):
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


class MatrixCTC(CTC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        super().__init__(dictionary, acceleration)

    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        mat_p = torch.tensor(item.mat_p, device=self.device, dtype=torch.float)
        mat_a = torch.tensor(item.mat_a, device=self.device, dtype=torch.float)
        mat_b = torch.tensor(item.mat_b, device=self.device, dtype=torch.float)
        y_mod = torch.matmul(y, mat_p)
        # forward
        gamma = torch.zeros((y.shape[0], mat_p.shape[1])).to(self.device)
        gamma[0, 0] = y_mod[0, 0]
        gamma[0, 1] = y_mod[0, 1]
        c = gamma[0].sum()
        gamma[0] /= c
        sum_log_c = torch.log(c)
        for t in range(1, y.shape[0]):
            gamma[t] = torch.matmul(mat_a, gamma[t - 1]) * y_mod[t]
            c = gamma[t].sum()
            sum_log_c += torch.log(c)
            gamma[t] /= c
        loss = -torch.log(gamma[-1, -1] + gamma[-1, -2]) - sum_log_c

        # backward
        beta_t = torch.zeros(mat_p.shape[1]).to(self.device)
        beta_t[-1] = y_mod[-1, -1]
        beta_t[-2] = y_mod[-1, -2]
        d = beta_t[-1].sum()
        beta_t /= d
        gamma[-1] *= beta_t
        for t in range(y.shape[0] - 2, -1, -1):
            beta_t = torch.matmul(mat_b, beta_t) * y_mod[t]
            d = beta_t.sum()
            beta_t /= d
            gamma[t] *= beta_t

        z = (gamma / y_mod).sum(dim=-1)
        mu = (torch.matmul(gamma, mat_p.T) / y) / z[:, None]

        return mu, loss

    def set_item_type(self):
        self.item_type = MatrixSampleItem
