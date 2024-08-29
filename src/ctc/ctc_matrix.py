import torch
from typing import Tuple, Sequence, Dict

from .ctc import CTC
from ..sample_item import SampleItem, MatrixSampleItem
from ..utils import Comparable


class MatrixCTC(CTC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        super().__init__(dictionary, acceleration)

    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        mat_p = item.mat_p
        mat_a = item.mat_a
        mat_b = item.mat_b
        with torch.no_grad():
            y_mod = torch.matmul(y, mat_p)
        # forward
        gamma = torch.zeros((y.shape[0], mat_p.shape[1]),
                            dtype=torch.float,
                            device=self.device,
                            requires_grad=False)
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
        beta_t = torch.zeros((mat_p.shape[1], ),
                             dtype=torch.float,
                             device=self.device,
                             requires_grad=False)
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
        with torch.no_grad():
            mu = (torch.matmul(gamma, mat_p.T) / y) / z[:, None]

        return mu, loss

    def set_item_type(self):
        self.item_type = MatrixSampleItem
