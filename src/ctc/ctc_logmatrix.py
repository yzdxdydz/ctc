from typing import Tuple, Sequence, Dict
import torch

from .ctc import CTC
from ..sample_item import SampleItem, LogmatrixSampleItem
from ..utils import Comparable, logmatmulexp


class MatrixLogCTC(CTC):
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
            log_y = torch.log(y)
        log_y_mod = logmatmulexp(log_y, mat_p)
        # forward
        gamma = torch.log(torch.zeros((y.shape[0],
                                       mat_p.shape[1]),
                                      device=self.device,
                                  requires_grad=False)
                          )
        gamma[0, 0] = log_y_mod[0, 0]
        gamma[0, 1] = log_y_mod[0, 1]
        for t in range(1, y.shape[0]):
            gamma[t] = logmatmulexp(mat_a, gamma[t - 1]) + log_y_mod[t]

        loss = -torch.logsumexp(gamma[-1, -2:], dim=0)

        # backward
        beta_t = torch.log(torch.zeros(mat_p.shape[1],
                                      device=self.device,
                                  requires_grad=False)
                          )
        beta_t[-1] = log_y_mod[-1, -1]
        beta_t[-2] = log_y_mod[-1, -2]
        gamma[-1] += beta_t
        for t in range(y.shape[0] - 2, -1, -1):
            beta_t = logmatmulexp(mat_b, beta_t) + log_y_mod[t]
            gamma[t] += beta_t

        mu = logmatmulexp(gamma, mat_p.T)

        return torch.exp(loss - log_y + mu), loss

    def set_item_type(self):
        self.item_type = LogmatrixSampleItem
