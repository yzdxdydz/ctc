from typing import Tuple, Sequence, Dict
import torch

from .ctc import CTC
from ..sample_item import SampleItem, ClassicSampleItem
from ..utils import Comparable


class ClassicCTC(CTC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        super().__init__(dictionary, acceleration)

    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        # forward
        gamma = torch.zeros((y.shape[0], 2 * len(item.p) + 1),
                            dtype=torch.float,
                            device=self.device,
                            requires_grad=False)
        gamma[0, 0] = y[0, -1]
        gamma[0, 1] = y[0, item.p[0]]
        c = gamma[0].sum()
        gamma[0] /= c
        sum_log_c = torch.log(c)
        for t in range(1, y.shape[0]):
            for s in range(2 * len(item.p) + 1):
                if s == 0:
                    gamma[t, s] = gamma[t - 1, s]
                elif s == 1 or \
                        s % 2 == 0 or \
                        item.p[s // 2] == item.p[(s - 2) // 2]:
                    gamma[t, s] = gamma[t - 1, s - 1:s + 1].sum()
                else:
                    gamma[t, s] = gamma[t - 1, s - 2:s + 1].sum()
                if s % 2 == 0:
                    gamma[t, s] *= y[t, -1]
                else:
                    gamma[t, s] *= y[t, item.p[s // 2]]
            c = gamma[t].sum()
            sum_log_c += torch.log(c)
            gamma[t] /= c

        loss = -torch.log(gamma[-1, -1] + gamma[-1, -2]) - sum_log_c

        # backward
        beta_t = torch.zeros((2 * len(item.p) + 1, ),
                             dtype=torch.float,
                             device=self.device,
                             requires_grad=False)
        beta_t[-1] = y[-1, -1]
        beta_t[-2] = y[-1, item.p[-1]]
        d = beta_t.sum()
        beta_t /= d
        beta_pr = torch.clone(beta_t)
        gamma[-1] *= beta_t
        for t in range(y.shape[0] - 2, -1, -1):
            for s in range(2 * len(item.p), -1, -1):
                if s == 2 * len(item.p):
                    beta_t[s] = beta_pr[s:s+1].sum()
                elif s == 2 * len(item.p) - 1 or \
                        s % 2 == 0 or \
                        item.p[s // 2] == item.p[(s + 2) // 2]:
                    beta_t[s] = beta_pr[s:s + 2].sum()
                else:
                    beta_t[s] = beta_pr[s:s + 3].sum()
                if s % 2 == 0:
                    beta_t[s] *= y[t, -1]
                else:
                    beta_t[s] *= y[t, item.p[s // 2]]
            d = beta_t.sum()
            beta_t /= d
            beta_pr = torch.clone(beta_t)
            gamma[t] *= beta_t

        zeta = torch.clone(gamma)
        for t in range(y.shape[0]):
            for s in range(2 * len(item.p) + 1):
                if s % 2 == 0:
                    zeta[t, s] /= y[t, -1]
                else:
                    zeta[t, s] /= y[t, item.p[s // 2]]
        z = zeta.sum(dim=-1)
        mu = torch.zeros_like(y, requires_grad=False)
        for t in range(y.shape[0]):
            for s in range(2 * len(item.p) + 1):
                if s % 2 == 0:
                    mu[t, -1] += gamma[t, s]
                else:
                    mu[t, item.p[s // 2]] += gamma[t, s]
            mu[t] /= z[t]

        with torch.no_grad():
            return mu/y, loss

    def set_item_type(self):
        self.item_type = ClassicSampleItem
