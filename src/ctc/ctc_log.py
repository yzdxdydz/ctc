from typing import Tuple, Dict
import torch

from .ctc import CTC
from ..sample_item import SampleItem, ClassicSampleItem
from ..utils import Comparable


class LogCTC(CTC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        super().__init__(dictionary, acceleration)

    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:

        with torch.no_grad():
            log_y = torch.log(y)

        # forward
        gamma = torch.full((y.shape[0], 2 * len(item.p) + 1),
                           fill_value=float('-inf'),
                           dtype=torch.float,
                           device=self.device,
                           requires_grad=False
                           )
        gamma[0, 0] = log_y[0, -1]
        gamma[0, 1] = log_y[0, item.p[0]]
        for t in range(1, y.shape[0]):
            for s in range(2 * len(item.p) + 1):
                if s == 0:
                    gamma[t, s] = gamma[t - 1, s]
                elif s == 1 or \
                        s % 2 == 0 or \
                        item.p[s // 2] == item.p[(s - 2) // 2]:
                    gamma[t, s] = torch.logsumexp(gamma[t-1, s-1:s+1], dim=0)
                else:
                    gamma[t, s] = torch.logsumexp(gamma[t-1, s-2:s+1], dim=0)
                if s % 2 == 0:
                    gamma[t, s] += log_y[t, -1]
                else:
                    gamma[t, s] += log_y[t, item.p[s // 2]]

        loss = -torch.logsumexp(gamma[-1, -2:], dim=0)

        # backward
        beta_t = torch.full((2 * len(item.p) + 1, ),
                            fill_value=float('-inf'),
                            dtype=torch.float,
                            device=self.device,
                            requires_grad=False
                            )
        beta_t[-1] = log_y[-1, -1]
        beta_t[-2] = log_y[-1, item.p[-1]]
        beta_pr = torch.clone(beta_t)
        gamma[-1] += beta_t
        for t in range(y.shape[0] - 2, -1, -1):
            for s in range(2 * len(item.p), -1, -1):
                if s == 2 * len(item.p):
                    beta_t[s] = beta_pr[s]
                elif s == 2 * len(item.p) - 1 or \
                        s % 2 == 0 or \
                        item.p[s // 2] == item.p[(s + 2) // 2]:
                    beta_t[s] = torch.logsumexp(beta_pr[s:s+2], dim=0)
                else:
                    beta_t[s] = torch.logsumexp(beta_pr[s:s + 3], dim=0)
                if s % 2 == 0:
                    beta_t[s] += log_y[t, -1]
                else:
                    beta_t[s] += log_y[t, item.p[s // 2]]
            beta_pr = torch.clone(beta_t)
            gamma[t] += beta_t

        mu = torch.full(y.shape,
                        fill_value=float('-inf'),
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False
                        )
        for t in range(y.shape[0]):
            for s in range(2 * len(item.p) + 1):
                if s % 2 == 0:
                    mu[t, -1] = torch.logsumexp(
                        torch.hstack((mu[t, -1], gamma[t, s])),
                        dim=0)
                else:
                    mu[t, item.p[s // 2]] = torch.logsumexp(
                        torch.hstack((mu[t, item.p[s // 2]], gamma[t, s])),
                        dim=0)

        return torch.exp(loss - log_y + mu), loss

    def set_item_type(self):
        self.item_type = ClassicSampleItem
