"""
This file provides the abstract base class for CTC algorithms.
"""


from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Dict, Type, Optional
import torch

from .. utils import Comparable
from .. rnn import RNN
from ..sample_item import SampleItem


class CTC(ABC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        self.dictionary = dictionary
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and acceleration else "cpu"
        )
        self.net = None
        self.set_item_type()

    def load_net_from_file(self, file_path: str):
        try:
            self.net.load_state_dict(torch.load(file_path,
                                                map_location=self.device))
            self.net.eval()
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"Error loading network from file: {e}")

    def save_net_to_file(self, file_path: str):
        try:
            torch.save(self.net.state_dict(), file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"Error saving network to file: {e}")

    def create_net(self,
                   rnn: Type[RNN],
                   input_size: int,
                   hidden_size: int,
                   bidirectional: bool):
        self.net = rnn(input_size=input_size,
                       output_size=len(self.dictionary) + 1,
                       hidden_size=hidden_size,
                       bidirectional=bidirectional
                       ).to(self.device)

    def train(self,
              training_sample: Sequence[
                  Tuple[Sequence[float], Sequence[int]]
              ],
              epochs: int = 10000,
              mr: int = 50,
              lr: float = 1e-4,
              optimizer_type: Optional[
                  Type[torch.optim.Optimizer]] = torch.optim.SGD,
              sr: int = 100,
              scheduler_type: Optional[
                  Type[torch.optim.lr_scheduler._LRScheduler]] = None,
              grad_type: str = "u"
              ):

        if self.net is None:
            print("You should set neural network first")
            return

        optimizer = optimizer_type(params=self.net.parameters(), lr=lr)
        if scheduler_type is not None:
            scheduler = scheduler_type(optimizer=optimizer, gamma=0.9)
        else:
            scheduler = None

        tr_data = [self.item_type(item[0],
                                  item[1],
                                  self.dictionary
                                  ) for item in training_sample]

        if grad_type in ["u", "y", "v"]:
            g_type = grad_type
        else:
            print("Gradient type didn't recognized, set by default")
            g_type = "u"

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = 0
            for item in tr_data:
                x = torch.tensor(item.x,
                                 device=self.device,
                                 dtype=torch.float
                                 )
                if x.dim() == 1:
                    x.unsqueeze_(1)
                u = self.net(x)
                y = torch.softmax(u, dim=-1)
                mu, loss_loc = self.mu_loss(y, item)
                if g_type == "u":
                    u.backward(y - mu)
                elif g_type == "y":
                    y.backward(-mu / y)
                else:
                    v = torch.log(y)
                    v.backward(-mu)
                loss += loss_loc
            optimizer.step()
            if scheduler and (epoch + 1) % sr == 0:
                scheduler.step()
            if (epoch + 1) % mr == 0:
                print("Epoch: {0:d}, "
                      "Loss: {1:.10f}".format(epoch+1,
                                              loss.item()
                                              )
                      )

    @abstractmethod
    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def set_item_type(self):
        pass
