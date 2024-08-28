"""
This file provides the abstract base class for CTC algorithms.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Dict, Type, Optional
from time import time
import torch

from .. utils import Comparable, Vector
from .. rnn import RNN
from ..sample_item import SampleItem
from ..sample_generator import SampleGenerator


class CTC(ABC):
    def __init__(self,
                 dictionary: Dict[Comparable, int],
                 acceleration: bool = True
                 ):
        self.dictionary = dictionary

        if acceleration:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.net = None
        self.set_item_type()

    def load_net_from_file(self, file_path: str, to_train: bool = True):
        try:
            self.net = torch.load(file_path, map_location=self.device)
            if to_train:
                self.net.train()
            else:
                self.net.eval()
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return -1
        except Exception as e:
            print(f"Error loading network from file: {e}")
            return -1
        return 0

    def save_net_to_file(self, file_path: str):
        try:
            torch.save(self.net, file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return -1
        except Exception as e:
            print(f"Error saving network to file: {e}")
            return -1
        return 0

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
              sample_generator: SampleGenerator,
              epochs: int = 10000,
              mr: int = 50,
              lr: float = 1e-4,
              optimizer_type: Optional[
                  Type[torch.optim.Optimizer]] = torch.optim.SGD,
              sr: int = 100,
              scheduler_type: Optional[
                  Type[torch.optim.lr_scheduler._LRScheduler]] = None,
              grad_type: str = "u",
              batch_size: int = 1 << 7,
              grad_clip: bool = True
              ):

        if self.net is None:
            print("You should set neural network first")
            return

        optimizer = optimizer_type(params=self.net.parameters(), lr=lr)
        if scheduler_type is not None:
            scheduler = scheduler_type(optimizer=optimizer)
        else:
            scheduler = None

        if grad_type in ["u", "y", "v"]:
            g_type = grad_type
        else:
            print("Gradient type didn't recognized, set by default")
            g_type = "u"

        for epoch in range(epochs):
            start_time = time()
            loss = 0
            sample = sample_generator.generate_sample()

            if batch_size is None:
                batches = [sample]
            else:
                batch_count = len(sample) // batch_size
                batches = [sample[i*batch_size:(i+1)*batch_size]
                           for i in range(batch_count)]

                if len(sample) % batch_size != 0:
                    batches.append(sample[batch_count*batch_size:])

            for batch in batches:
                batch_loss = 0
                tr_data = [self.item_type(item[0],
                                          item[1],
                                          self.dictionary
                                          ) for item in batch]
                optimizer.zero_grad()
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
                    v = None
                    if g_type == "u":
                        u.backward(y - mu)
                    elif g_type == "y":
                        y.backward(-mu / y)
                    else:
                        v = torch.log(y)
                        v.backward(-mu)
                    batch_loss += loss_loc

                if grad_clip:
                    for param in self.net.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(
                                    param.grad).any():
                                print("Gradient contains NaN or inf!")
                                return
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                   max_norm=1.0)

                optimizer.step()

                loss += batch_loss.item()

                del x, y, u, mu, loss_loc, batch_loss
                if v is not None:
                    del v
                if self.device == torch.device('cuda'):
                    torch.cuda.empty_cache()
                elif self.device == torch.device('mps'):
                    torch.mps.empty_cache()
            end_time = time()

            if (epoch + 1) % mr == 0:
                print("Epoch: {0:d}, "
                      "Loss: {1:.10f}, "
                      "Time: {2:d} sec".format(
                        epoch + 1,
                        loss/len(sample),
                        int(end_time - start_time)))

            if scheduler and (epoch + 1) % sr == 0:
                scheduler.step()

    @abstractmethod
    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def set_item_type(self):
        pass
