"""
This file provides the abstract base class for CTC algorithms.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Dict, Type, Optional
from time import time
from copy import deepcopy

from .. utils import Comparable, Vector
from .. rnn import RNN
from ..sample_item import SampleItem


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

    def preprocess_sample(self,
                          sample: Sequence[
                              Tuple[np.ndarray, Sequence[Comparable]]
                          ],
                          max_item_size: int,
                          batch_size: int,
                          padding: Tuple[int, float, Comparable] = None):

        # padding: (length, pause_value, pause_symbol)
        indices = np.arange(len(sample))
        np.random.shuffle(indices)
        concat_sample = []
        x, z = sample[indices[0]]
        long_z = deepcopy(z)
        long_x = x
        for i in indices[1:]:
            x, z = sample[indices[i]]
            if x.shape[0] + long_x.shape[0] <= max_item_size:
                if padding is None:
                    long_x = np.concatenate((
                        long_x, x
                    ))
                    long_z += z
                else:
                    long_x = np.concatenate((
                        long_x,
                        np.full((padding[0], ) + x.shape[1:], padding[1]),
                        x
                    ))
                    long_z += [padding[2]] + z
            else:
                concat_sample.append((long_x, long_z))
                long_x = x
                long_z = deepcopy(z)
        concat_sample.append((long_x, long_z))

        batch_count = len(concat_sample) // batch_size
        batches = [concat_sample[i * batch_size:(i + 1) * batch_size]
                   for i in range(batch_count)]

        if len(concat_sample) % batch_size != 0:
            batches.append(concat_sample[batch_count * batch_size:])
            batch_count += 1

        return batches

    def train(self,
              sample: Sequence[
                     Tuple[Sequence[Vector], Sequence[Comparable]]
              ],
              epochs: int = 100,
              mr: int = 50,
              lr: float = 1e-4,
              optimizer_type: Optional[
                  Type[torch.optim.Optimizer]] = torch.optim.SGD,
              sr: int = 100,
              scheduler_type: Optional[
                  Type[torch.optim.lr_scheduler._LRScheduler]] = None,
              grad_type: str = "u",
              batch_size: int = 1 << 7,
              max_item_size: int = 1 << 8,
              padding: Tuple[int, float, Comparable] = None,
              grad_clip: bool = True,
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

        batches = self.preprocess_sample(sample,
                                         max_item_size,
                                         batch_size,
                                         padding)

        for batch_num in range(len(batches)):
            print(f"Batch: {batch_num + 1} / {len(batches)}")
            tr_data = [self.item_type(item[0],
                                      item[1],
                                      self.dictionary,
                                      self.device
                                      ) for item in batches[batch_num]]
            total_length = sum(len(item[1]) for item in batches[batch_num])

            for epoch in range(epochs):
                start_time = time()
                loss = torch.zeros(1,
                                   dtype=torch.float,
                                   device=self.device,
                                   requires_grad=False)
                optimizer.zero_grad()
                for item in tr_data:
                    x = item.x
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
                    loss += loss_loc

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
                end_time = time()

                if (epoch + 1) % mr == 0:
                    print("Epoch: {0:d}, "
                          "Loss: {1:.10f}, "
                          "Time: {2:d} sec".format(epoch + 1,
                                                   loss.item() / total_length,
                                                   int(end_time - start_time)
                                                   )
                          )

                if scheduler and (epoch + 1) % sr == 0:
                    scheduler.step()

    @abstractmethod
    def mu_loss(self, y: torch.tensor, item: SampleItem) -> \
            Tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def set_item_type(self):
        pass
