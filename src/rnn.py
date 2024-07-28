"""
This file includes the abstract RNN class as well as its various
implementations.
"""

from abc import ABC, abstractmethod
import torch


class RNN(ABC, torch.nn.Module):
    """
    Abstract base class for reccurent neural network.

    We assume that the input would be at least 2-dimensional tensor,
    where -1th dim stands for input size, -2th dim stands for time
    length of the sequence. Then the output has the same number of
    dimension, where -1th dim stands for output size, -2th dim stands
    for time length of the sequence. Other dimension may refer to
    batch size, etc.

    Attributes:
        input_size (int):
            Size of each element in input sequence.
        output_size (int):
            Size of each element in output sequence. Generally it is
            equal to size of the dictionary + 1 for blank element.
        hidden_size (int):
            Output size of each hidden layer.
        bidirectional (bool):
            If the network deals with both previous and future data.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 bidirectional: bool
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    @abstractmethod
    def forward(self, x):
        pass


class LSTM(RNN):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 bidirectional: bool
                 ):
        super().__init__(input_size,
                         output_size,
                         hidden_size,
                         bidirectional
                         )
        self.model = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   bidirectional=bidirectional)
        if bidirectional:
            self.W = torch.nn.Linear(2*hidden_size, output_size)
        else:
            self.W = torch.nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)

        torch.nn.init.uniform_(self.W.weight, -0.1, 0.1)
        if self.W.bias is not None:
            torch.nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        return self.W(self.model(x)[0])
