"""
This file includes the abstract RNN class.
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

