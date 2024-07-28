import torch
from .rnn import RNN


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