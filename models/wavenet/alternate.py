import torch
from torch import nn
from torch.nn import functional as F

class AlternateLayer(nn.Module):
    def __init__(self):
        super(AlternateLayer, self).__init__()
        self.timdistLSTM = nn.LSTM(500, 1, 64, batch_first=True)

    def forward(self, x):
        x = x.flip(-1)
        batch_size, seq_len, input_dim = x.size()
        x = x.reshape(batch_size*seq_len, 30, 500)
        x, _ = self.timdistLSTM(x)
        x = x.reshape(batch_size, seq_len, 30)

        

