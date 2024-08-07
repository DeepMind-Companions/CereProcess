import torch
from torch import nn
from torch.nn import functional as F

class AlternateLayer(nn.Module):
    def __init__(self, input_shape):
        self.seq_len, self.input_size = input_shape
        assert self.input_size == 15000
        super(AlternateLayer, self).__init__()
        self.timdistLSTM = nn.LSTM(500, 1, 1, batch_first=True)
        self.attFCN = nn.Linear(self.seq_len * 30, 30)
        self.seqLSTM = nn.LSTM(self.seq_len, self.seq_len, batch_first=True)
        self.findense = nn.Linear(self.seq_len, 2)
        self.fintanh = nn.Tanh()
        # Applying the Xavier initialization
        torch.nn.init.xavier_uniform_(self.attFCN.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.findense.weight, gain=1.0)


    def forward(self, x):
        x = x.flip(-1)
        batch_size, seq_len, input_dim = x.size()
        x = x.reshape(batch_size*seq_len*30, 1, 500)
        _, (x, _) = self.timdistLSTM(x)
        x = x.reshape(batch_size, seq_len, 30)
        x = x.transpose(1, 2)
        att = x.reshape(batch_size, seq_len*30)
        att = self.attFCN(att)
        att = F.softmax(att, dim =-1)
        att = att.unsqueeze(-1)
        x = x * att
        x, _ = self.seqLSTM(x)
        x = F.dropout(x, 0.2, training = self.training)
        x = self.findense(x)
        x = self.fintanh(x)
        x = x.reshape(batch_size, 60)
        return x

if __name__ == '__main__':
    input_shape = (32, 22, 15000)
    x = torch.randn(input_shape)
    model = AlternateLayer(input_shape)
    output = model(x)
    print(output.shape)

        

