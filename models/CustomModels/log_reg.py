import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(WaveLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
       # self.skip = nn.Conv1d(out_channels, in_channels, 1)
       # self.residual = nn.Conv1d(out_channels, in_channels, 1)
        
    def forward(self, x):
        # x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh*sig
        z = z[:,:,:-self.padding]
        x = x + z
        return x

class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(WaveBlock, self).__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x

class MFFMBlock(nn.Module):
    def __init__(self, in_channels):
        super(MFFMBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)


    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)

class LogReg(nn.Module):
    def __init__(self, input_shape):
        super(LogReg, self).__init__()
        # self.mffm_block1 = MFFMBlock(25)
        self.wave_block1 = WaveBlock(25, 16, 3, 8)
        # self.encoder_layer = nn.TransformerEncoderLayer(49, 1, dropout=0.3, batch_first=True) 
        # self.mffm_block2 = MFFMBlock(65)
        # self.wave_block2 = WaveBlock(65, 32, 3, 5)
        self.fc = nn.Linear(16, 2)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        # gap = F.dropout(gap, 0.05, training=self.training)
        # gmp = F.dropout(gmp, 0.05, training=self.training)
        # gsp = F.dropout(gsp, 0.05, training=self.training)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        x = self.wave_block1(x)
        # x2 = self.mffm_block1(x)
        # x = torch.concat([x1, x2], dim=1)

        # x = F.dropout(x, p=0.3, training=self.training)

        # x = F.max_pool1d(x, kernel_size=2, stride=2)
        # x1 = self.mffm_block2(x)
        # x2 = self.wave_block2(x)
        # x = torch.concat([x1, x2], dim=1)
        # x = F.avg_pool1d(x, kernel_size=5, stride=5)
        # x = x.permute(0, 2, 1)
        # x = self.encoder_layer(x)
        # x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=-1)
        x = self.fc(x)
        return x