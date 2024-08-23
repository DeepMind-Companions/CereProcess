import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(4747)
import random
random.seed(4747)

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

class SCNetWave(nn.Module):
    def __init__(self, input_shape):
        super(SCNetWave, self).__init__()
        self.block1 = WaveBlock(50, 16, 3, 8)
        self.block2 = WaveBlock(50, 16, 3, 8)
        self.block3 = WaveBlock(32, 64, 3, 5)
        self.block4 = WaveBlock(32, 64, 3, 5)
        self.block5 = WaveBlock(32, 32, 3, 3)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(50)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = x1 + x2

        #Apply spatial dropout
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv1(x)))
        x1 = self.block3(x)
        x2 = self.block4(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        x = self.block5(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x