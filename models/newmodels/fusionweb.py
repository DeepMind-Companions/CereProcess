import torch
from torch import nn
from torch.nn import functional as F
from models.scnet.scnet import SILM

class FusionMod(nn.Module):
    def __init__(self, in_channels):
        super(FusionMod, self).__init__()
        self.convL1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bnL1 = nn.BatchNorm1d(8)
        self.convL2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bnL2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.convL1.weight)
        nn.init.xavier_uniform_(self.convL2.weight)


        self.convR1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bnR1 = nn.BatchNorm1d(8)
        self.convR2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bnR2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.convR1.weight)
        nn.init.xavier_uniform_(self.convR2.weight)

    def forward(self, input):
        xL1 = F.relu(self.bnL1(self.convL1(input)))
        xR1 = F.relu(self.bnR1(self.convR1(input)))
        x1 = torch.cat((xL1 + xR1, input), dim=1)
        xL2 = F.relu(self.bnL2(self.convL2(x1)))
        xR2 = F.relu(self.bnR2(self.convR2(x1)))
        x = torch.cat((xL2 + xR2, x1), dim=1)
        return x

class FusionSCWave(nn.Module):
    def __init__(self, input_shape):
        super(FusionSCWave, self).__init__()
        self.silm = SILM()
        self.fusion1 = FusionMod(50)
        self.fusion2 = FusionMod(50)
        self.fusion3 = FusionMod(32)
        self.conv1 = nn.Conv1d(in_channels=74, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 56, out_channels = 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.silm(x)
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.fusion1(x)
        x2 = self.fusion2(x)
        x = x1 + x2

        #Apply spatial dropout
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=2, stride=4)
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.fusion3(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x


