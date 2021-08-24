import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

class SAWB(nn.Module):
    def __init__(self, num_classes=9, dataset_name='pavia'):
        super(SAWB, self).__init__()
        self.name = 'FULL'
        self.conv1 = nn.Conv1d(1, 24, 11)
        self.bn1 = nn.BatchNorm1d(24)
        self.conv2 = nn.Conv1d(24, 24, 11, 1, 5, bias=False)
        self.bn2 = nn.BatchNorm1d(24)
        self.conv3 = nn.Conv1d(24, 24, 11, 1, 5, bias=False)
        self.bn3 = nn.BatchNorm1d(24)
        self.bnr1 = nn.BatchNorm1d(24)
        self.conv4 = nn.Conv1d(24, 128, 5, 1, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 24, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(24)
        self.conv6 = nn.Conv1d(24, 24, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(24)
        self.conv7 = nn.Conv1d(24, 24, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm1d(24)
        self.conv8 = nn.Conv1d(24, 24, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm1d(24)       
        self.bnr2 = nn.BatchNorm1d(24)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace = True)
        if dataset_name == 'pavia':
            self.fc = nn.Linear(24*92, num_classes)
        else:
            self.fc = nn.Linear(24*194, num_classes)
        
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        shortcut1 = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = shortcut1 + x
        x = self.relu(self.bnr1(x))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        shortcut2 = x
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = shortcut2 + x
        x = self.relu(self.bnr2(x))
        # print(x.size())
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
            

        