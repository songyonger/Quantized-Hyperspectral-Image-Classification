import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# A(特征)量化
class QuantActivation(nn.Module):
    def __init__(self, a_bits):
        super(QuantActivation, self).__init__()
        self.a_bits = a_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        # output = torch.clamp(input * 0.1, 0, 1)      # 特征A截断前先进行缩放（* 0.1），以减小截断误差
        output  = torch.clamp(input, 0, 1)
        scale = 1 / float(2 ** self.a_bits - 1)      # scale
        output = self.round(output / scale) * scale  # 量化/反量化
        return output

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride=1, padding=0):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size
        self.shape = (out_chn, in_chn, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(abs(real_weights), dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach()-cliped_weights.detach()+cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

class SAWB(nn.Module):
    def __init__(self, num_classes=9, dataset_name='pavia'):
        super(SAWB, self).__init__()
        self.name = 'SAWB-C-dorefa_1w2a'
        self.conv1 = nn.Conv1d(1, 24, 11)
        self.bn1 = nn.BatchNorm1d(24)
        self.saq1 = QuantActivation(2)
        self.conv2 = HardBinaryConv(24, 24, 11, 1, 5)
        self.bn2 = nn.BatchNorm1d(24)
        self.saq2 = QuantActivation(2)
        self.conv3 = HardBinaryConv(24, 24, 11, 1, 5)
        self.bn3 = nn.BatchNorm1d(24)
        self.saq3 = QuantActivation(2)
        self.bnr1 = nn.BatchNorm1d(24)
        self.saqr1 = QuantActivation(2)
        self.conv4 = HardBinaryConv(24, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm1d(128)
        self.saq4 = QuantActivation(2)
        self.conv5 = HardBinaryConv(128, 24, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(24)
        self.saq5 = QuantActivation(2)
        self.conv6 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn6 = nn.BatchNorm1d(24)
        self.saq6 = QuantActivation(2)
        self.conv7 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn7 = nn.BatchNorm1d(24)
        self.saq7 = QuantActivation(2)
        self.conv8 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn8 = nn.BatchNorm1d(24)
        self.saq8 = QuantActivation(2)        
        self.bnr2 = nn.BatchNorm1d(24)
        self.saqr2 = QuantActivation(2)
        self.dropout = nn.Dropout(0.5)
        if dataset_name == 'pavia':
            self.fc = nn.Linear(24*92, num_classes)
        else:
            self.fc = nn.Linear(24*194, num_classes)
        
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.saq1(self.bn1(self.conv1(x)))
        shortcut1 = x
        x = self.saq2(self.bn2(self.conv2(x)))
        x = self.saq3(self.bn3(self.conv3(x)))
        x = shortcut1 + x
        x = self.saqr1(self.bnr1(x))
        x = self.saq4(self.bn4(self.conv4(x)))
        x = self.saq5(self.bn5(self.conv5(x)))
        shortcut2 = x
        x = self.saq6(self.bn6(self.conv6(x)))
        x = self.saq7(self.bn7(self.conv7(x)))
        x = self.saq8(self.bn8(self.conv8(x)))
        x = shortcut2 + x
        x = self.saqr2(self.bnr2(x))
        # print(x.size())
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
            

        