import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

class BinaryQuantize(Function):
	@staticmethod
	def forward(ctx, x, k, t):
		ctx.save_for_backward(x, k, t)
		out = torch.sign(x)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		x, k, t = ctx.saved_tensors
		grad_input = k * t * (1 - torch.pow(torch.tanh(x * t), 2)) * grad_output
		return grad_input, None, None

class HardBinaryConv(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(HardBinaryConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv1d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

class SAWB(nn.Module):
    def __init__(self, num_classes=9, dataset_name='pavia'):
        super(SAWB, self).__init__()
        self.name = 'SAWB-C-IRNet'
        self.conv1 = nn.Conv1d(1, 24, 11)
        self.bn1 = nn.BatchNorm1d(24)
        # self.saq1 = QuantActivation()
        self.conv2 = HardBinaryConv(24, 24, 11, 1, 5)
        self.bn2 = nn.BatchNorm1d(24)
        # self.saq2 = QuantActivation()
        self.conv3 = HardBinaryConv(24, 24, 11, 1, 5)
        self.bn3 = nn.BatchNorm1d(24)
        # self.saq3 = QuantActivation()
        self.bnr1 = nn.BatchNorm1d(24)
        # self.saqr1 = QuantActivation()
        self.conv4 = HardBinaryConv(24, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm1d(128)
        # self.saq4 = QuantActivation()
        self.conv5 = HardBinaryConv(128, 24, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(24)
        # self.saq5 = QuantActivation()
        self.conv6 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn6 = nn.BatchNorm1d(24)
        # self.saq6 = QuantActivation()
        self.conv7 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn7 = nn.BatchNorm1d(24)
        # self.saq7 = QuantActivation()
        self.conv8 = HardBinaryConv(24, 24, 3, 1, 1)
        self.bn8 = nn.BatchNorm1d(24)
        # self.saq8 = QuantActivation()        
        self.bnr2 = nn.BatchNorm1d(24)
        # self.saqr2 = QuantActivation()
        self.dropout = nn.Dropout(0.5)
        if dataset_name == 'pavia':
            self.fc = nn.Linear(24*92, num_classes)
        else:
            self.fc = nn.Linear(24*194, num_classes)
        
        self.apply(_weights_init)
    
    def forward(self, x):
        x = (self.bn1(self.conv1(x)))
        shortcut1 = x
        x = (self.bn2(self.conv2(x)))
        x = (self.bn3(self.conv3(x)))
        x = shortcut1 + x
        x = (self.bnr1(x))
        x = (self.bn4(self.conv4(x)))
        x = (self.bn5(self.conv5(x)))
        shortcut2 = x
        x = (self.bn6(self.conv6(x)))
        x = (self.bn7(self.conv7(x)))
        x = (self.bn8(self.conv8(x)))
        x = shortcut2 + x
        x = (self.bnr2(x))
        # print(x.size())
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
            

        