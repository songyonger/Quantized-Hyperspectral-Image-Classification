import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

class BinaryQuantize_A(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input>1.0, 0.0)
        grad_input.masked_fill_(input<1.0, 0.0)
        mask_pos = (input>=0.0) & (input<1.0)
        mask_neg = (input<0.0) & (input>=-1.0)
        grad_input.masked_scatter_(mask_pos, input[mask_pos].mul_(-2.0).add_(2.0))
        grad_input.masked_scatter_(mask_neg, input[mask_neg].mul_(2.0).add_(2.0))
        return grad_input * grad_output

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
        
class HardBinaryConv(nn.Conv1d):

    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(HardBinaryConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0),1), requires_grad=True)
        self.tau = torch.tensor([1.0]).float().cuda()
        self.weight_mean = args.weight_mean
        self.act_std = args.act_std
        self.perchannel_clamp = args.perchannel_clamp

    def forward(self, input):
        w = self.weight
        a = input
        if self.weight_mean == '0':
            w0 = w
        else:
            w0 = w - w.mean([1,2], keepdim=True)
        
        # 该做法进行归一化时，减均值和除以标准差都是在每个输出通道上单独做
        w1 = w0 / (torch.sqrt(w0.var([1,2], keepdim=True) + 1e-5) / 2 / np.sqrt(2))

        # 计算b的最大似然估计以及后面的clamp操作都是所有输出通道上是同一个值
        if self.perchannel_clamp == '0':
            EW = torch.mean(torch.abs(w1))
            Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
            w2 = torch.clamp(w1, -Q_tau, Q_tau)
        else:
            # 计算b的最大似然估计以及后面的clamp操作都是逐输出通道进行的
            EW = torch.abs(w1).mean([1,2], keepdim=True)
            w1 = w1.detach().cpu().numpy()
            Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().numpy()
            w2 = np.clip(w1, -Q_tau, Q_tau)
            w2 = torch.from_numpy(w2).cuda()            


        if self.act_std == '1':
            if self.training:
                a0 = a / torch.sqrt(a.var([1,2], keepdim=True) + 1e-5)
            else: 
                a0 = a
        else:
            a0 = a
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_A().apply(a0)

        # 1bit conv
        output = F.conv1d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        # scaling factor
        output = output * self.alpha 
        return output

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

class SAWB(nn.Module):
    def __init__(self, args, num_classes=9, dataset_name='pavia'):
        super(SAWB, self).__init__()
        self.name = 'SAWB-C-ReCU-Net'
        self.conv1 = nn.Conv1d(1, 24, 11)
        self.bn1 = nn.BatchNorm1d(24)
        # self.saq1 = QuantActivation()
        self.conv2 = HardBinaryConv(args, 24, 24, 11, 1, 5)
        self.bn2 = nn.BatchNorm1d(24)
        # self.saq2 = QuantActivation()
        self.conv3 = HardBinaryConv(args, 24, 24, 11, 1, 5)
        self.bn3 = nn.BatchNorm1d(24)
        # self.saq3 = QuantActivation()
        self.bnr1 = nn.BatchNorm1d(24)
        # self.saqr1 = QuantActivation()
        self.conv4 = HardBinaryConv(args, 24, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm1d(128)
        # self.saq4 = QuantActivation()
        self.conv5 = HardBinaryConv(args, 128, 24, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(24)
        # self.saq5 = QuantActivation()
        self.conv6 = HardBinaryConv(args, 24, 24, 3, 1, 1)
        self.bn6 = nn.BatchNorm1d(24)
        # self.saq6 = QuantActivation()
        self.conv7 = HardBinaryConv(args, 24, 24, 3, 1, 1)
        self.bn7 = nn.BatchNorm1d(24)
        # self.saq7 = QuantActivation()
        self.conv8 = HardBinaryConv(args, 24, 24, 3, 1, 1)
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