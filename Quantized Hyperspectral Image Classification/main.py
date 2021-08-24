
import argparse
import os
import shutil
import time
import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as udata
from tensorboardX import SummaryWriter
from models.model_dorefa_1w2a import SAWB
# from models.model_bireal_1w1a import SAWB
# from models.model_react_1w1a import SAWB
# from models.model_irnet_1w1a import SAWB
# from models.model_recu_1w1a import SAWB
# from models.model_fp import SAWB
from train import train, test
from dataset import HSIDataset


parser = argparse.ArgumentParser(description='Net for HSI classification in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--dataset_name', default='pavia', type=str, help='dataset name')
parser.add_argument('--num_classes', default=9, type=int, metavar='N', help='number of classes')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--milestones', default=[100,150,200], type=list, help='when the lr decay')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--tau_min', default=0.85, type=float, metavar='M', help='tau_min')
parser.add_argument('--tau_max', default=0.99, type=float, metavar='M', help='tau_max')
parser.add_argument('--act_std', default='1', type=str, help='whether activation std')
parser.add_argument('--weight_mean', default='0', type=str, help='whether weight minus mean')
parser.add_argument('--perchannel_clamp', default='0', type=str, help='whether clamp per channelly or not')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4) recu condition is 1e-4')
parser.add_argument('--print_freq', default=5, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--gpus', default='0', type=str, help='which gpus to run')


best_oa = 0
corresponding_aa = 0    
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
def main():
    global args, best_oa, corresponding_aa
    
    # ************************* condition for recu *****************************************
    # model = SAWB(args, num_classes = args.num_classes, dataset_name = args.dataset_name)
    # **************************************************************************************
    
    model = SAWB(num_classes = args.num_classes, dataset_name = args.dataset_name)
    model.cuda()
    args.save_dir = 'logs_{}/{}_{}_{}_{}_{}_{}'.format(\
        args.dataset_name, model.name, args.lr, args.weight_decay, args.batch_size, args.epochs, args.milestones)
    print(args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(args.save_dir)
    
    trainDataset = HSIDataset(dataset_name=args.dataset_name, mode='train')
    testDataset = HSIDataset(dataset_name=args.dataset_name, mode='test')
    train_loader = udata.DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = udata.DataLoader(testDataset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
        
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                weight_decay=args.weight_decay, eps=1e-8, betas=(0.9, 0.999))
                                
    # ****************************** condition for recu *****************************************                           
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay)
    # *******************************************************************************************
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_oa = checkpoint['best_oa']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    for i in range(args.start_epoch):
        lr_scheduler.step()
        
    test(test_loader, model, criterion, args, -1, writer)
    
    # *********************************** condition for irnet **************************************************
    # T_min, T_max = 1e-3, 1e1

    # def Log_UP(K_min, K_max, epoch):
        # Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        # return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()
    # **********************************************************************************************************

    # *********************************** condition for recu ***************************************************
    # def cpt_tau(epoch):
        # "compute tau"
        # a = torch.tensor(np.e)
        # T_min, T_max = torch.tensor(args.tau_min).float(), torch.tensor(args.tau_max).float()
        # A = (T_max - T_min) / (a - 1)
        # B = T_min - A
        # tau = A * torch.tensor([torch.pow(a, epoch/args.epochs)]).float() + B
        # return tau
    # **********************************************************************************************************
    
    for epoch in range(args.start_epoch, args.epochs):
        # *************************** condition for irnet ******************************************
        # t = Log_UP(T_min, T_max, epoch)
        # if (t < 1):
            # k = 1 / t
        # else:
            # k = torch.tensor([1]).float().cuda()
        
        # model.conv2.k = k
        # model.conv2.t = t
        # model.conv3.k = k
        # model.conv3.t = t
        # model.conv4.k = k
        # model.conv4.t = t
        # model.conv5.k = k
        # model.conv5.t = t
        # model.conv6.k = k
        # model.conv6.t = t            
        # model.conv7.k = k
        # model.conv7.t = t
        # model.conv8.k = k
        # model.conv8.t = t
        # ******************************************************************************************
        
        # ************************* condition for recu *********************************************
        # t = cpt_tau(epoch)
        # t = t.cuda()
        # model.conv2.tau = t
        # model.conv3.tau = t
        # model.conv4.tau = t
        # model.conv5.tau = t
        # model.conv6.tau = t
        # model.conv7.tau = t
        # model.conv8.tau = t
        # ******************************************************************************************
        
        print(args.save_dir)
        print('current lr {:.5e}, using gpu : {}'.format(optimizer.param_groups[0]['lr'], args.gpus))
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        torch.cuda.empty_cache()
        lr_scheduler.step()

        oa,aa = test(test_loader, model, criterion, args, epoch, writer)
        is_best = oa > best_oa
        best_oa = max(oa, best_oa)

        if is_best:
            model.zero_grad()
            corresponding_aa = aa
            torch.save({
                'epoch': epoch,
                'optim_state_dict': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_oa': best_oa,
                'co_aa': corresponding_aa,
            },os.path.join(args.save_dir,'best_model.pth'))
        
        torch.save({
            'epoch': epoch,
            'optim_state_dict': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'best_oa': best_oa,
            'co_aa': corresponding_aa,
            'oa': oa,
            'aa': aa,
            },os.path.join(args.save_dir,'latest_model.pth'))

        torch.cuda.empty_cache()

    print('final best OA:{}\t corresponding AA:{}'.format(best_oa, corresponding_aa))
    with open(os.path.join(args.save_dir, 'best_result.txt'), 'w') as f:
        f.write('final best OA{}\t corresponding AA{}\n'.format(best_oa, corresponding_aa))


if __name__ == '__main__':
    main()
