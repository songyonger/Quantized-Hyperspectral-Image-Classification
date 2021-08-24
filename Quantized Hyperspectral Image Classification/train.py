import time
import torch
import numpy as np
from utils import AverageMeter, accuracy, count_correct_perclass


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None, index=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    oa = AverageMeter()

    # switch to train mode
    model.train()

    num = len(train_loader)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        model.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.long().cuda()
        input_var = input.cuda()
        input_var = input_var.unsqueeze(1)
        target_var = target
        # print("input_var size:",input_var.size())

        # compute output
        output = model(input_var)
        if index is not None:
            output = output[index]
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            oa.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            step = epoch * num + i
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'OA {oa.val:.3f} ({oa.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, oa=oa))
                if writer is not None:
                    writer.add_scalar('loss', losses.val, step)
                    writer.add_scalar('train_oa_val', oa.val, step)

    if writer is not None:
        writer.add_scalar('train_oa_avg', oa.avg, step)


def test(test_loader, model, criterion, args, epoch=-1, writer=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    oa = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct_per_class = np.zeros(args.num_classes)
    sample_per_class = np.zeros(args.num_classes)
    
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.long().cuda()
            input_var = input.cuda()
            input_var = input_var.unsqueeze(1)
            target_var = target
            # print("input_var size:", input_var.size())

            # compute output
            output = model(input_var)
            # print(output.size())
            # print(target_var.size())
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            
            count_correct_perclass(output, target, correct_per_class, sample_per_class)
            
            losses.update(loss.item(), input.size(0))
            oa.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (args.print_freq *10) == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'OA {oa.val:.3f} ({oa.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses,
                          oa=oa))
    
    print("correct_per_class:", correct_per_class)
    print("sample_per_class:", sample_per_class)    
    oa_per_class = correct_per_class / sample_per_class * 100
    aa = float(np.mean(oa_per_class))
    
    print(' * OA {oa.avg:.3f}\n'.format(oa=oa))
    print(' * AA {aa:.3f}\n'.format(aa=aa))
    if writer is not None:
        writer.add_scalar('test_oa_avg', oa.avg, epoch)
        writer.add_scalar('test_aa', aa, epoch)

    return oa.avg, aa