import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Variable
from torch.utils import data
import os

from train_data_loader_multiclass_seg import trainLoader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util
from torch.utils.data import random_split
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def train_loss_plot(train_loss_list=[], val_loss_list=[], name=''):
    x1 = range(0, len(train_loss_list))
    x2 = range(0, len(val_loss_list))
    y1 = train_loss_list
    y2 = val_loss_list
    plt.switch_backend('agg')
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('loss vs. iterators')
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'o-')
    plt.ylabel('val loss')
    plt.savefig(name)

LossCL = nn.CrossEntropyLoss().cuda()

def train(train_loader, val_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    val_losses = AverageMeter()
    val_batch_time = AverageMeter()
    val_data_time = AverageMeter()
    
    end = time.time()
    for batch_idx, (imgs, gt) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs = Variable(imgs.cuda())
        gt = Variable(gt.cuda())

        
        outputs = model(imgs)
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, gt)
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg)
            print(output_log)
            sys.stdout.flush()
    
    #validation
    with torch.no_grad():
        end = time.time()
        for batch_idx, (imgs, gt) in enumerate(val_loader):
            val_data_time.update(time.time() - end)

            imgs = Variable(imgs.cuda())
            gt = Variable(gt.cuda())

            outputs = model(imgs)
            # outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, gt)
            val_losses.update(loss.item(), imgs.size(0))

            val_batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 5 == 0:
                output_log  = 'Validation: ({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    bt=val_batch_time.avg,
                    total=val_batch_time.avg * batch_idx / 60.0,
                    eta=val_batch_time.avg * (len(val_loader) - batch_idx) / 60.0,
                    loss=val_losses.avg)
                print(output_log)
                sys.stdout.flush()

    return losses.avg, val_losses.avg

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints_multi_seg/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print ('checkpoint path: %s'%args.checkpoint)
    print ('init lr: %.8f'%args.lr)
    print ('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    start_epoch = 0

    data_loader = trainLoader(is_transform=True, img_size=args.img_size)
    

    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=9)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=9)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=9)
    
    model = torch.nn.DataParallel(model).cuda()
    
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)

    title = 'ima205_challenge1'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss'])


    # train
    train_loss_list = []
    val_loss_list = []
    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))


        train_size = int(0.7 * len(data_loader))
        val_size = len(data_loader) - train_size
        train_loader, val_loader = random_split(data_loader, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(
            train_loader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=3,
            drop_last=False,
            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            val_loader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=3,
            drop_last=False,
            pin_memory=True)
            
        train_loss, val_loss = train(train_loader, val_loader, model, LossCL, optimizer, epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_loss_plot(train_loss_list,val_loss_list,'train_val_loss_multi_seg'+args.arch+'.png')
        
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

        logger.append([optimizer.param_groups[0]['lr'], train_loss])
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=30, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=96, 
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    main(args)
