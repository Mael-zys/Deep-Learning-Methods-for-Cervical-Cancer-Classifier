import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data
import shutil
from test_data_loader_seg import testLoader
import models
import util

def test(args):
    data_loader = testLoader(image_size=args.image_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)


    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=9)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=9)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=9)
    
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0

    save_path = args.save

    util.io.write_lines(save_path, "ID"+","+"GROUP"+'\n', 'w')
    for idx, img in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        img = Variable(img.cuda(), volatile=True)
        
        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)

        outputs = outputs.squeeze(0).cpu().numpy().astype(np.float32)
        result = np.argmax(outputs)

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        image_name = data_loader.img_paths[idx][0].split('/')[-1].split('.')[0]
        util.io.write_lines(save_path, image_name+","+str(result)+'\n', 'a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default='checkpoints_multi/ic15_resnet50_bs_96_ep_30/checkpoint.pth.tar',    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--image_size', nargs='?', type=int, default=256)
    parser.add_argument('--save', nargs='?', type=str, default='./submission_multi_seg.csv',    
                        help='Path to save results')
    args = parser.parse_args()
    test(args)
