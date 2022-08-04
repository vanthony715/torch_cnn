#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avasquez
"""

import sys
import os
import shutil

sys.path.append('/home/domain/avasquez/ML/APIs/Torch_API/')

import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable

# from imageFolderWithPaths import ImageFolderWithPaths 
import torchvision.datasets as dset
from simplePytorchCNN import avNet

np.set_printoptions(suppress=True)

##Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, default = '/mnt/opsdata/neurocondor/datasets/avasquez/data/Neuro/MWIRN/RIPS/cyclegan/training/synthetic_classifier/data/gan_inverse_real_transfer/',
                    help="path to large images")

parser.add_argument("--train_dir", type=str, default = 'train/',
                    help="path to train data")

parser.add_argument("--valid_dir", type=str, default = 'valid/',
                    help="path to valid data")

parser.add_argument("--out_dir", type=str, default = 'output/', 
                    help="where to write output such as weights, checkpoints, etc.")

parser.add_argument("--epochs", type=int, default = 15, 
                    help="train epochs")

parser.add_argument("--batch_size", type=int, default = 64, 
                    help="size of train batch")

parser.add_argument("--x_dim", type=int, default = 32, 
                    help="image dimension")

parser.add_argument("--num_workers", type=int, default = 4, 
                    help="number of processes")

parser.add_argument("--cuda", type=bool, default = True, 
                    help="enable cuda")

parser.add_argument("--cudnn_benchmark", type=bool, default = True, 
                    help="enable cudnn bench_mark")

parser.add_argument("--log_interval", type=int, default = 1, 
                    help="enable cuda")

parser.add_argument("--clear_output_dir", type=bool, default = False, 
                    help="Deletes the output directory that has weights and other output")

args = parser.parse_args()

print('\n ************************ Job Description ************************')
print('Job Name: Simple Train Classifier')
print('Common P: ', args.basepath)
print('Train Data Folder: ', args.train_dir)
print('Validation Folder: ', args.valid_dir)
print('Output Folder: ', args.out_dir)
print('Epochs: ', args.epochs)
print('Batch Size: ', args.batch_size)
print('Image Size: ', args.x_dim)
print('Number of Processes: ', args.num_workers)
print('Cuda Enabled: ', args.cuda)
print('Cudnn Benchmark Enabled: ', args.cudnn_benchmark)
print('Log Interval: ', args.log_interval)
print('Delete output directory: ', args.clear_output_dir)
print(' *****************************************************************')

CUDA = args.cuda and torch.cuda.is_available()
print('Pytorch Version: {}'.format(torch.__version__))
if args.cuda:
    print('CUDA version: {}'.format(torch.version.cuda))
cudnn.benchmark = args.cudnn_benchmark
device = torch.device("cuda:0" if args.cuda else "cpu")

if __name__ == "__main__":
    ##start clock
    t0 = time.time()
    
    trainpath = os.path.join(args.basepath, args.train_dir)
    validpath = os.path.join(args.basepath, args.valid_dir)
    outpath = os.path.join(args.basepath, args.out_dir)
    weights = os.path.join(outpath, 'gan_inverse_real_transfer_' + str(args.epochs) + '.pth')
    
    ##Remove/Create output directory
    if os.path.exists(outpath) and args.clear_output_dir:
        print('Removing Directory')
        shutil.rmtree(outpath)
        os.mkdir(outpath)
    elif not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        pass
    
    ##instantiate network
    net = avNet(len(os.listdir(trainpath)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    ##Get data and instantiate transformer
    trainset = dset.ImageFolder(root=trainpath,
                                transform=transforms.Compose([
                                                            transforms.Resize(args.x_dim),
                                                            transforms.CenterCrop(args.x_dim),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),]))
    
    ##create data loader for training                             
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    
    ##train model 
    for epoch in tqdm(range(args.epochs), desc = 'Training Model', colour = 'green'):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels, filename = data
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % args.log_interval == (args.log_interval - 1):    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.6f' %
                #       (epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0
    
    ##save model
    print('\nSaving Weights to: ', weights)
    torch.save(net.state_dict(), weights)
    
    t1 = time.time()
    print('\n----------------Stats---------------------')
    print('Execution Time: ', round((t1 - t0), 5))
