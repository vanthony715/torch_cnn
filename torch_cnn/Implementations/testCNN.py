#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avasquez

Summary:
"""

import os
import sys
import time
import argparse

sys.path.append('/home/domain/avasquez/ML/APIs/')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
# from Torch_API.imageFolderWithPaths import ImageFolderWithPaths 
from Torch_API.simplePytorchCNN import avNet
import torchvision.datasets as dset

sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
plt.rcParams["figure.figsize"] = (10,8)

##Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, default = '/mnt/opsdata/neurocondor/datasets/avasquez/data/Neuro/MWIRN/RIPS/cyclegan/training/synthetic_classifier/data/gan_inverse_real_transfer/',
                    help="path to large images")

parser.add_argument("--test_dir", type=str, default = 'test/',
                    help="path to test data")

parser.add_argument("--out_dir", type=str, default = 'output/', 
                    help="where to write output such as plots, stats, etc.")

parser.add_argument("--weights_name", type=str, default = 'gan_inverse_real_transfer_15.pth', 
                    help="where to write output such as weights, checkpoints, etc.")

parser.add_argument("--x_dim", type=int, default = 32, 
                    help="image dimension")

parser.add_argument("--num_workers", type=int, default = 4, 
                    help="number of processes")

parser.add_argument("--cuda", type=bool, default = True, 
                    help="enable cuda")

parser.add_argument("--cudnn_benchmark", type=bool, default = True, 
                    help="enable cudnn bench_mark")

parser.add_argument("--probability_threshold", type=float, default = 0.85, 
                    help="probability threshold")

parser.add_argument("--write_results", type=bool, default = False, 
                    help="Write results to file")

parser.add_argument("--write_cm_plot", type=bool, default = True, 
                    help="Output a Confusion Matrix Plot?")

parser.add_argument("--write_roc", type=bool, default = False, 
                    help="Output a ROC Plot?")

parser.add_argument("--results_file", type=str, default = './results.txt', 
                    help="Where to write results")

args = parser.parse_args()

print('\n ************************ Job Description ************************')
print('Job Name: Simple Train Classifier')
print('Common P: ', args.basepath)
print('Validation Folder: ', args.test_dir)
print('Output Folder: ', args.out_dir)
print('Name of Weights: ', args.weights_name)
print('Image Size: ', args.x_dim)
print('Number of Processes: ', args.num_workers)
print('Cuda Enabled: ', args.cuda)
print('Cudnn Benchmark Enabled: ', args.cudnn_benchmark)
print('Probability Threshold: ', args.probability_threshold)
print('Write Results: ', args.write_results)
print('Results File: ', args.results_file)
print(' *****************************************************************')

##cuda block
CUDA = args.cuda and torch.cuda.is_available()
print('Pytorch Version: {}'.format(torch.__version__))
if args.cuda:
    print('CUDA version: {}'.format(torch.version.cuda))
cudnn.benchmark = args.cudnn_benchmark
device = torch.device("cuda:0" if args.cuda else "cpu")

##implementation
if __name__ == "__main__":
    ##start clock
    t0 = time.time()
    
    ##define paths
    testpath = os.path.join(args.basepath, args.test_dir)
    outpath = os.path.join(args.basepath, args.out_dir)
    weights = os.path.join(outpath, args.weights_name)
    
    ##get data from directories
    testSet = dset.ImageFolder(root=testpath,
                                transform=transforms.Compose([
                                                            transforms.Resize(args.x_dim),
                                                            transforms.CenterCrop(args.x_dim),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),]))
    
    ##batch size the size of the number of files in the test directory
    class_directories = [i for i in os.listdir(testpath)]
    class_cnt_dict = {'class': [], 'class_cnt': []}
    for class_directory in class_directories:
        class_cnt_dict['class'].append(class_directory)
        class_cnt_dict['class_cnt'].append(len(os.listdir(testpath + class_directory)))
        
    BATCH_SIZE = 1
    
    ##define data loader
    assert testSet
    testloader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, 
                                             shuffle=True, num_workers=args.num_workers)
    ##instantiate iterator
    dataiter = iter(testloader)
    inputs, label = dataiter.next()
    
    net = avNet(len(class_cnt_dict['class']))
    net.load_state_dict(torch.load(weights))
    net.eval()
    
    ##define classes
    classes = sorted(class_cnt_dict['class'])
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in tqdm(testloader, desc='Testing One at a Time', colour='green'):
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("\nAccuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))
        
    stats_dict = {'class: '}
    ##get confusion matrix
    cm = confusion_matrix(classes, classes)
    
    ##plot confusion matrix
    df_cm = pd.DataFrame(cm, columns=list(correct_pred.values()), index=list(correct_pred.keys()))
    plt.figure(0)
    plt.title(args.weights_name.split('.pth', 1)[0] + ' CM')
    
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    
    if args.write_cm_plot:
        ##saves as png by default
        plt.savefig(args.basepath + args.weights_name.split('.pth', 1)[0] + '-CM')
    
   
        
    # print('\n******************** Results ********************')
    # def calculateMetrics():
        
    #     correct_cnt_0 = 0
    #     correct_cnt_1 = 0
        
    #     for i in range(len(statsDict['filename'])):
    #         if statsDict['predicted_label'][i] == statsDict['truth_label'][i]:
    #             if statsDict['truth_label'][i] == 0:
    #                 correct_cnt_0 += 1
    #             else:
    #                 correct_cnt_1 += 1
        
    #     incorrect_cnt_0 = class_cnt_dict['class_cnt'][0] - correct_cnt_0
    #     incorrect_cnt_1 = class_cnt_dict['class_cnt'][1] - correct_cnt_1
        
    #     accuracy_0 = (1 - (incorrect_cnt_0 / correct_cnt_0))*100
    #     accuracy_1 = (1 - (incorrect_cnt_1 / correct_cnt_1))*100
    #     print('correct_cnt_0: ', correct_cnt_0)
    #     print('correct_cnt_1: ', correct_cnt_1)
    #     print('incorrect_cnt_0: ', incorrect_cnt_0)
    #     print('incorrect_cnt_1: ', incorrect_cnt_1)
    #     print('Model: ', weights)
    #     print('Class: ', class_cnt_dict['class'][0], 'Accuracy:', accuracy_0)
    #     print('Class: ', class_cnt_dict['class'][1], 'Accuracy:', accuracy_1)
        
    # calculateMetrics()
        

    ##end clock
    t1 = time.time()
    ##print statistics
    print('\n--------------------------- Stats --------------------------------')
    print('Number of files tested: ', BATCH_SIZE)
    print('Classes Tested: ', class_cnt_dict['class'])
    print('Execution Time: ', round((t1 - t0), 5))