#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:23:39 2021

@author: avasquez
"""
import os
import shutil
import cv2
import torch
from tqdm import tqdm
import torchvision.utils as vutils
import xml.etree.ElementTree as ET

def clearFolder(Path):
    if os.path.isdir(Path):
        print('Removing File: ', Path)
        shutil.rmtree(Path)
    print('Creating File: ', Path)
    os.mkdir(Path)
    
##gets chip coords from XML file
def getChip(PredFile):
    ##store bounding boxes
    bbDict = {
        'Name' : [], 'Xmin' : [], 'Ymin' : [], 'Xmax' : [], 'Ymax' : [],
        'Annotation': [], 'Chip': [], 'Score': []} ## end dict 
    tree = ET.parse(PredFile)
    root = tree.getroot()
    ##iterate through xml tree 
    try:
        for elem in root.iter('name'):
            if elem.text != 'SRC INC':
                name = elem.text
                bbDict['Name'].append(name)
        for elem in root.iter('xmin'): 
            xmin = elem.text
            bbDict['Xmin'].append(xmin)
            bbDict['Annotation'].append(PredFile)
        for elem in root.iter('ymin'): 
            ymin = elem.text
            bbDict['Ymin'].append(ymin)
        for elem in root.iter('xmax'): 
            xmax = elem.text
            bbDict['Xmax'].append(xmax)
        for elem in root.iter('ymax'): 
            ymax = elem.text 
            bbDict['Ymax'].append(ymax)
        # for elem in root.iter('confidence'): 
        #     ymax = elem.text 
        #     bbDict['Confidence'].append(ymax)
    except:
        print('Error in in Functions.py')
    return bbDict
        
def writeChip(Name, Xmin, Ymin, Xmax, Ymax, LargeImageFile, ChipWritePath):
    pre, _ = os.path.splitext(LargeImageFile)
    image = cv2.imread(LargeImageFile)
    deltaX = int(Xmax) - int(Xmin)
    deltaY = int(Ymax) - int(Ymin)
    imageChip = image[int(Ymin) : int(Ymin) + deltaY, int(Xmin) : int(Xmin) + deltaX]
    writeName = '_' + Name + '_' + Xmin + '_' + Ymin + '_' + Xmax + '_' + Ymax + '.png'
    cv2.imwrite(ChipWritePath + writeName, imageChip)
    return (ChipWritePath + writeName)

def renamePredictions(Annotation, Xmin, Ymin, Xmax, Ymax, NewName, ClassNames, WritePath, FLAG):
    ##store bounding boxes
    tree = ET.parse(Annotation)
    root = tree.getroot()
    ##iterate through xml tree 
    try:
        for name, xmin, ymin, xmax, ymax in zip(root.iter('name'), root.iter('xmin'), root.iter('ymin'), root.iter('xmax'), root.iter('ymax')):

            if name.text in ClassNames:
                if int(xmin.text) == int(Xmin) and int(ymin.text) == int(Ymin) and int(xmax.text) == int(Xmax) and int(ymax.text) == int(Ymax):
                    if FLAG:
                        name.text = NewName
                        with open(WritePath,"wb") as f:
                            tree.write(f)
                    else:
                        with open(WritePath,"wb") as f:
                            tree.write(f)
    except:
            print('Error in in Functions.py')


    