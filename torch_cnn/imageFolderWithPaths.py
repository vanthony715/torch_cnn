#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avasquez
"""
import torchvision.datasets as dset
class ImageFolderWithPaths(dset.ImageFolder):
    """dataset from folder that included the filename and filepath"""
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path