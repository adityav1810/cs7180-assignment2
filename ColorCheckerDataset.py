"""
File: ColorCheckerDataset.py

Description: 
    Custom Dataloader class to load and prepare images for neural net.
    
Authors:
    Author 1 (Aditya Varshney,varshney.ad@northeastern.edu, Northeastern University)
    Author 2 (Luv Verma, verma.lu@northeastern.edu , Northeastern University)

Citations and References:
    - Reference 1: https://github.com/matteo-rizzo/fc4-pytorch
    
"""

import os
from typing import Tuple
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from Augmenter import Augment

NUM_IMAGES = 100

class ColorCheckerDataset(data.Dataset):
    def __init__(self, train: bool = True, folds_num: int = 1):
        self.__train = train
        self.__da = Augment()
        path_to_folds = os.path.join("dataset", "folds.mat")
        path_to_metadata = os.path.join("dataset", "metadata.txt")
        self.__path_to_data = os.path.join("dataset", "preprocessed", "numpy_data")
        self.__path_to_label = os.path.join("dataset", "preprocessed", "numpy_labels")
        folds = scipy.io.loadmat(path_to_folds)
        img_idx = [i for i in folds["tr_split" if self.__train else "te_split"][0][folds_num][0] if i< NUM_IMAGES]
        print(img_idx)
        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='float32')
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')
        if self.__train:
            img, illuminant = self.__da.augment(img, illuminant)
        else:
            img = self.__da.crop(img)
        img = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(img))))
        img = torch.from_numpy(img.copy())
        illuminant = torch.from_numpy(illuminant.copy())
        if not self.__train:
            img = img.type(torch.FloatTensor)
        return img, illuminant, file_name

    def __len__(self) -> int:
        return len(self.__fold_data)
