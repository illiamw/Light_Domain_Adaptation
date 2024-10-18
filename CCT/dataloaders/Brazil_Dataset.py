# -*- coding: utf-8 -*-

import random, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from scipy import ndimage
from math import ceil
import os
from torch.utils.data import DataLoader

from dataloaders.cityscapes_Dataset import CityScape


ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES = 3

class Brazil_Dataset(CityScape):
    def __init__(self, data_dir, split, mean, std, ignore_index, base_size=None, augment=True, val=False,
                jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False, flip=False, rotate=False,
                blur=False, return_id=False, n_labeled_examples=None):
        self.num_classes = NUM_CLASSES
        self.root = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.jitter = jitter
        self.image_padding = (np.array(mean)*255.).tolist()
        self.ignore_index = ignore_index
        self.return_id = return_id
        self.n_labeled_examples = n_labeled_examples
        self.val = val

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)


        if self.split  == 'train_unsupervised':
            self._set_files(self.root,'train')
        elif self.split == 'train_supervised':
            self._set_files(self.root,'train')
        elif self.split == 'val':
            self._set_files(self.root,'val')


        cv2.setNumThreads(0)

    def _set_files(self,
        root,
        split= "train"):
        self.root = root

        self.images_dir = os.path.join(self.root, split)
        self.targets_dir = os.path.join(self.root, split)
        self.split = split
        self.images = []
        self.targets = []

        for city in os.listdir(self.images_dir):
            # if city != "RodoviasNordesteSul": # remove rodovias
            for file_name in os.listdir(os.path.join(self.images_dir,city)):
                self.images.append(os.path.join(self.images_dir,city , file_name))

        
        print(f"{self.split}: {len(self.images), {len(self.targets)}}")
        
    def _mask_transform(self, gt_image):
        target = torch.from_numpy(gt_image)

        return target
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = np.asarray(image, dtype=np.float32)
        #target falso
        target = np.full(( image.shape[0], image.shape[1]), -1, dtype=np.int8)
        label = target
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = self._mask_transform(label)
        return image, label

class CityscapeBrazilDataLoader(DataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = -1
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = Brazil_Dataset(**kwargs)

        self.shuffle = shuffle
        self.nbr_examples = len(self.dataset)
        val_split = None
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': False
        }
        

        super(CityscapeBrazilDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)
