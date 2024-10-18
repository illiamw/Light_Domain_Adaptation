# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import os
import torch
from dataloaders.cityscapes_Dataset import CityScape
import numpy as np
import random, math
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES = 3

ignore_label = -1

class GTA5_Dataset(CityScape):
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

         # Label map
        self.id_to_trainid = {
            -1: ignore_label,
            0: ignore_label,
            1: ignore_label,
            2: ignore_label,
            3: ignore_label,
            4: 2,
            5: 2,
            6: 1,
            7: 0,
            8: 1,
            9: 1,
            10: 1,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            18: 2,
            19: 2,
            20: 2,
            21: 2,
            22: 1,
            23: 1,
            24: 2,
            25: 2,
            26: 2,
            27: 2,
            28: 2,
            29: 2,
            30: 2,
            31: 2,
            32: 2,
            33: 2
        }

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

        self.images_dir = os.path.join(self.root, 'images')
        self.targets_dir = os.path.join(self.root, 'labels')
        self.split = split
        self.images = []
        self.targets = []

        if self.split  == 'train':
            items = [id.strip() for id in open('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/datasets/GTA5Dataset/gta5_list/train.txt')]
        elif self.split == 'test':
            items = [id.strip() for id in open('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/datasets/GTA5Dataset/gta5_list/test.txt')]
        elif self.split == 'val':
            items = [id.strip() for id in open('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/datasets/GTA5Dataset/gta5_list/val.txt')]

        for file_name in items:
            id = int(file_name)
            file_name = f"{id:0>5d}.png"
            target_name = file_name
            self.targets.append(os.path.join(self.targets_dir, target_name))
            self.images.append(os.path.join(self.images_dir, file_name))

        #balanciador de carga ----------- HardCoder --------------------
        if self.split == 'train':

            aux = []
            aux2 = []
            tam = len(self.images)
            j = 0
            for i in range(self.n_labeled_examples):
                
                aux.append(self.images[j])
                aux2.append(self.targets[j])
                j+=1
                if j == tam:
                    j=0
            self.images = aux
            self.targets = aux2
            
        print(f"{self.split}: {len(self.images), {len(self.targets)}}")
    
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = np.asarray(image, dtype=np.float32)
        target = Image.open(self.targets[index])  # type: ignore[assignment]

        label = target
        label = np.asarray(label, dtype=np.int32)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = self._mask_transform(label)
        return image, label


class GTA5DataLoader(DataLoader):
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

        self.dataset = GTA5_Dataset(**kwargs)

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
        

        super(GTA5DataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)
