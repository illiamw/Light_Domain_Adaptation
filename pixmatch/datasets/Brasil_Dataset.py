# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import os
import torch
from datasets.cityscapes_Dataset import City_Dataset, to_tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


class Brasil_Dataset(City_Dataset):
    def __init__(
        self,
        root='./datasets/GTA5',
        list_path='./datasets/GTA5/list',
        split='train',
        base_size=769,
        crop_size=769,
        training=True,
        random_mirror=False,
        random_crop=False,
        resize=False,
        gaussian_blur=False,
        class_16=False,
        class_13=False
    ):

        # Args
        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training
        self.class_16 = False
        self.class_13 = False
        assert class_16 == False

        # Augmentation
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.resize = resize
        self.gaussian_blur = gaussian_blur

        # Files
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval/test/all")
        self.image_filepath = self.data_path
        self.items = [id.strip() for id in open(item_list_filepath)]

        # Label map
        self.id_to_trainid = {
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

        # Print
        print("{} num images in Brazil {} set have been loaded.".format(
            len(self.items), self.split))

    def __getitem__(self, item):
        id = self.items[item]
        name = f"{id}.jpg"

        # Open image and label
        image_path = os.path.join(self.image_filepath, self.split, name.split('_')[0])
        
        image_path = os.path.join(image_path, name)
        image = Image.open(image_path).convert("RGB")
        gt_image = Image.new("L", (image.size[0], image.size[1]), (-1))
        
        # Augmentation
        if (self.split == "train" or self.split == "trainval" or self.split == "all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)
        return image, gt_image, item
