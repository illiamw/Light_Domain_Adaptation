import os, sys
import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import shutil


class testDataset(Dataset):
    def __init__(self, root, list_path, split="test", base="GTA5"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base = base

        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
          raise Warning("split must be train/val/trainval")

        if base  == "Cityscapes":
          self.image_filepath = os.path.join(self.data_path, "leftImg8bit")
        elif base == "GTA5":
          self.image_filepath = os.path.join(self.data_path, "images")
        elif base == "Brazil":
          self.image_filepath = os.path.join(self.data_path, self.split)
        else:
          raise Warning("base must be Cityscapes/GTA5/Brazil")

        self.items = [id.strip() for id in open(item_list_filepath)]



        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.base  == "Cityscapes":
          id = self.items[index]
          filename = id.split("train_")[-1].split("val_")[-1].split("test_")[-1]
          image_filepath = os.path.join(
              self.image_filepath, id.split("_")[0], id.split("_")[1])
          image_filename = filename + "_leftImg8bit.png"
          image_path = os.path.join(image_filepath, image_filename)
          image = Image.open(image_path).convert("RGB")
        elif self.base == "GTA5":
          id = int(self.items[index])
          name = f"{id:0>5d}.png"
          # Open image and label
          image_path = os.path.join(self.image_filepath, name)
          image = Image.open(image_path).convert("RGB")
        elif self.base == "Brazil":
          id = self.items[index]
          name = f"{id}.jpg"

          # Open image and label
          image_path = os.path.join(self.image_filepath,name.split('_')[0])

          image_path = os.path.join(image_path, name)
          image = Image.open(image_path).convert("RGB")

        image = self.normalize(self.to_tensor(image))

        
          
        return image, str(image_path).split("/")[-1].split(".")[0], image_path

def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(scaled_img))

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(fliped_img))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

NUM_CLASSES = 3

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

# For visualization
label_colours = list(map(tuple, [
    [128, 64, 128],
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 0],  # the color of ignored label
]))

paleta = [
    128, 64, 128,
    255, 0, 0,
    255, 255, 0,
]

def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    h, w = mask.shape
    

    temp=mask.copy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, num_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

def load_config_from_yaml(yaml_file: str) -> DictConfig:
  # Carregar o arquivo YAML em um DictConfig
  config = OmegaConf.load(yaml_file)
  return config

def main(configmodel,path_image, path_list, path_checkpoint,path_save, save=True, split='test', base= 'GTA5', model = 'CCT', multiscale=False):
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    # DATA
    testdataset = testDataset(path_image, path_list, split, base)
    loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    num_classes = 3

    # SAVE DATA
    if not multiscale:
      path_save_mask= path_save+'/'+model+'/'+base+'/'+split+'/mask/'
      path_save_image= path_save+'/'+model+'/'+base+'/'+split+'/image/'
    else:
      path_save_mask= path_save+'/'+'multiscale'+model+'/'+base+'/'+split+'/mask/'
      path_save_image= path_save+'/'+'multiscale'+model+'/'+base+'/'+split+'/image/'
    print(path_save_mask)

    if save and not os.path.exists(path_save_mask):
        os.makedirs(path_save_mask)
    if save and not os.path.exists(path_save_image):
        os.makedirs(path_save_image)

    # MODEL
    checkpoint = torch.load(path_checkpoint)

    if model == 'CCT':
      path_to_remove = '/content/drive/MyDrive/TCC/LIGHT_ADAPTATION_DOMIAN/pixmatch'
      if path_to_remove in sys.path:
        sys.path.remove(path_to_remove)
      sys.path.append('/content/drive/MyDrive/TCC/LIGHT_ADAPTATION_DOMIAN/CCT')
      from models import CCT
      configmodel = json.load(open(configmodel))
      configmodel['model']['supervised'] = True; configmodel['model']['semi'] = False
      model = CCT(num_classes=num_classes,
                          conf=configmodel['model'], testing=True, versionmode=2)
      try:
          if 'module.' in list(checkpoint['state_dict'].keys())[0]: ## Correção de nomenclatura dos pesos
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
          model.load_state_dict(state_dict, strict=True)
      except Exception as e:
          print(f'Some modules are missing: {e}')
          
          model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
      # Carregar a configuração
      path_to_remove = '/content/drive/MyDrive/TCC/LIGHT_ADAPTATION_DOMIAN/CCT'
      if path_to_remove in sys.path:
        sys.path.remove(path_to_remove)
      sys.path.append('/content/drive/MyDrive/TCC/LIGHT_ADAPTATION_DOMIAN/pixmatch')
      from models import get_model_test
      model = get_model_test()
      try:
          model.load_state_dict(checkpoint['state_dict'], strict=True)
      except Exception as e:
          print(f'Some modules are missing: {e}')
          model.load_state_dict(checkpoint['state_dict'], strict=False)

    
    model.eval()
    model.cuda()

    

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    labels, predictions = [], []

    for index, data in enumerate(tbar):
        image, image_id, image_path = data

        image = image.cuda()
        

        # PREDICT
        with torch.no_grad():
            if multiscale:
              output = multi_scale_predict(model, image, scales, num_classes)
            else:
              output = model(image)
              output = output.data.cpu().numpy().squeeze(0)
            prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
            # SAVE RESULTS
            image_to_save = Image.fromarray(prediction, mode='P')
            image_to_save.putpalette(paleta)        
            image_to_save.save(path_save_mask+image_id[0]+'.png')
            shutil.copy(image_path[0], path_save_image+str(image_path[0]).split("/")[-1])

            if index == 10:
              break


