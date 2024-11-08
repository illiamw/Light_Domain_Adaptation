import os
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np
from datetime import datetime
import torch
import argparse

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.Brasil_Dataset import Brasil_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models import get_model
from models.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13
from scipy.stats import entropy

class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=3):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index) 
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss



def compute_entropy_shannon(pred):
    argpred = np.argmax(pred.data.cpu().numpy(), axis=1)
    # Supondo que `probabilities` tenha a forma (Número de imagens, Número de classes, Altura, Largura)
    
    flattened = argpred.flatten()

    # Contar a frequência de cada classe
    unique, counts = np.unique(flattened, return_counts=True)
    probabilities = counts / counts.sum()
    image_entropy = entropy(probabilities, base=2)


    return image_entropy


class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_iter = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes)
        self.evaluator_target_train = Eval(self.cfg.data.num_classes)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.loss_MaxSquareloss = MaxSquareloss(self.ignore_index, self.cfg.data.num_classes)

        # Model
        self.model, params = get_model(self.cfg)
        # self.model = nn.DataParallel(self.model, device_ids=[0])  # TODO: test multi-gpu
        self.model.to(self.device)      

        # Modelo Zero
        state_dict = torch.load("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch/pretrained/liteseg-mobilenet-cityscapes-v2.pth")
        if 'module.' in list(state_dict.keys())[0]: ## Correção de nomenclatura dos pesos
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  

        state_dict["last_conv.6.weight"] = torch.randn(19, 96, 1, 1)
        state_dict["last_conv.6.bias"] = torch.randn(19)
        print("Carregando modelo zero: liteseg-mobilenet-cityscapes-v2.pth")
        self.model.load_state_dict(state_dict)

        # print(self.model)

        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        # Optimizer
        if self.cfg.opt.kind == "SGD":
            self.optimizer = torch.optim.SGD(
                params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)
        elif self.cfg.opt.kind == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(
                0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        else:
            raise NotImplementedError()
        self.lr_factor = 10

        self.save_checkpoint(self.cfg.name+'inicial_teste.pth')


        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'cityscapes':
            source_train_dataset = City_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = City_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        elif self.cfg.data.target.dataset == 'brazil':
            target_train_dataset = Brasil_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = Brasil_Dataset(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        
        

        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()


    def train(self):

        # Loop over epochs
        self.continue_training = True
        while self.continue_training:

            # Train for a single epoch
            self.train_one_epoch()

            # Use EMA params to evaluate performance
            self.ema.apply_shadow()
            self.ema.model.eval()
            self.ema.model.cuda()

            # a cada x epocas
            if (self.epoch + 1) % self.cfg.val_per_epoc == 0:
              # Validate on source (if possible) and target
              if self.cfg.data.source_val_iterations > 0:
                PA_source, MPA_source, MIoU_source, FWIoU_source = self.validate(mode='source')
              PA_target, MPA_target, MIoU_target, FWIoU_target = self.validate()
             
            

            # Restore current (non-EMA) params for training
            self.ema.restore()

            if (self.epoch + 1) % self.cfg.val_per_epoc == 0:

              # Log val results
              if self.cfg.data.target.dataset != 'brazil':
                MIoU = MIoU_target
              else:
                MIoU = MIoU_source
                print("Brazil: Não possui target real")  
            
            
              # Save checkpoint if new best model
              self.current_MIoU = MIoU
              is_best = MIoU > self.best_MIou
              if is_best:
                  self.best_MIou = MIoU
                  self.best_iter = self.iter
                  self.logger.info("=> Saving a new best checkpoint...")
                  print("=> Saving a new best checkpoint...")

                  self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                      self.best_MIou, self.best_iter))
                  self.save_checkpoint(self.cfg.name+'best.pth')
              else:
                  self.logger.info("=> The MIoU of val did not improve.")
                  self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                      self.best_MIou, self.best_iter))
            
            
            #Save a cada x epocas
            if (self.epoch + 1) % self.cfg.save_per_epoc == 0:
              self.save_checkpoint(self.cfg.name+f"_epoca_{self.epoch}.pth")

            self.epoch += 1

        # Save final checkpoint
        self.logger.info("=> The best MIou was {:.3f} at iter {}".format(
            self.best_MIou, self.best_iter))
        self.logger.info(
            "=> Saving the final checkpoint to {}".format('final.pth'))
        self.save_checkpoint(self.cfg.name+'final.pth')

    def train_one_epoch(self):

        print("Epoca:", self.epoch)

        # Load and reset
        self.model.train()
        self.evaluator.reset()
        self.evaluator_target_train.reset()

        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_dataloader))
        print(total)
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1} - {self.iter}"
        )):

            # Learning rate
            self.poly_lr_scheduler(optimizer=self.optimizer)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.iter)

            # Losses
            losses = {}

            ##########################
            # Source supervised loss #
            ##########################
            x, y, _ = batch_s

            if True:  # For VS Code collapsing

                # Data
                x = x.to(self.device)
                y = y.squeeze(dim=1).to(device=self.device,
                                        dtype=torch.long, non_blocking=True)

                # Fourier mix: source --> target
                if self.cfg.source_fourier:
                    x = fourier_mix(src_images=x, tgt_images=batch_t[0].to(
                        self.device), L=self.cfg.fourier_beta)

                # Forward
                pred = self.model(x)
                self.metrics('source', 1, y, pred)
                pred_1, pred_2 = unpack(pred)

                # Loss (source)
                loss_source_1 = self.loss(pred_1, y)
                if self.cfg.aux:
                    loss_source_2 = self.loss(pred_2, y) * self.cfg.lam_aux
                    loss_source = loss_source_1 + loss_source_2
                else:
                    loss_source = loss_source_1

                # Backward
                loss_source.backward()

                # Clean up
                losses['source_main'] = loss_source_1.cpu().item()
                if self.cfg.aux:
                    losses['source_aux'] = loss_source_2.cpu().item()
                    del loss_source_2
                del x, y, loss_source, loss_source_1

            ####################################
            # Shannon Entropy Metrics - Target #
            ####################################
            self.writer.add_scalar(f'train/shannon_source', compute_entropy_shannon(pred_1), self.iter)

            ######################
            # Target Pseudolabel #
            ######################
            x, _, _ = batch_t
            x = x.to(self.device)

            # First step: run non-augmented image though model to get predictions
            with torch.no_grad():

                # Substep 1: forward pass
                pred = self.model(x.to(self.device))
                pred_1, pred_2 = unpack(pred)

                if self.cfg.data.target.dataset != 'brazil':
                  _, y, _ = batch_t
                  self.metrics('target', 1, y, pred)
                

                # Substep 2: convert soft predictions to hard predictions
                pred_P_1 = F.softmax(pred_1, dim=1)
                label_1 = torch.argmax(pred_P_1.detach(), dim=1)
                maxpred_1, argpred_1 = torch.max(pred_P_1.detach(), dim=1)
                T = self.cfg.pseudolabel_threshold
                mask_1 = (maxpred_1 > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long) * self.ignore_index
                label_1 = torch.where(mask_1, label_1, ignore_tensor)
                if self.cfg.aux:
                    pred_P_2 = F.softmax(pred_2, dim=1)
                    maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
                    pred_c = (pred_P_1 + pred_P_2) / 2
                    maxpred_c, argpred_c = torch.max(pred_c, dim=1)
                    mask = (maxpred_1 > T) | (maxpred_2 > T)
                    label_2 = torch.where(mask, argpred_c, ignore_tensor)

            ###################
            # Max Square loss #
            ###################
            if self.cfg.lam_squaremax > 0:
              loss_square = self.loss_MaxSquareloss(pred_1) * \
                    self.cfg.lam_squaremax
              losses['square_main'] = loss_square.cpu().item()

            ####################################
            # Shannon Entropy Metrics - Target #
            ####################################
            self.writer.add_scalar(f'train/shannon_target', compute_entropy_shannon(pred_1), self.iter)
            

            ############
            # Aug loss #
            ############
            if self.cfg.lam_aug > 0:

                # Second step: augment image and label
                x_aug, y_aug_1 = augment(
                    images=x.cpu(), labels=label_1.detach().cpu(), aug=self.aug)
                y_aug_1 = y_aug_1.to(device=self.device, non_blocking=True)
                if self.cfg.aux:
                    _, y_aug_2 = augment(
                        images=x.cpu(), labels=label_2.detach().cpu(), aug=self.aug)
                    y_aug_2 = y_aug_2.to(device=self.device, non_blocking=True)

                # Third step: run augmented image through model to get predictions
                pred_aug = self.model(x_aug.to(self.device))
                pred_aug_1, pred_aug_2 = unpack(pred_aug)

                # Fourth step: calculate loss
                loss_aug_1 = self.loss(pred_aug_1, y_aug_1) * \
                    self.cfg.lam_aug
                if self.cfg.aux:
                    loss_aug_2 = self.loss(pred_aug_2, y_aug_2) * \
                        self.cfg.lam_aug * self.cfg.lam_aux
                    loss_aug = loss_aug_1 + loss_aug_2
                else:
                    loss_aug = loss_aug_1

                # Backward
                loss_aug.backward()

                # Clean up
                losses['aug_main'] = loss_aug_1.cpu().item()
                if self.cfg.aux:
                    losses['aug_aux'] = loss_aug_2.cpu().item()
                    del pred_aug_2, loss_aug_2
                del pred_aug, pred_aug_1, loss_aug, loss_aug_1
  


            ################
            # Fourier Loss #
            ################
            if self.cfg.lam_fourier > 0:

                # Second step: fourier mix
                x_fourier = fourier_mix(
                    src_images=x.to(self.device),
                    tgt_images=batch_s[0].to(self.device),
                    L=self.cfg.fourier_beta)

                # Third step: run mixed image through model to get predictions
                pred_fourier = self.model(x_fourier.to(self.device))
                pred_fourier_1, pred_fourier_2 = unpack(pred_fourier)

                # Fourth step: calculate loss
                loss_fourier_1 = self.loss(pred_fourier_1, label_1) * \
                    self.cfg.lam_fourier

                if self.cfg.aux:
                    loss_fourier_2 = self.loss(pred_fourier_2, label_2) * \
                        self.cfg.lam_fourier * self.cfg.lam_aux
                    loss_fourier = loss_fourier_1 + loss_fourier_2
                else:
                    loss_fourier = loss_fourier_1

                # Backward
                loss_fourier.backward()

                # Clean up
                losses['fourier_main'] = loss_fourier_1.cpu().item()
                if self.cfg.aux:
                    losses['fourier_aux'] = loss_fourier_2.cpu().item()
                del pred_fourier, pred_fourier_1, pred_fourier_2, loss_fourier, loss_fourier_1, loss_fourier_2

            ###############
            # CutMix Loss #
            ###############
            if self.cfg.lam_cutmix > 0:

                # Second step: CutMix
                x_cutmix, y_cutmix = cutmix_combine(
                    images_1=x,
                    labels_1=label_1.unsqueeze(dim=1),
                    images_2=batch_s[0].to(self.device),
                    labels_2=batch_s[1].unsqueeze(dim=1).to(self.device, dtype=torch.long))
                y_cutmix = y_cutmix.squeeze(dim=1)

                # Third step: run mixed image through model to get predictions
                pred_cutmix = self.model(x_cutmix)
                pred_cutmix_1, pred_cutmix_2 = unpack(pred_cutmix)

                # Fourth step: calculate loss
                loss_cutmix_1 = self.loss(pred_cutmix_1, y_cutmix) * \
                    self.cfg.lam_cutmix
                if self.cfg.aux:
                    loss_cutmix_2 = self.loss(pred_cutmix_2, y_cutmix) * \
                        self.cfg.lam_cutmix * self.cfg.lam_aux
                    loss_cutmix = loss_cutmix_1 + loss_cutmix_2
                else:
                    loss_cutmix = loss_cutmix_1

                # Backward
                loss_cutmix.backward()

                # Clean up
                losses['cutmix_main'] = loss_cutmix_1.cpu().item()
                if self.cfg.aux:
                    losses['cutmix_aux'] = loss_cutmix_2.cpu().item()
                del pred_cutmix, pred_cutmix_1, pred_cutmix_2, loss_cutmix, loss_cutmix_1, loss_cutmix_2

            ###############
            # CutMix Loss #
            ###############

            # Step optimizer if accumulated enough gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update model EMA parameters each step
            self.ema.update_params()

            # Calculate total loss
            total_loss = sum(losses.values())
            self.writer.add_scalar(f'train/total_loss_sup_unsup', total_loss, self.iter)

            # Log main losses
            for name, loss in losses.items():
                self.writer.add_scalar(f'train/{name}', loss, self.iter)

            # Log
            if batch_idx % 100 == 0:
                log_string = f"[Epoch {self.epoch}]\t"
                log_string += '\t'.join([f'{n}: {l:.3f}' for n, l in losses.items()])
                self.logger.info(log_string)

            # Increment global iteration counter
            self.iter += 1

            
            # End training after finishing iterations
            if self.iter > self.cfg.opt.iterations:
                self.continue_training = False
                return

        # After each epoch, update model EMA buffers (i.e. batch norm stats)
        self.ema.update_buffer()
        self.metrics(data="none", mode=2)


    def metrics(self, data, mode=1, y=None, pred=None):
        
        if mode == 1:
          # Convert to numpy
          label = y.squeeze(dim=1).cpu().numpy()
          argpred = np.argmax(pred.data.cpu().numpy(), axis=1)

          # Add to evaluator
          if data == 'source':
            self.evaluator.add_batch(label, argpred)
          else:
            self.evaluator_target_train.add_batch(label, argpred)
          

        else:
          PA = self.evaluator.Pixel_Accuracy()
          MPA = self.evaluator.Mean_Pixel_Accuracy()
          MIoU = self.evaluator.Mean_Intersection_over_Union()
          FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
          PC = self.evaluator.Mean_Precision()
          self.logger.info('TRAINEpoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
              self.epoch, PA, MPA, MIoU, FWIoU, PC))
          print('TRAINEpoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
              self.epoch, PA, MPA, MIoU, FWIoU, PC))
          self.writer.add_scalar(f"train/PA/source", PA, self.epoch)
          self.writer.add_scalar(f"train/MPA/source", MPA, self.epoch)
          self.writer.add_scalar(f"train/MIoU/source", MIoU, self.epoch)
          self.writer.add_scalar(f"train/FWIoU/source", FWIoU, self.epoch)
          self.evaluator.reset()
          
          if self.cfg.data.target.dataset != 'brazil':
            PA = self.evaluator_target_train.Pixel_Accuracy()
            MPA = self.evaluator_target_train.Mean_Pixel_Accuracy()
            MIoU = self.evaluator_target_train.Mean_Intersection_over_Union()
            FWIoU = self.evaluator_target_train.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator_target_train.Mean_Precision()
            self.logger.info('TRAINEpoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            print('TRAINEpoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar(f"train/PA/target", PA, self.epoch)
            self.writer.add_scalar(f"train/MPA/target", MPA, self.epoch)
            self.writer.add_scalar(f"train/MIoU/target", MIoU, self.epoch)
            self.writer.add_scalar(f"train/FWIoU/target", FWIoU, self.epoch)
            self.evaluator_target_train.reset()
          
          
        


    @ torch.no_grad()
    def validate(self, mode='target'):
        """Validate on target"""
        print('Validating')
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_val_dataloader
        elif mode == 'source':
            val_loader = self.source_val_dataloader
        else:
            raise NotImplementedError()

        total_data = len(val_loader)
        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]

            # Convert to numpy
            label = y.squeeze(dim=1).cpu().numpy()
            argpred = np.argmax(pred.data.cpu().numpy(), axis=1)

            # print(x.shape)
            # print(torch.unique(x))
            # print(y.shape)
            # print(torch.unique(y))
            # print(label.shape)
            
            # print(pred.shape)
            # print(argpred.shape)
            # print(np.unique(argpred))

            loss_source_1 = self.loss(pred, y)
            loss = loss_source_1.cpu().item()

            self.writer.add_scalar('val/'+str(mode)+"/"+"loss", loss, self.epoch)

            # Add to evaluator
            self.evaluator.add_batch(label, argpred)

            ####################################
            # Shannon Entropy Metrics - Target #
            ####################################
            
            self.writer.add_scalar(f"val/{mode}/shannon", compute_entropy_shannon(pred), (self.epoch*total_data) + val_idx)



        # Tensorboard images
        vis_imgs = 2
        images_inv = inv_preprocess(x.clone().cpu(), vis_imgs, numpy_transform=True)
        labels_colors = decode_labels(label, vis_imgs)
        preds_colors = decode_labels(argpred, vis_imgs)

        for index, (img, lab, predc) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('val/'+str(mode)+"/"+str(index) + '/images', img, self.epoch)
            self.writer.add_image('val/'+str(mode)+"/"+str(index) + '/labels', lab.permute(2, 0, 1), self.epoch)
            self.writer.add_image('val/'+str(mode)+"/"+str(index) + '/preds', predc.permute(2, 0, 1), self.epoch)

        # Calculate and log
        if self.cfg.data.source.kwargs.class_16:
            PA = self.evaluator.Pixel_Accuracy()
            MPA_16, MPA_13 = self.evaluator.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = self.evaluator.Mean_Precision()
            self.logger.info('VALEpoch:{:.3f}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                self.epoch, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
            self.logger.info('VALEpoch:{:.3f}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                self.epoch, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
            self.writer.add_scalar(f"val/PA/{mode}", PA, self.epoch)
            self.writer.add_scalar(f"val/MPA_16/{mode}", MPA_16, self.epoch)
            self.writer.add_scalar(f"val/MIoU_16/{mode}", MIoU_16, self.epoch)
            self.writer.add_scalar(f"val/FWIoU_16/{mode}", FWIoU_16, self.epoch)
            self.writer.add_scalar(f"val/MPA_13/{mode}", MPA_13, self.epoch)
            self.writer.add_scalar(f"val/MIoU_13/{mode}", MIoU_13, self.epoch)
            self.writer.add_scalar(f"val/FWIoU_13/{mode}", FWIoU_13, self.epoch)
            PA, MPA, MIoU, FWIoU = PA, MPA_13, MIoU_13, FWIoU_13
        else:
            PA = self.evaluator.Pixel_Accuracy()
            MPA = self.evaluator.Mean_Pixel_Accuracy()
            MIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator.Mean_Precision()
            self.logger.info('VALEpoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar(f"val/PA/{mode}", PA, self.epoch)
            self.writer.add_scalar(f"val/MPA/{mode}", MPA, self.epoch)
            self.writer.add_scalar(f"val/MIoU/{mode}", MIoU, self.epoch)
            self.writer.add_scalar(f"val/FWIoU/{mode}", FWIoU, self.epoch)

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename='checkpoint.pth'):
        torch.save({
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'state_dict': self.ema.model.state_dict(),
            'shadow': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }, 'C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch/checkpoint/' + filename)

    def load_checkpoint(self, filename):


        print(f"Carregando modelo: {filename}")
        checkpoint = torch.load(filename, map_location='cpu')

        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}

        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                self.epoch = checkpoint['epoch']
                self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
        init_lr = self.cfg.opt.lr if init_lr is None else init_lr
        iter = self.iter if iter is None else iter
        max_iter = self.cfg.opt.iterations if max_iter is None else max_iter
        power = self.cfg.opt.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


# @hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch_extend', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/experimento/Metrics/Metrics/PIXMATCH/"+cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)

    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)

    # Print configuration
    logger.info('/n' + OmegaConf.to_yaml(cfg))

    # # Train
    if cfg.train:
        trainer.train()

    # # Evaluate
    # else:
    #     trainer.validate()
    #     trainer.evaluator.Print_Every_class_Eval(
    #         out_16_13=(int(cfg.data.num_classes) in [16, 13]))

def load_config_from_yaml(yaml_file: str) -> DictConfig:
    # Carregar o arquivo YAML em um DictConfig
    config = OmegaConf.load(yaml_file)
    return config

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    args = parser.parse_args()
    # Carregar a configuração
    # config = load_config_from_yaml('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch/configs/GTA5toCITYSCAPE.yaml')
    # config = load_config_from_yaml('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch/configs/CITYSCAPEtoBrazil.yaml')
    config = load_config_from_yaml(args.config)
    main(config)
