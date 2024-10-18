import os
import json
import argparse
import torch
import math
from torch.utils.data.dataloader import DataLoader


#custom libs
from dataloaders.cityscapes_Dataset import CityscapeDataLoader, CityScape, inv_preprocess, decode_labels
from dataloaders.gta5_Dataset import GTA5_Dataset, GTA5DataLoader
from dataloaders.Brazil_Dataset import Brazil_Dataset, CityscapeBrazilDataLoader
from utils import Logger
from trainer import Trainer
from dataloaders import VOC
from models import CCT
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    torch.manual_seed(42)
    train_logger = Logger()

    versionmode = config['versionmode']

    if config['uda']:        
        supervised_loader_1 = GTA5DataLoader(config['train_supervised_1'])
        supervised_loader_2 = CityscapeDataLoader(config['train_supervised_2'])
        unsupervised_loader_2 = CityscapeBrazilDataLoader(config['train_unsupervised_2'])
        sup_val_loader_1 = GTA5DataLoader(config['sup_val_loader_1'])
        sup_val_loader_2 = CityscapeDataLoader(config['sup_val_loader_2'])
        un_val_loader_2 = CityscapeBrazilDataLoader(config['un_val_loader_2'])
    elif config['bases'] == "CB":
        supervised_loader_2 = CityscapeDataLoader(config['train_supervised_2'])
        unsupervised_loader_2 = CityscapeBrazilDataLoader(config['train_unsupervised_2'])
        sup_val_loader_2 = CityscapeDataLoader(config['sup_val_loader_2'])
        un_val_loader_2 = CityscapeBrazilDataLoader(config['un_val_loader_2'])
        supervised_loader_1 = None
        sup_val_loader_1 = None
    elif config['bases'] == "GC":
        supervised_loader_2 = GTA5DataLoader(config['train_supervised_2'])
        unsupervised_loader_2 = CityscapeDataLoader(config['train_unsupervised_2'])
        sup_val_loader_2 = GTA5DataLoader(config['sup_val_loader_2'])
        un_val_loader_2 = CityscapeDataLoader(config['un_val_loader_2'])
        supervised_loader_1 = None
        sup_val_loader_1 = None

    elif config['bases'] == "GB":
        supervised_loader_2 = GTA5DataLoader(config['train_supervised_2'])
        unsupervised_loader_2 = CityscapeBrazilDataLoader(config['train_unsupervised_2'])
        sup_val_loader_2 = GTA5DataLoader(config['sup_val_loader_2'])
        un_val_loader_2 = CityscapeBrazilDataLoader(config['un_val_loader_2'])
        supervised_loader_1 = None
        sup_val_loader_1 = None

    val_loader = sup_val_loader_2

    iter_per_epoch = len(unsupervised_loader_2)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(supervised_loader_2) # calculare class occurences
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, ignore_index = config['ignore_index'], alpha = alpha, gamma = 2, smooth = 1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader_2),
                                        rampup_ends=rampup_ends)
    print(val_loader.dataset.num_classes)
    print(val_loader.dataset.ignore_index)
    model = CCT(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
                            ignore_index=val_loader.dataset.ignore_index,
                            pretreined_path = "C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT/models/backbones/pretrained/mobilenetv2_weights.pth",
                            versionmode= versionmode)
    print(f'Model:\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader_1=supervised_loader_1,
        supervised_loader_2=supervised_loader_2,
        unsupervised_loader_2=unsupervised_loader_2,
        sup_val_loader_1=sup_val_loader_1,
        sup_val_loader_2=sup_val_loader_2,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger,
        un_val_loader_2=un_val_loader_2)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    main(config, args.resume)
