import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
import gc
from scipy.stats import entropy


def compute_entropy_shannon(pred):
    argpred = pred.data.max(1)[1].cpu().numpy()
    # Supondo que `probabilities` tenha a forma (Número de imagens, Número de classes, Altura, Largura)
    
    flattened = argpred.flatten()

    # Contar a frequência de cada classe
    unique, counts = np.unique(flattened, return_counts=True)
    probabilities = counts / counts.sum()
    image_entropy = entropy(probabilities, base=2)


    return image_entropy


class Trainer(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader_1, supervised_loader_2, unsupervised_loader_2, iter_per_epoch,
                sup_val_loader_1=None,sup_val_loader_2=None , train_logger=None, un_val_loader_2=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger)
        
        self.supervised_loader_1 = supervised_loader_1
        self.supervised_loader_2 = supervised_loader_2
        self.unsupervised_loader_2 = unsupervised_loader_2
        self.val_loader_1 = sup_val_loader_1
        self.val_loader_2 = sup_val_loader_2
        self.un_val_loader_2 = un_val_loader_2

        self.ignore_index = self.val_loader_2.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader_2.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader_2.batch_size) + 1

        self.num_classes = self.val_loader_2.dataset.num_classes
        self.mode_1 = self.model.module.mode_1
        self.mode_2 = self.model.module.mode_2
        self.uda = config["uda"]

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader_2.MEAN, self.val_loader_2.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()



    def _train_epoch(self, epoch):
        self.html_results.save()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info('\n')
        self.model.train()
        self.iter = 0

        

        if self.uda:
            dataloader = zip(self.supervised_loader_1,self.supervised_loader_2, self.unsupervised_loader_2)

            tbar = tqdm(dataloader, ncols=150, total=len(self.unsupervised_loader_2), desc=f"Epoch {epoch} - {self.iter}")
            print(f"D1: {len(self.supervised_loader_1)}, D2: {len(self.supervised_loader_2)}, Du2: {len(self.unsupervised_loader_2)}")
        else:
            dataloader = zip(self.supervised_loader_2, self.unsupervised_loader_2)

            tbar = tqdm(dataloader, ncols=150, total=len(self.unsupervised_loader_2), desc=f"Epoch {epoch} - {self.iter}")
            print(f"D2: {len(self.supervised_loader_2)}, Du2: {len(self.unsupervised_loader_2)}")


           
        self._reset_metrics()
        # for batch_idx in tbar:
        if self.uda:
            print("Treinamento Adapitação de Dominio Semi-Supervisionado")
            for batch_idx, (batch_s, batch_t, batch_t_ul) in enumerate(tbar):
                
                (input_l_1_cpu, target_l_1_cpu), (input_l_2_cpu, target_l_2_cpu) ,(input_ul_2_cpu, target_ul_2_cpu) = batch_s, batch_t , batch_t_ul
                input_ul_2, target_ul_2 = input_ul_2_cpu.to(device), target_ul_2_cpu.to(device)

                input_l_1, target_l_1 = input_l_1_cpu.to(device), target_l_1_cpu.to(device)
                input_l_2, target_l_2 = input_l_2_cpu.to(device), target_l_2_cpu.to(device)
                torch.cuda.synchronize()
                del input_l_1_cpu, target_l_1_cpu, input_l_2_cpu, target_l_2_cpu, input_ul_2_cpu, target_ul_2_cpu,batch_s,batch_t,batch_t_ul
                input_l_1_cpu, target_l_1_cpu, input_l_2_cpu, target_l_2_cpu, input_ul_2_cpu, target_ul_2_cpu,batch_s,batch_t,batch_t_ul = None,None,None,None,None,None,None,None,None
                gc.collect()

                if input_l_1.shape[0] > 1:
                
                    self.optimizer_1.zero_grad()
                    total_loss_1, cur_losses_1, outputs_1 = self.model(x_l=input_l_1, target_l=target_l_1, x_ul=None,
                                                                curr_iter=batch_idx, target_ul=None, epoch=epoch-1, domain=1)
                    
                    total_loss_1 = total_loss_1.mean()
                    total_loss_1.backward()
                    self.optimizer_1.step()  


                    self._update_losses(cur_losses_1)
                    self._compute_metrics(outputs_1, target_l_1, None, epoch-1, 1)
                    logs = self._log_values(cur_losses_1)



                    self.optimizer_2.zero_grad()
                    total_loss_2, cur_losses_2, outputs_2   = self.model(x_l=input_l_2, target_l=target_l_2, x_ul=input_ul_2,
                                                                curr_iter=batch_idx, target_ul=target_ul_2, epoch=epoch-1, domain=2)   


                
                    total_loss_2 = total_loss_2.mean()
                    total_loss_2.backward()
                    self.optimizer_2.step()

                    self._update_losses(cur_losses_2)
                    self._compute_metrics(outputs_2, target_l_2, target_ul_2, epoch-1, 2)
                    logs = self._log_values(cur_losses_2)
                    
                    
                    if batch_idx % self.log_step == 0:
                        self.wrt_step = (epoch - 1) * len(self.unsupervised_loader_2) + batch_idx
                        self._write_scalars_tb(logs)
                        #########################################
                        # Shannon Entropy Metrics - SUP e UNSUP #
                        #########################################
                        self.writer.add_scalar(f'train/shannon_sup_1', compute_entropy_shannon(outputs_1["sup_pred_1"]), self.wrt_step)
                        self.writer.add_scalar(f'train/shannon_sup_2', compute_entropy_shannon(outputs_2["sup_pred_2"]), self.wrt_step)
                        self.writer.add_scalar(f'train/shannon_unsup_2', compute_entropy_shannon(outputs_2["unsup_pred_2"]), self.wrt_step)
                    

                    if batch_idx % int(len(self.supervised_loader_2)*0.9) == 0:
                        self._write_img_tb(input_l_1, target_l_1, input_l_2, target_l_2, input_ul_2, target_ul_2, outputs_1, outputs_2,epoch)

                    del input_l_1, target_l_1, input_l_2, target_l_2, input_ul_2, target_ul_2
                    del total_loss_1, cur_losses_1, outputs_1
                    del total_loss_2, cur_losses_2, outputs_2
                
                tbar.set_description('T ({}) D1 | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f}| D2 | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f}|'.format(
                    epoch, self.loss_sup_1.average, self.loss_unsup_1.average, self.loss_weakly_1.average,
                    self.pair_wise_1.average, self.mIoU_l_1, self.mIoU_ul_1, self.loss_sup_2.average, self.loss_unsup_2.average, self.loss_weakly_2.average,
                    self.pair_wise_2.average, self.mIoU_l_2, self.mIoU_ul_2))

                self.lr_scheduler_1.step(epoch=epoch-1)
                self.lr_scheduler_2.step(epoch=epoch-1)

                self.iter += 1

        else:
            print("Treinamento Semi-Supervisionado")
            for batch_idx, (batch_t, batch_t_ul) in enumerate(tbar):
                
                (input_l_2_cpu, target_l_2_cpu) ,(input_ul_2_cpu, target_ul_2_cpu) =  batch_t , batch_t_ul
                input_ul_2, target_ul_2 = input_ul_2_cpu.to(device), target_ul_2_cpu.to(device)

                
                input_l_2, target_l_2 = input_l_2_cpu.to(device), target_l_2_cpu.to(device)
                torch.cuda.synchronize()
                del input_l_2_cpu, target_l_2_cpu, input_ul_2_cpu, target_ul_2_cpu,batch_t,batch_t_ul
                input_l_2_cpu, target_l_2_cpu, input_ul_2_cpu, target_ul_2_cpu,batch_t,batch_t_ul = None,None,None,None,None,None
                gc.collect()
                
                if input_l_2.shape[0] > 1:
                
                    self.optimizer_2.zero_grad()
                    total_loss_2, cur_losses_2, outputs_2   = self.model(x_l=input_l_2, target_l=target_l_2, x_ul=input_ul_2,
                                                                curr_iter=batch_idx, target_ul=target_ul_2, epoch=epoch-1, domain=2)   


                
                    total_loss_2 = total_loss_2.mean()
                    total_loss_2.backward()
                    self.optimizer_2.step()

                    self._update_losses(cur_losses_2)
                    self._compute_metrics(outputs_2, target_l_2, target_ul_2, epoch-1, 2)
                    logs = self._log_values(cur_losses_2)
                    
                    
                    if batch_idx % self.log_step == 0:
                        self.wrt_step = (epoch - 1) * len(self.unsupervised_loader_2) + batch_idx
                        self._write_scalars_tb(logs)
                        #########################################
                        # Shannon Entropy Metrics - SUP e UNSUP #
                        #########################################
                        self.writer.add_scalar(f'train/shannon_sup_2', compute_entropy_shannon(outputs_2["sup_pred_2"]), self.wrt_step)
                        self.writer.add_scalar(f'train/shannon_unsup_2', compute_entropy_shannon(outputs_2["unsup_pred_2"]), self.wrt_step)
                    

                    if batch_idx % int(len(self.supervised_loader_2)*0.9) == 0:
                        self._write_img_tb(None, None, input_l_2, target_l_2, input_ul_2, target_ul_2, None, outputs_2,epoch)

                    del input_l_2, target_l_2, input_ul_2, target_ul_2
                    del total_loss_2, cur_losses_2, outputs_2
                
                tbar.set_description('T ({}) D2 | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f}|'.format(
                    epoch, self.loss_sup_2.average, self.loss_unsup_2.average, self.loss_weakly_2.average,
                    self.pair_wise_2.average, self.mIoU_l_2, self.mIoU_ul_2))

                
                self.lr_scheduler_2.step(epoch=epoch-1)

                self.iter += 1
        torch.cuda.empty_cache()

        return logs



    def _valid_epoch(self, epoch, domain = 1):

        print(f"Validação Sup D{domain}")

        if domain == 1:
            dataloader_val = self.val_loader_1
        if domain == 2:
            dataloader_val = self.val_loader_2
    
        if dataloader_val is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(dataloader_val, ncols=130)
        with torch.no_grad():
            val_visual = []
            total_imgs = len(tbar)
            entropy_shannon = []
            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data, domain=domain)
                output = output[:, :, :H, :W]

                entropy_shannon.append(compute_entropy_shannon(output))

                # LOSS
                target = target.long()
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter+inter, total_union+union
                total_correct, total_label = total_correct+correct, total_label+labeled

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    if isinstance(data, list): data = data[0]
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                total_loss_val.average, pixAcc, mIoU))

            self._add_img_tb(val_visual, f'{self.wrt_mode}_D{domain}')

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(dataloader_val)
            #########################################
            # Shannon Entropy Metrics - SUP         #
            #########################################
            self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/shannon_sup', np.sum(entropy_shannon) / total_imgs, self.wrt_step)
            

            self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/loss', total_loss_val.average, self.wrt_step)
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/{k}', v, self.wrt_step)

            

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            self.html_results.add_results(epoch=epoch, seg_resuts=log)
            self.html_results.save()

            if (time.time() - self.start_time) / 3600 > 22:
                self._save_checkpoint(epoch, save_best=self.improved)

        torch.cuda.empty_cache()
        return log

    def _valid_epoch_unsup(self, epoch, domain = 1):

        print(f"Validação Unsup D{domain}")

        if domain == 1:
            dataloader_val = self.un_val_loader_2
        if domain == 2:
            dataloader_val = self.un_val_loader_2

        if dataloader_val is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION UNSUP ######')

        self.model.eval()
        self.wrt_mode = 'val_unsup'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(dataloader_val, ncols=130)
        with torch.no_grad():
            val_visual = []
            total_imgs = len(tbar)
            entropy_shannon = []
            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data, domain=domain)
                output = output[:, :, :H, :W]

                entropy_shannon.append(compute_entropy_shannon(output))

                # LOSS
                target = target.long()
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter+inter, total_union+union
                total_correct, total_label = total_correct+correct, total_label+labeled

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    if isinstance(data, list): data = data[0]
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                total_loss_val.average, pixAcc, mIoU))

            self._add_img_tb(val_visual, 'val_unsup')

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(dataloader_val)
            #########################################
            # Shannon Entropy Metrics - UNSUP         #
            #########################################
            self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/shannon_unsup', np.sum(entropy_shannon) / total_imgs, self.wrt_step)
            

            self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/loss', total_loss_val.average, self.wrt_step)
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}_D{domain}/{k}', v, self.wrt_step)

            if (time.time() - self.start_time) / 3600 > 22:
                self._save_checkpoint(epoch, save_best=self.improved)
        torch.cuda.empty_cache()
        



    def _reset_metrics(self):
        self.loss_sup_1 = AverageMeter()
        self.loss_unsup_1  = AverageMeter()
        self.loss_weakly_1 = AverageMeter()
        self.pair_wise_1 = AverageMeter()
        self.total_inter_l_1, self.total_union_l_1 = 0, 0
        self.total_correct_l_1, self.total_label_l_1 = 0, 0
        self.total_inter_ul_1, self.total_union_ul_1 = 0, 0
        self.total_correct_ul_1, self.total_label_ul_1 = 0, 0
        self.mIoU_l_1, self.mIoU_ul_1 = 0, 0
        self.pixel_acc_l_1, self.pixel_acc_ul_1 = 0, 0
        self.class_iou_l_1, self.class_iou_ul_1 = {}, {}

        self.loss_sup_2 = AverageMeter()
        self.loss_unsup_2  = AverageMeter()
        self.loss_weakly_2 = AverageMeter()
        self.pair_wise_2 = AverageMeter()
        self.total_inter_l_2, self.total_union_l_2 = 0, 0
        self.total_correct_l_2, self.total_label_l_2 = 0, 0
        self.total_inter_ul_2, self.total_union_ul_2 = 0, 0
        self.total_correct_ul_2, self.total_label_ul_2 = 0, 0
        self.mIoU_l_2, self.mIoU_ul_2 = 0, 0
        self.pixel_acc_l_2, self.pixel_acc_ul_2 = 0, 0
        self.class_iou_l_2, self.class_iou_ul_2 = {}, {}



    def _update_losses(self, cur_losses):
        if "loss_sup_1" in cur_losses.keys():
            self.loss_sup_1.update(cur_losses['loss_sup_1'].mean().item())
        if "loss_unsup_1" in cur_losses.keys():
            self.loss_unsup_1.update(cur_losses['loss_unsup_1'].mean().item())
        if "loss_weakly_1" in cur_losses.keys():
            self.loss_weakly_1.update(cur_losses['loss_weakly_1'].mean().item())
        if "pair_wise_1" in cur_losses.keys():
            self.pair_wise_1.update(cur_losses['pair_wise_1'].mean().item())

        if "loss_sup_2" in cur_losses.keys():
            self.loss_sup_2.update(cur_losses['loss_sup_2'].mean().item())
        if "loss_unsup_2" in cur_losses.keys():
            self.loss_unsup_2.update(cur_losses['loss_unsup_2'].mean().item())
        if "loss_weakly_2" in cur_losses.keys():
            self.loss_weakly_2.update(cur_losses['loss_weakly_2'].mean().item())
        if "pair_wise_2" in cur_losses.keys():
            self.pair_wise_2.update(cur_losses['pair_wise_2'].mean().item())



    def _compute_metrics(self, outputs, target_l, target_ul, epoch, domain = 1):

        if domain == 1:
            seg_metrics_l_1 = eval_metrics(outputs[f'sup_pred_{domain}'], target_l, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_l_1, True, domain=1)
            seg_metrics_l_1 = self._get_seg_metrics(True, domain=1)
            self.pixel_acc_l_1, self.mIoU_l_1, self.class_iou_l_1 = seg_metrics_l_1.values()

        if domain == 2:
            seg_metrics_l_2 = eval_metrics(outputs[f'sup_pred_{domain}'], target_l, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_l_2, True, domain=2)
            seg_metrics_l_2 = self._get_seg_metrics(True, domain=2)
            self.pixel_acc_l_2, self.mIoU_l_2, self.class_iou_l_2 = seg_metrics_l_2.values()

        if self.mode_2 == 'semi' and domain == 2:
            seg_metrics_ul_2 = eval_metrics(outputs['unsup_pred_2'], target_ul, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_ul_2, False, domain=2)
            seg_metrics_ul_2 = self._get_seg_metrics(False, domain=2)
            self.pixel_acc_ul_2, self.mIoU_ul_2, self.class_iou_ul_2 = seg_metrics_ul_2.values()
            


    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True, domain=1):

        if domain == 1:
            if supervised:
                self.total_correct_l_1 += correct
                self.total_label_l_1 += labeled
                self.total_inter_l_1 += inter
                self.total_union_l_1 += union
            else:
                self.total_correct_ul_1 += correct
                self.total_label_ul_1 += labeled
                self.total_inter_ul_1 += inter
                self.total_union_ul_1 += union

        if domain == 2:
            if supervised:
                self.total_correct_l_2 += correct
                self.total_label_l_2 += labeled
                self.total_inter_l_2 += inter
                self.total_union_l_2 += union
            else:
                self.total_correct_ul_2 += correct
                self.total_label_ul_2 += labeled
                self.total_inter_ul_2 += inter
                self.total_union_ul_2 += union



    def _get_seg_metrics(self, supervised=True, domain = 1):
        if domain == 1:
            if supervised:
                pixAcc = 1.0 * self.total_correct_l_1 / (np.spacing(1) + self.total_label_l_1)
                IoU = 1.0 * self.total_inter_l_1 / (np.spacing(1) + self.total_union_l_1)
            else:
                pixAcc = 1.0 * self.total_correct_ul_1 / (np.spacing(1) + self.total_label_ul_1)
                IoU = 1.0 * self.total_inter_ul_1 / (np.spacing(1) + self.total_union_ul_1)
        if domain == 2:
            if supervised:
                pixAcc = 1.0 * self.total_correct_l_2 / (np.spacing(1) + self.total_label_l_2)
                IoU = 1.0 * self.total_inter_l_2 / (np.spacing(1) + self.total_union_l_2)
            else:
                pixAcc = 1.0 * self.total_correct_ul_2 / (np.spacing(1) + self.total_label_ul_2)
                IoU = 1.0 * self.total_inter_ul_2 / (np.spacing(1) + self.total_union_ul_2)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }



    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup_1" in cur_losses.keys():
            logs['loss_sup_1'] = self.loss_sup_1.average
        if "loss_unsup_1" in cur_losses.keys():
            logs['loss_unsup_1'] = self.loss_unsup_1.average
        if "loss_weakly_1" in cur_losses.keys():
            logs['loss_weakly_1'] = self.loss_weakly_1.average
        if "pair_wise_1" in cur_losses.keys():
            logs['pair_wise_1'] = self.pair_wise_1.average

        logs['mIoU_labeled_1'] = self.mIoU_l_1
        logs['pixel_acc_labeled_1'] = self.pixel_acc_l_1
        if self.mode_1 == 'semi':
            logs['mIoU_unlabeled_1'] = self.mIoU_ul_1
            logs['pixel_acc_unlabeled_1'] = self.pixel_acc_ul_1


        if "loss_sup_2" in cur_losses.keys():
            logs['loss_sup_2'] = self.loss_sup_2.average
        if "loss_unsup_2" in cur_losses.keys():
            logs['loss_unsup_2'] = self.loss_unsup_2.average
        if "loss_weakly_2" in cur_losses.keys():
            logs['loss_weakly_2'] = self.loss_weakly_2.average
        if "pair_wise_2" in cur_losses.keys():
            logs['pair_wise_2'] = self.pair_wise_2.average

        logs['mIoU_labeled_2'] = self.mIoU_l_2
        logs['pixel_acc_labeled_2'] = self.pixel_acc_l_2
        if self.mode_2 == 'semi':
            logs['mIoU_unlabeled_2'] = self.mIoU_ul_2
            logs['pixel_acc_unlabeled_2'] = self.pixel_acc_ul_2
        return logs


    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        if self.uda:
            for i, opt_group in enumerate(self.optimizer_1.param_groups):
                self.writer.add_scalar(f'train/1/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        for i, opt_group in enumerate(self.optimizer_2.param_groups):
            self.writer.add_scalar(f'train/2/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)



    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = [
            128, 64, 128,
            255, 0, 0,
            255, 255, 0,
        ]
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                        else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)



    def _write_img_tb(self, input_l_1, target_l_1,input_l_2, target_l_2, input_ul_2, target_ul_2, outputs_1,outputs_2, epoch):
        
        if input_l_1 != None  and target_l_1 != None:
            outputs_l_np_1 = outputs_1['sup_pred_1'].data.max(1)[1].cpu().numpy()
            targets_l_np_1 = target_l_1.data.cpu().numpy()
            imgs_1 = [[i.data.cpu(), j, k] for i, j, k in zip(input_l_1, outputs_l_np_1, targets_l_np_1)]
            self._add_img_tb(imgs_1, 'supervised_1')

        outputs_l_np_2 = outputs_2['sup_pred_2'].data.max(1)[1].cpu().numpy()
        targets_l_np_2 = target_l_2.data.cpu().numpy()
        imgs_2 = [[i.data.cpu(), j, k] for i, j, k in zip(input_l_2, outputs_l_np_2, targets_l_np_2)]
        self._add_img_tb(imgs_2, 'supervised_2')

        if self.mode_2 == 'semi':
            outputs_ul_np_2 = outputs_2['unsup_pred_2'].data.max(1)[1].cpu().numpy()
            targets_ul_np_2 = target_ul_2.data.cpu().numpy()
            imgs_2 = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul_2, outputs_ul_np_2, targets_ul_np_2)]
            self._add_img_tb(imgs_2, 'unsupervised_2')

