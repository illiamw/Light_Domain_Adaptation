import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import MainDecoder, VATDecoder, DropOutDecoder, CutOutDecoder, ContextMaskingDecoder, ObjectMaskingDecoder, FeatureDropDecoder, FeatureNoiseDecoder
from models.encoder import Encoder, EncoderLiteSegV1, EncoderLiteSegV2
from utils.losses import CE_loss



class CCT(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, ignore_index=None, testing=False,
            pretrained=False, use_weak_lables=False, weakly_loss_w=0.4, versionmode=0, pretreined_path= None):

        if not testing:
            assert (ignore_index is not None) and (sup_loss is not None) and (cons_w_unsup is not None)

        super(CCT, self).__init__()
        assert int(conf['supervised_1']) + int(conf['semi_1']) == 1, 'one mode only'
        if conf['supervised_1']:
            self.mode_1 = 'supervised'
        else:
            self.mode_1 = 'semi'

        assert int(conf['supervised_2']) + int(conf['semi_2']) == 1, 'one mode only'
        if conf['supervised_2']:
            self.mode_2 = 'supervised'
        else:
            self.mode_2 = 'semi'

        # Supervised and unsupervised losses
        self.ignore_index = ignore_index
        if conf['un_loss'] == "KL":
          self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
          self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
          self.unsuper_loss = softmax_js_loss
        else:
            raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']
        self.softmax_temp = conf['softmax_temp']
        self.sup_loss = sup_loss
        self.sup_type = conf['sup_loss']

        # Use weak labels
        self.use_weak_lables = use_weak_lables
        self.weakly_loss_w = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # The main encoder
        if versionmode == 0:
          self.encoder = Encoder(pretrained=pretrained)
        elif versionmode == 1:
          self.encoder = EncoderLiteSegV1(pretrained=pretrained, PRETRAINED_WEIGHTS=pretreined_path)
        elif versionmode == 2:
          self.encoder = EncoderLiteSegV2(pretrained=pretrained, PRETRAINED_WEIGHTS=pretreined_path)
        # Create the model


        # The main encoder
        if versionmode == 0:
          upscale = 8
          num_out_ch = 2048
          decoder_in_ch = num_out_ch // 4
        elif versionmode == 1:
          upscale = 4
          decoder_in_ch = 120
        elif versionmode == 2:
          upscale = 4
          decoder_in_ch = 96

        if conf["uda"]: self.main_decoder_1 = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes , versionmode=versionmode)

        self.main_decoder_2 = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes , versionmode=versionmode)

        # The auxilary decoders
        if self.mode_1 == 'semi' or self.mode_1 == 'weakly_semi':
            vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
            							eps=conf['eps'], versionmode= versionmode) for _ in range(conf['vat'])]
            drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
            							drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'],versionmode= versionmode)
            							for _ in range(conf['drop'])]
            cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'], versionmode=versionmode)
            							for _ in range(conf['cutout'])]
            context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['context_masking'])]
            object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['object_masking'])]
            feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['feature_drop'])]
            feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
            							uniform_range=conf['uniform_range'], versionmode=versionmode)
            							for _ in range(conf['feature_noise'])]

            
            self.aux_decoders_1 = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
            
          # The auxilary decoders
        if self.mode_2 == 'semi' or self.mode_2 == 'weakly_semi':
            vat_decoder_2 = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
            							eps=conf['eps'], versionmode= versionmode) for _ in range(conf['vat'])]
            drop_decoder_2 = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
            							drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'],versionmode= versionmode)
            							for _ in range(conf['drop'])]
            cut_decoder_2 = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'], versionmode=versionmode)
            							for _ in range(conf['cutout'])]
            context_m_decoder_2 = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['context_masking'])]
            object_masking_2 = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['object_masking'])]
            feature_drop_2 = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes, versionmode=versionmode)
            							for _ in range(conf['feature_drop'])]
            feature_noise_2 = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
            							uniform_range=conf['uniform_range'], versionmode=versionmode)
            							for _ in range(conf['feature_noise'])]

            
            self.aux_decoders_2 = nn.ModuleList([*vat_decoder_2, *drop_decoder_2, *cut_decoder_2,
                                    *context_m_decoder_2, *object_masking_2, *feature_drop_2, *feature_noise_2])
            
    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, domain=1):
        if not self.training:
            if domain== 1:
              return self.main_decoder_1(self.encoder(x_l))
            if domain== 2:
              return self.main_decoder_2(self.encoder(x_l))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size = (x_l.size(2), x_l.size(3))

        if domain== 1:
          output_l = self.main_decoder_1(self.encoder(x_l))
        if domain== 2:
          output_l = self.main_decoder_2(self.encoder(x_l))

        
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        target_l = target_l.long()
        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index, temperature=self.softmax_temp) * self.sup_loss_w
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch, ignore_index=self.ignore_index) * self.sup_loss_w

        if domain == 1:

          # If supervised mode only, return
          if self.mode_1 == 'supervised':
              curr_losses = {'loss_sup_1': loss_sup}
              outputs = {'sup_pred_1': output_l}
              total_loss = loss_sup
              return total_loss, curr_losses, outputs

          # If semi supervised mode
          elif self.mode_1 == 'semi':
              # Get main prediction
              x_ul = self.encoder(x_ul)
              output_ul = self.main_decoder_1(x_ul)
              # Get auxiliary predictions
              outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders_1]
              targets = F.softmax(output_ul.detach(), dim=1)

              # Compute unsupervised loss
              loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                              conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                              for u in outputs_ul])
              loss_unsup = (loss_unsup / len(outputs_ul))
              curr_losses = {'loss_sup_1': loss_sup}

              if output_ul.shape != x_l.shape:
                  output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
              outputs = {'sup_pred_1': output_l, 'unsup_pred_1': output_ul}

              # Compute the unsupervised loss
              weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
              loss_unsup = loss_unsup * weight_u
              curr_losses['loss_unsup_1'] = loss_unsup
              total_loss = loss_unsup  + loss_sup

              # If case we're using weak lables, add the weak loss term with a weight (self.weakly_loss_w)
              if self.use_weak_lables:
                  weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                  loss_weakly = sum([CE_loss(outp, target_ul, ignore_index=self.ignore_index) for outp in outputs_ul]) / len(outputs_ul)
                  loss_weakly = loss_weakly * weight_w
                  curr_losses['loss_weakly_1'] = loss_weakly
                  total_loss += loss_weakly

              # Pair-wise loss
              if self.aux_constraint:
                  pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                  curr_losses['pair_wise_1'] = pair_wise
                  loss_unsup += pair_wise

        if domain == 2:

          if self.mode_2 == 'supervised':
              curr_losses = {'loss_sup_2': loss_sup}
              outputs = {'sup_pred_2': output_l}
              total_loss = loss_sup
              return total_loss, curr_losses, outputs

          # If semi supervised mode
          elif self.mode_2 == 'semi':
              input_size = (x_ul.size(2), x_ul.size(3))
              
              # Get main prediction
              x_ul = self.encoder(x_ul)
              output_ul = self.main_decoder_2(x_ul)
              # Get auxiliary predictions
              outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders_2]
              targets = F.softmax(output_ul.detach(), dim=1)

              # Compute unsupervised loss
              loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                              conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                              for u in outputs_ul])
              loss_unsup = (loss_unsup / len(outputs_ul))
              curr_losses = {'loss_sup_2': loss_sup}


              if output_ul.shape != x_ul.shape:
                  output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
              outputs = {'sup_pred_2': output_l, 'unsup_pred_2': output_ul}

              # Compute the unsupervised loss
              weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
              loss_unsup = loss_unsup * weight_u
              curr_losses['loss_unsup_2'] = loss_unsup
              total_loss = loss_unsup  + loss_sup

              # If case we're using weak lables, add the weak loss term with a weight (self.weakly_loss_w)
              if self.use_weak_lables:
                  weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                  loss_weakly = sum([CE_loss(outp, target_ul, ignore_index=self.ignore_index) for outp in outputs_ul]) / len(outputs_ul)
                  loss_weakly = loss_weakly * weight_w
                  curr_losses['loss_weakly_2'] = loss_weakly
                  total_loss += loss_weakly

              # Pair-wise loss
              if self.aux_constraint:
                  pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                  curr_losses['pair_wise_2'] = pair_wise
                  loss_unsup += pair_wise



          return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params_1(self):
        if self.mode_1 == 'semi':
            return chain(self.encoder.get_module_params(), self.main_decoder_1.parameters(), 
                        self.aux_decoders_1.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder_1.parameters())
    
    def get_other_params_2(self):
        if self.mode_2 == 'semi':
            return chain(self.encoder.get_module_params(), self.main_decoder_2.parameters(), 
                        self.aux_decoders_2.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder_2.parameters())

