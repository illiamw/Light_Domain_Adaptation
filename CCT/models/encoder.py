from models.backbones.resnet_backbone import ResNetBackbone
from utils.helpers import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from models.backbones import aspp
from models.backbones.separableconv import SeparableConv2d 
from models.backbones import MobileNetV2


resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}

class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        if pretrained and not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)
        x = self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()

class EncoderLiteSegV1(nn.Module):
    def __init__(self, n_classes=19,PRETRAINED_WEIGHTS=None, pretrained=False):
        
        super(EncoderLiteSegV1, self).__init__()
        print("LiteSeg-MobileNet-V1...")

        self.mobile_features=MobileNetV2.MobileNetV2()
        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS)
            if 'module.' in list(state_dict.keys())[0]: ## Correção de nomenclatura dos pesos
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.mobile_features.load_state_dict(state_dict)
            print("Carregando pretreino")
        
        rates = [1, 3, 6, 9]

        
        global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1280, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        self.aspp = nn.Sequential(
          aspp.ASPP(1280, 96, rate=rates[0]), #0
          aspp.ASPP(1280, 96, rate=rates[1]), #1
          aspp.ASPP(1280, 96, rate=rates[2]), #2
          aspp.ASPP(1280, 96, rate=rates[3]), #3
          nn.ReLU(),                          #4
          global_avg_pool,                    #5
          SeparableConv2d(480+1280,96,1),     #6
          nn.BatchNorm2d(96)                  #7
        )
        
        
    def forward(self, input):
        x, low_level_features = self.mobile_features(input)
        #print(x.size())
        x1 = self.aspp[0](x)
        x2 = self.aspp[1](x)
        x3 = self.aspp[2](x)
        x4 = self.aspp[3](x)
        x5 = self.aspp[5](x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)
        #print('after aspp cat',x.size())
        x = self.aspp[6](x)
        x = self.aspp[7](x)
        x = self.aspp[4](x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
       
        
        x = torch.cat((x, low_level_features), dim=1)

        return x#,ablation

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_backbone_params(self):
        return self.mobile_features.parameters()

    def get_module_params(self):
        return self.aspp.parameters()

class EncoderLiteSegV2(nn.Module):
    def __init__(self, n_classes=19,PRETRAINED_WEIGHTS=None, pretrained=True):
        
        super(EncoderLiteSegV2, self).__init__()
        print("LiteSeg-MobileNet-V2...")

        self.mobile_features=MobileNetV2.MobileNetV2()
        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS)
            
            if 'module.' in list(state_dict.keys())[0]: ## Correção de nomenclatura dos pesos
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            if 'mobile_features.' in list(state_dict.keys())[0]: ## Correção de nomenclatura dos pesos
                state_dict = {k.replace('mobile_features.', ''): v for k, v in state_dict.items()}
            self.mobile_features.load_state_dict(state_dict)

        print("Pretreine carregado!")
        
        rates = [1, 3, 6, 9]

        last_conv = nn.Sequential(#nn.Conv2d(24+96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(24+96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU())

        global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1280, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        self.aspp = nn.Sequential(
          aspp.ASPP(1280, 96, rate=rates[0]), #0
          aspp.ASPP(1280, 96, rate=rates[1]), #1
          aspp.ASPP(1280, 96, rate=rates[2]), #2
          aspp.ASPP(1280, 96, rate=rates[3]), #3
          nn.ReLU(),                          #4
          global_avg_pool,                    #5
          SeparableConv2d(480+1280,96,1),     #6
          nn.BatchNorm2d(96),                 #7
          last_conv                           #8
        )
        
    
        
        
    def forward(self, input):
        x, low_level_features = self.mobile_features(input)
        #print(x.size())
        x1 = self.aspp[0](x)
        x2 = self.aspp[1](x)
        x3 = self.aspp[2](x)
        x4 = self.aspp[3](x)
        x5 = self.aspp[5](x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)
        #print('after aspp cat',x.size())
        x = self.aspp[6](x)
        x = self.aspp[7](x)
        x = self.aspp[4](x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
       
        
        x = torch.cat((x, low_level_features), dim=1)

        x = self.aspp[8](x)

        return x#,ablation

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_backbone_params(self):
        return self.mobile_features.parameters()

    def get_module_params(self):
        return self.aspp.parameters()

