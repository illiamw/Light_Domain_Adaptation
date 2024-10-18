import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from models import aspp
from models.separableconv import SeparableConv2d 
from models import MobileNetV2




class LiteSeg(nn.Module):
    
    def __init__(self, n_classes=3,PRETRAINED_WEIGHTS=None, pretrained=False):
        
        super(LiteSeg, self).__init__()
        print("LiteSeg-MobileNet...")

        self.mobile_features=MobileNetV2.MobileNetV2()
        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS)
            if 'module.' in list(state_dict.keys())[0]: ## Correção de nomenclatura dos pesos
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.mobile_features.load_state_dict(state_dict)
        
        rates = [1, 3, 6, 9]


        self.aspp1 = aspp.ASPP(1280, 96, rate=rates[0])
        self.aspp2 = aspp.ASPP(1280, 96, rate=rates[1])
        self.aspp3 = aspp.ASPP(1280, 96, rate=rates[2])
        self.aspp4 = aspp.ASPP(1280, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1280, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        self.conv1 =SeparableConv2d(480+1280,96,1)
        self.bn1 = nn.BatchNorm2d(96)
    
        self.last_conv = nn.Sequential(#nn.Conv2d(24+96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(24+96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       nn.Conv2d(96, n_classes, kernel_size=1, stride=1))
        
    def forward(self, input):
        x, low_level_features = self.mobile_features(input)
        #print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)
        #print('after aspp cat',x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
       # ablation=torch.max(low_level_features, 1)[1]
        #print('after con on aspp output',x.size())

        ##comment to remove low feature
        #low_level_features = self.conv2(low_level_features)
        #low_level_features = self.bn2(low_level_features)
        #low_level_features = self.relu(low_level_features)
        #print("low",low_level_features.size())
        
        x = torch.cat((x, low_level_features), dim=1)
        #print('after cat low feature with output of aspp',x.size())

        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

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

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.mobile_features)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.last_conv.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

