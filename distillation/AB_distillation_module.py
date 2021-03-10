import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AB_Distill_Resnet_50_18(nn.Module):
    '''Distill Activation Boundary from Resnet50 to Resnet18.'''
    
    def __init__(self, t_net, s_net, batch_size, loss_multiplier):
        super(AB_Distill_Resnet_50_18, self).__init__()

        self.batch_size = batch_size
        self.loss_multiplier = loss_multiplier

        # Connector
        C1 = [nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256)]
        C2 = [nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512)]
        C3 = [nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1024)]
        C4 = [nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2048)]

        for m in C1 + C2 + C3 + C4:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        self.Connect3 = nn.Sequential(*C3)
        self.Connect4 = nn.Sequential(*C4)

        self.Connectors = nn.ModuleList([self.Connect1, self.Connect2, self.Connect3, self.Connect4])

        self.t_net = t_net
        self.s_net = s_net

        self.stage1 = True

    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def forward(self, x):
        inputs = x

        # Teacher network
        res0_t = self.t_net.maxpool(self.t_net.relu(self.t_net.bn1(self.t_net.conv1(inputs))))

        res1_t = self.t_net.layer1(res0_t)
        res2_t = self.t_net.layer2(res1_t)
        res3_t = self.t_net.layer3(res2_t)
        res4_t = self.t_net.layer4(res3_t)

        # Student network
        res0_s = self.s_net.maxpool(self.s_net.relu(self.s_net.bn1(self.s_net.conv1(inputs))))

        res1_s = self.s_net.layer1(res0_s)
        res2_s = self.s_net.layer2(res1_s)
        res3_s = self.s_net.layer3(res2_s)
        res4_s = self.s_net.layer4(res3_s)

        # Activation transfer loss
        loss_AT4 = ((self.Connect4(res4_s) > 0) ^ (res4_t > 0)).sum().float() / res4_t.nelement()
        loss_AT3 = ((self.Connect3(res3_s) > 0) ^ (res3_t > 0)).sum().float() / res3_t.nelement()
        loss_AT2 = ((self.Connect2(res2_s) > 0) ^ (res2_t > 0)).sum().float() / res2_t.nelement()
        loss_AT1 = ((self.Connect1(res1_s) > 0) ^ (res1_t > 0)).sum().float() / res1_t.nelement()

        loss_AT4 = loss_AT4.unsqueeze(0).unsqueeze(1)
        loss_AT3 = loss_AT3.unsqueeze(0).unsqueeze(1)
        loss_AT2 = loss_AT2.unsqueeze(0).unsqueeze(1)
        loss_AT1 = loss_AT1.unsqueeze(0).unsqueeze(1)

        # Alternative loss
        if self.stage1 is True:
            margin = 1.0
            loss = self.criterion_active_L2(self.Connect4(res4_s), res4_t.detach(), margin) / self.batch_size
            loss += self.criterion_active_L2(self.Connect3(res3_s), res3_t.detach(), margin) / self.batch_size / 2
            loss += self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 4
            loss += self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 8

            loss /= 1000

            loss = loss.unsqueeze(0).unsqueeze(1)
        else:
            loss = torch.zeros(1, 1).cuda()
            
        loss *= self.loss_multiplier

        # Return all losses
        return loss, loss_AT4, loss_AT3, loss_AT2, loss_AT1