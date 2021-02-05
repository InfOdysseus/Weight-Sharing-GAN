import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorMiddleBlock(nn.Module):
    def __init__(self, convInfo, nextConvInfo=(1, 1, 1, 1, 1), leaky_relu_alpha=0.2, batchNorm=False, last=False, img_channel = 3):
        super(GeneratorMiddleBlock, self).__init__()
        self.last = last
        self.img_channel = img_channel
        self.batchNorm = batchNorm
        self.leaky_relu_alpha = leaky_relu_alpha
        self.branch = nn.Conv2d(convInfo[0], convInfo[0], 3, padding=1)
        if(self.batchNorm):
            self.bn = nn.BatchNorm2d(convInfo[0])
        self.conv = nn.Conv2d(
            convInfo[0], convInfo[1], convInfo[2], convInfo[3], convInfo[4])
        
        if(self.last):
            self.firstTransposed = nn.ConvTranspose2d(
                convInfo[1], self.img_channel, convInfo[2], convInfo[3], convInfo[4])
        else:
            self.firstTransposed = nn.ConvTranspose2d(
                convInfo[1], nextConvInfo[1], convInfo[2], convInfo[3], convInfo[4])
            self.secondTransposed = nn.ConvTranspose2d(
                nextConvInfo[1]*2, nextConvInfo[0], nextConvInfo[2], nextConvInfo[3], nextConvInfo[4])
        for params in self.conv.parameters():
            params.requires_grad = False

    def forward(self, input):
        if(self.batchNorm):
            input = self.bn(input)
        connect = self.branch(input)
        output = self.conv(F.leaky_relu(input, self.leaky_relu_alpha))
        output = self.firstTransposed(
            F.leaky_relu(output.detach(), self.leaky_relu_alpha))
        if(not self.last):
            output = self.secondTransposed(
                F.leaky_relu(torch.cat([output, connect], dim=1), self.leaky_relu_alpha))
            return output
        else:
            return output+connect