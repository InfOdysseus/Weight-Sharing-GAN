import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorMiddleBlock(nn.Module):
    def __init__(self, intChannel, outChannel, convFilter, convStride=1, convPadding=0, leaky_relu_alpha=0.2, dropout_ratio=0, batchNorm=False):
        super(DiscriminatorMiddleBlock, self).__init__()
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batchNorm = batchNorm
        self.dropout_ratio = dropout_ratio
        self.conv = nn.Conv2d(intChannel, outChannel,
                              convFilter, convStride, convPadding)
        if(self.batchNorm):
            self.bn = nn.BatchNorm2d(outChannel)
        if(self.dropout_ratio != 0):
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def forward(self, input):
        output = self.conv(input)
        if(self.batchNorm):
            output = self.bn(output)
        output = F.leaky_relu(output, self.leaky_relu_alpha)
        if(self.dropout_ratio != 0):
            output = self.dropout(output)

        return output
