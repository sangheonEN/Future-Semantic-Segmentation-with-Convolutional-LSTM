import torch
import torch.nn as nn
import convlstm

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channels

    def transpose_conv(self):
        return nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channles,
                                  kernel_size=3, stride=2, padding=1)

    def forward(self, feature_1, feature_2, feature_3, feature_4):
        # feature shape: batch, channels, height, width
        transposeconv_1 = self.transpose_conv(feature_1)
        transposeconv_2 = self.transpose_conv(torch.cat((transposeconv_1, feature_2), dim=1))
        transposeconv_3 = self.transpose_conv(torch.cat((transposeconv_2, feature_3), dim=1))
        transposeconv_4 = self.transpose_conv(torch.cat((transposeconv_3, feature_4), dim=1))

        # Sigmoid for binary classification.
        return nn.Sigmoid(transposeconv_4)








