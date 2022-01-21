import torch
import torch.nn as nn
import convlstm


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channels

        self.transpose = nn.Sequential()

        for i in range(len(in_channels[0])):
            self.transpose.add_module(
                f"transpose_conv_{i+1}", nn.ConvTranspose2d(in_channels= in_channels[i],
                                                          out_channels=out_channels[i])
            )


    def forward(self, feature_1, feature_2, feature_3, feature_4):
        # feature shape: batch, channels, height, width
        transposeconv_1 = self.transpose.transpose_conv_1(feature_1)
        transposeconv_2 = self.transpose.transpose_conv_2(torch.cat((transposeconv_1, feature_2), dim=1))
        transposeconv_3 = self.transpose.transpose_conv_3(torch.cat((transposeconv_2, feature_3), dim=1))
        transposeconv_4 = self.transpose.transpose_conv_4(torch.cat((transposeconv_3, feature_4), dim=1))

        # Sigmoid for binary classification.
        return nn.Sigmoid(transposeconv_4)








