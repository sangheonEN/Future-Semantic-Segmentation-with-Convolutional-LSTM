import torch
import torch.nn as nn
import convlstm


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channels

        self.sequential = nn.Sequential()

        # 1x1 conv -> upsampling(transpose2d) -> element wise addition
        for i in range(len(in_channels[0])):
            self.sequential.add_module(
                f"conv_{i + 1}", nn.Conv2d(in_channels=in_channels[i],
                                           out_channels=in_channels[i], kernel_size=1),
            )
            self.sequential.add_module(
                f"transpose_conv_{i+1}", nn.ConvTranspose2d(in_channels= in_channels[i],
                                                          out_channels=out_channels[i], kernel_size=1, stride=2)
            )


    def forward(self, feature_1, feature_2, feature_3, feature_4):
        # feature shape: batch, channels, height, width
        # 1x1 conv -> upsampling(transpose2d) -> element wise addition
        conv_1 = self.sequential.conv_1(feature_1)
        transposeconv_1 = self.sequential.transpose_conv_1(conv_1)
        conv_2 = self.sequential.conv_2(torch.add(transposeconv_1, feature_2))
        transposeconv_2 = self.sequential.transpose_conv_2(conv_2)
        conv_3 = self.sequential.conv_3(torch.add(transposeconv_2, feature_3))
        transposeconv_3 = self.sequential.transpose_conv_3(conv_3)
        conv_4 = self.sequential.conv_4(torch.add(transposeconv_3, feature_4))
        transposeconv_4 = self.sequential.transpose_conv_4(conv_4)

        # softmax for cross entropy loss
        # nn.BCELoss binary cross entropy loss -> softmax or sigmoid function을 output에 넣어야함.
        # nn.CrossEntropyLoss는 필요 없음. 내장되어 있음.

        return transposeconv_4








