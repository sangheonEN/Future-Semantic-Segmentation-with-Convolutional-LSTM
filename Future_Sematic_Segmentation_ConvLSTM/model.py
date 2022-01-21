import torch.nn as nn
import encoder
import convlstm
import decoder


class Ensemble(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers):
        super(Ensemble).__init__()
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers

        self.back_bone = encoder.resnet50(pretrained=True)
        self.conv_lstm = convlstm.Seq2Seq(self.num_channels, self.num_kernels, self.kernel_size,
                                          self.padding, self.activation, self.frame_size,
                                          self.num_layers)
        self.decoder = decoder.Decoder()

    def forward(self, image):
        feature_1, feature_2, feature_3, feature_4 = self.back_bone(image)
        lstm_output_1, lstm_output_2, lstm_output_3, lstm_output_4 = self.conv_lstm(feature_1, feature_2, feature_3, feature_4)
        final_output = self.decoder(lstm_output_1, lstm_output_2, lstm_output_3, lstm_output_4)

        return final_output