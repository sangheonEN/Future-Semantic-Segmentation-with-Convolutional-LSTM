import torch.nn as nn
import new_encoder
import convlstm
import decoder


class Ensemble(nn.Module):
    def __init__(self, convLSTM_parameters_list, num_layers, transpose_channels_list):
        super(Ensemble, self).__init__()
        self.convLSTM_parameters_list = convLSTM_parameters_list
        self.num_layers = num_layers
        self.transpose_channels_list = transpose_channels_list

        self.back_bone = new_encoder.Encoder(class_num=3)
        self.conv_lstm = convlstm.Seq2Seq(self.convLSTM_parameters_list, self.num_layers)
        self.decoder = decoder.Decoder(in_channels=transpose_channels_list[0], out_channels=transpose_channels_list[1])

    def forward(self, image):
        # backbone에 들어가는건 sequence dimension 없이 batch, channel, h, w shape으로 해야해서 reshape 실시.
        batch, channel, sequence, h, w = image.shape
        image = image.reshape(batch*sequence, channel, h, w)
        feature_1, feature_2, feature_3, feature_4 = self.back_bone(image)
        # feature map batch*sequence -> batch, sequence shape으로 변경
        feature1_batch_sequence, feature1_channel, feature1_h, feature1_w = feature_1.shape
        feature2_batch_sequence, feature2_channel, feature2_h, feature2_w = feature_2.shape
        feature3_batch_sequence, feature3_channel, feature3_h, feature3_w = feature_3.shape
        feature4_batch_sequence, feature4_channel, feature4_h, feature4_w = feature_4.shape
        feature_1 = feature_1.reshape(batch, feature1_channel, sequence, feature1_h, feature1_w)
        feature_2 = feature_2.reshape(batch, feature2_channel, sequence, feature2_h, feature2_w)
        feature_3 = feature_3.reshape(batch, feature3_channel, sequence, feature3_h, feature3_w)
        feature_4 = feature_4.reshape(batch, feature4_channel, sequence, feature4_h, feature4_w)
        lstm_output_1, lstm_output_2, lstm_output_3, lstm_output_4 = self.conv_lstm(feature_1, feature_2, feature_3, feature_4)

        # decoder에서 transpose2d conv layer의 input, output channels lstm output인 input data와 일치해야함.
        # 디버그 하면서 channels 확인 필요.
        #RuntimeError: Given groups=1, weight of size [256, 65, 3, 3], expected input[16, 74, 16, 16] to have 65 channels, but got 74 channels instead
        final_output = self.decoder(lstm_output_1, lstm_output_2, lstm_output_3, lstm_output_4)

        return final_output