"""
Network Layer를 구성하기 위한 parameters를 저장하는 변수들을 전달하기 위해 선언
        # feature1 shape: 16, 10, 256, 16, 16
        # feature2 shape: 16, 10, 512, 8, 8
        # feature3 shape: 16, 10, 1024, 4, 4
        # feature4 shape: 16, 10, 2048, 2, 2
        # multi layer num: 3
        #
1. convLSTM Network parameters: num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers

2. Transpose convolution 2d parameters: input_channels_num, output_channels_num
   -> transpose_channels_list[0] == layer input_channels_num
   -> transpose_channels_list[1] == layer output_channels_num

"""
from collections import OrderedDict

transpose_channels_list = [
    # transpose_in_channels
    [2048, 1024, 512, 256],
    # transpose_out_channels
    [1024, 512, 256, 256]
]

convLSTM_parameters_list = [
    OrderedDict({"feature_1":[256, 256, (3, 3), (1, 1), "relu", (16, 16)]}),
    OrderedDict({"feature_2":[512, 512, (3, 3), (1, 1), "relu", (8, 8)]}),
    OrderedDict({"feature_3":[1024, 1024, (3, 3), (1, 1), "relu", (4, 4)]}),
    OrderedDict({"feature_4":[2048, 2048, (3, 3), (1, 1), "relu", (2, 2)]})
]