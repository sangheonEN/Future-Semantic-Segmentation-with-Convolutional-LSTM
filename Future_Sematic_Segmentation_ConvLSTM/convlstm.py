import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTMCell(nn.Module):
    """
    1. 시퀀스별로 나눈 input data를 받아 LSTM 내부적인 연산 수행
    2. input과 hidden state concat -> channels dimension기준으로
    3. input과 hidden state concat 후 각 Gate에 input하기 위해 4등분으로 나누어 Convolution 연산 진행
    4. FC LSTM에서 나온 이전 Cell State의 weight를 각 Gate에 Hadamard Products (self.W_ci, self.W_co, self.W_cf)
    5. forget, input, output gate 계산
    6. 미래 시점(t+1)의 Cell, Hidden State 계산

    :return
    미래 시점 Hidden, Cell State

    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C

class ConvLSTM(nn.Module):
    """
    1. input data를 전달 받아 반복문으로 시퀀스별로 나눈 input data를 convLSTMcell에 전달
    2. 초기 hidden, cell state 생성
    3. 마지막 hidden state array 생성

    :param
    input data: X shape -> [batch_size, num_channels, seq_len, height, width]
    H, C data: H, C shape -> [batch_size, out_channels, height, width]
    output data: output shape -> [batch_size, out_channels, seq_len, height, width]
    :return
    시퀀스 수만큼의 Hidden State를 Sequence Dimension을 기준으로 취합하고 반환 [batch_size, output_channels, seq_len, height, width]
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, activation, frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


class Seq2Seq(nn.Module):
    """
    Multi Layer를 순차적으로 구성하기 위해 nn.Sequential() 사용

    :param
    num_channels: Initial Input Image channels num
    num_kernels: Hidden, Cell State channels num
    kernel_size: kernel size for Convolution operation
    padding: padding size for Convolution operation
    activation: tanh or relu for nonlinear LSTM paper 참고
    frame_size: image frame size
    num_layers: Multi layer num

    """

    def __init__(self, convLSTM_parameters_list, num_layers):
        super(Seq2Seq, self).__init__()
        """
        # parameters shape: num_channels, num_kernels, kernel_size, padding, activation, frame_size
        convLSTM_parameters_list[0].get("feature_1")
        convLSTM_parameters_list[0].get("feature_2")
        convLSTM_parameters_list[0].get("feature_3")
        convLSTM_parameters_list[0].get("feature_4")
        
        # layer name description
        - feature별 first convlstm layer
        convlstm1_feature_{1 ~ 4}
        - feature별 first convlstm layer의 batch normalization layer
        batchnorm_conv1_feature_{1~4}
        - feature별 second ~ final convlstm layer
        convlstm{1~multi_layer_num}_feature_{1~4}
        - feature별 second ~ final convlstm layer의 batch normalization layer
        batchnorm_conv{1 ~ multi_layer_num}_feature_{1~4}
        """

        self.sequential1 = nn.Sequential()
        self.sequential2 = nn.Sequential()
        self.sequential3 = nn.Sequential()
        self.sequential4 = nn.Sequential()

        self.sequence_layer(self.sequential1, convLSTM_parameters_list, num_layers, feature_num=1)
        self.sequence_layer(self.sequential2, convLSTM_parameters_list, num_layers, feature_num=2)
        self.sequence_layer(self.sequential3, convLSTM_parameters_list, num_layers, feature_num=3)
        self.sequence_layer(self.sequential4, convLSTM_parameters_list, num_layers, feature_num=4)

        # conv의 역할은 마지막 sequence output만 다시 convolution operate해서 차원을 축소하기 위함.
        # 있어야지. 왜냐하면, feature마다 sequence 마지막 output을 가져와서 결합해야하니까!
        # 여기서 convolution 연산을 통해 input 할때 channels 수로 다시 만들어줌
        self.conv1 = nn.Conv2d(
            in_channels=convLSTM_parameters_list[0].get(f"feature_{1}")[1],
            out_channels=convLSTM_parameters_list[0].get(f"feature_{1}")[0],
            kernel_size=convLSTM_parameters_list[0].get(f"feature_{1}")[2],
            padding=convLSTM_parameters_list[0].get(f"feature_{1}")[3]
        )

        self.conv2 = nn.Conv2d(
            in_channels=convLSTM_parameters_list[1].get(f"feature_{2}")[1],
            out_channels=convLSTM_parameters_list[1].get(f"feature_{2}")[0],
            kernel_size=convLSTM_parameters_list[1].get(f"feature_{2}")[2],
            padding=convLSTM_parameters_list[1].get(f"feature_{2}")[3]
        )

        self.conv3 = nn.Conv2d(
            in_channels=convLSTM_parameters_list[2].get(f"feature_{3}")[1],
            out_channels=convLSTM_parameters_list[2].get(f"feature_{3}")[0],
            kernel_size=convLSTM_parameters_list[2].get(f"feature_{3}")[2],
            padding=convLSTM_parameters_list[2].get(f"feature_{3}")[3]
        )

        self.conv4 = nn.Conv2d(
            in_channels=convLSTM_parameters_list[3].get(f"feature_{4}")[1],
            out_channels=convLSTM_parameters_list[3].get(f"feature_{4}")[0],
            kernel_size=convLSTM_parameters_list[3].get(f"feature_{4}")[2],
            padding=convLSTM_parameters_list[3].get(f"feature_{4}")[3]
        )

    def sequence_layer(self, sequential_instance, param_data, num_layers, feature_num):
        # feature_1 sequential layer
        sequential_instance.add_module(
            f"convlstm1_feature_{feature_num}", ConvLSTM(
                in_channels=param_data[feature_num-1].get(f"feature_{feature_num}")[0],
                out_channels=param_data[feature_num-1].get(f"feature_{feature_num}")[1],
                kernel_size=param_data[feature_num-1].get(f"feature_{feature_num}")[2],
                padding=param_data[feature_num-1].get(f"feature_{feature_num}")[3],
                activation=param_data[feature_num-1].get(f"feature_{feature_num}")[4],
                frame_size=param_data[feature_num-1].get(f"feature_{feature_num}")[5]
            )
        )
        sequential_instance.add_module(
            f"batchnorm_conv1_feature_{feature_num}",
            nn.BatchNorm3d(num_features=param_data[feature_num-1].get(f"feature_{feature_num}")[1])
        )

        # multi layer delete
        # for j in range(num_layers):
        #     sequential_instance.add_module(
        #         f"convlstm{j + 2}_feature_{feature_num}", ConvLSTM(
        #             in_channels=param_data[feature_num-1].get(f"feature_{feature_num}")[1],
        #             out_channels=param_data[feature_num-1].get(f"feature_{feature_num}")[1],
        #             kernel_size=param_data[feature_num-1].get(f"feature_{feature_num}")[2],
        #             padding=param_data[feature_num-1].get(f"feature_{feature_num}")[3],
        #             activation=param_data[feature_num-1].get(f"feature_{feature_num}")[4],
        #             frame_size=param_data[feature_num-1].get(f"feature_{feature_num}")[5]
        #         )
        #     )
        #     sequential_instance.add_module(
        #         f"batchnorm_conv{j + 2}_feature_{feature_num}",
        #         nn.BatchNorm3d(num_features=param_data[feature_num-1].get(f"feature_{feature_num}")[1])
        #     )

    def forward(self, feature_1, feature_2, feature_3, feature_4):
        # Forward propagation through all the layers
        # 각 ConvLSTM 모델에 들어가는 input feature의 shape을 ConvLSTM 내부 Layer Shape에도 맞춰줘야함.
        # batch, sequence, channel, h, w
        # feature1 shape: 16, 10, 256, 16, 16
        # feature2 shape: 16, 10, 512, 8, 8
        # feature3 shape: 16, 10, 1024, 4, 4
        # feature4 shape: 16, 10, 2048, 2, 2
        output_1 = self.sequential1(feature_1)
        output_2 = self.sequential2(feature_2)
        output_3 = self.sequential3(feature_3)
        output_4 = self.sequential4(feature_4)

        # feature map's Return only the last output frame
        output_1 = self.conv1(output_1[:, :, -1])
        output_2 = self.conv2(output_2[:, :, -1])
        output_3 = self.conv3(output_3[:, :, -1])
        output_4 = self.conv4(output_4[:, :, -1])

        # output_last_feature map to obtain future map and concatenation
        return output_1, output_2, output_3, output_4


""" debug
if __name__ == "__main__":

    # debug

    # Input Image each sequence -> dim(batch size, time, channel, height, width)
    x = torch.rand(32, 10, 64, 128, 128)

    # model 인스턴스 생성자 선언
    # ConvLSTM Parameters: input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False
    # * ConvLSTM()은 뭘하는지? ConvLSTM안에 ConvLSTMCell 클래스는 뭘하는지?
    model = Seq2Seq(64, 16, (3, 3), 1, True, False, False)

    # input x를 투입하여 model layer를 거친 후 output 확인
    _, last_layer = model(x)

    # 최종 hidden layer, cell layer
    hidden = last_layer[0][0]
    cell = last_layer[0][1]
    
----------------------------------------------------------------------------------------------------------------

# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=1e-4)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')
    
"""

#--------------- add_module loop문으로 작성해봤는데 안되었다.
# 하나의 sequence instance로 모두 저장하게되면 forward할때 feature별로 별도로 투입할 수 없음.
# 추후에 하나의 sequence로 여러 input을 받아 layer를 수행할 수 있도록 작성해봐야겠다.

# # add first layer
# for i in range(len(convLSTM_parameters_list)):
#     self.sequential.add_module(
#         f"convlstm1_feature_{i+1}", ConvLSTM(
#             in_channels=convLSTM_parameters_list[0].get(f"feature_{i+1}")[0],
#             out_channels=convLSTM_parameters_list[0].get(f"feature_{i+1}")[1],
#             kernel_size=convLSTM_parameters_list[0].get(f"feature_{i+1}")[2],
#             padding=convLSTM_parameters_list[0].get(f"feature_{i+1}")[3],
#             activation=convLSTM_parameters_list[0].get(f"feature_{i+1}")[4],
#             frame_size=convLSTM_parameters_list[0].get(f"feature_1{i+1}")[5]
#         )
#     )
#     self.sequential.add_module(
#         f"batchnorm_conv1_feature_{i+1}", nn.BatchNorm3d(num_features=convLSTM_parameters_list[0].get(f"feature_{i+1}")[1])
#     )
#     for j in range(len(num_layers)):
#         self.sequential.add_module(
#             f"convlstm{j+2}_feature_{i+1}", ConvLSTM(
#                 in_channels=convLSTM_parameters_list[0].get(f"feature_{i+1}")[1],
#                 out_channels=convLSTM_parameters_list[0].get(f"feature_{i+1}")[1],
#                 kernel_size=convLSTM_parameters_list[0].get(f"feature_{i+1}")[2],
#                 padding=convLSTM_parameters_list[0].get(f"feature_{i+1}")[3],
#                 activation=convLSTM_parameters_list[0].get(f"feature_{i+1}")[4],
#                 frame_size=convLSTM_parameters_list[0].get(f"feature_{i+1}")[5]
#             )
#         )
#         self.sequential.add_module(
#             f"batchnorm_conv{j+2}_feature_{i+1}", nn.BatchNorm3d(num_features=convLSTM_parameters_list[0].get(f"feature_{i+1}")[1])
#         )