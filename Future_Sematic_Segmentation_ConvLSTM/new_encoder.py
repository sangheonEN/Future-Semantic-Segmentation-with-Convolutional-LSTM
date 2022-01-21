import torch.nn as nn
from backbone import resnet

class Encoder(nn.Module):
    def __init__(self, class_num):
        super(Encoder, self).__init__()
        self.model = resnet.resnet50(pretrained=True, class_num=class_num)

    def forward(self, inputs):
        feature_1, feature_2, feature_3, feature_4 = self.model(inputs)

        return feature_1, feature_2, feature_3, feature_4


