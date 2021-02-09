import torch
import torch.nn.functional as F
from torch import nn

from mobilenetv3 import mobilenetv3_large


def get_mobilenet_model(n_channels, pretrained_path=None):
    """Load MobilenetV3 model with specified in and out channels"""
    model = mobilenetv3_large()
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))

    if n_channels == 1:
        model.features[0][0].weight.data = torch.sum(
            model.features[0][0].weight.data, dim=1, keepdim=True
        )
    elif n_channels == 2:
        model.features[0][0].weight.data = model.features[0][0].weight.data[:, :2]
    model.features[0][0].in_channels = n_channels

    return model


class TorqueModel(nn.Module):
    def __init__(self, out_features_conv, out_features_dense, mid_features, n_channels, pretrained_path=None):
        super(TorqueModel, self).__init__()
        self.mnet = get_mobilenet_model(n_channels, pretrained_path)
        self.fc1 = nn.Linear(out_features_conv + out_features_dense, mid_features)
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.fc3 = nn.Linear(mid_features, 1)

    def forward(self, image, data):
        x1 = self.mnet(image)
        x2 = data
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
