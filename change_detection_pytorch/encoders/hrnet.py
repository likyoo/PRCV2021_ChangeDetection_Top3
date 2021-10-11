from copy import deepcopy

import torch.nn as nn

from pretrainedmodels.models.torchvision_models import pretrained_settings
import yaml
from change_detection_pytorch.V2.backbones.hrnet import HighResolutionNet
from change_detection_pytorch.encoders._base import EncoderMixin


class HRNetEncoder(HighResolutionNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

    def forward(self, x):
        features = [x, ]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        features.extend(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)


hrnet_weights = {
    'hrnet_w18_small': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v1-f460c6bc.pth'},
    'hrnet_w18_small_v2': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v2-4c50a8cb.pth'},
    'hrnet_w18': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w18-8cb57bb9.pth'},
    'hrnet_w30': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w30-8d7f8dab.pth'},
    'hrnet_w32': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w32-90d8c5fb.pth'},
    'hrnet_w40': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w40-7cd397a4.pth'},
    'hrnet_w44': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w44-c9ac8c18.pth'},
    'hrnet_w48': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w48-abd2e6ab.pth'},
    'hrnet_w64': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w64-b47cc881.pth'},
}


pretrained_settings = {}
for model_name, sources in hrnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }

hrnet_encoders = {
    "hrnet_w18": {
        "encoder": HRNetEncoder,
        "pretrained_settings": pretrained_settings['hrnet_w18'],
        "params": {
            "out_channels": (3, 64, 18, 36, 72, 144),
            "config": yaml.load(open("change_detection_pytorch/V2/configs/hrnet_w18.yaml"), Loader=yaml.FullLoader),
        },
    },
}

if __name__ == "__main__":
    import torch
    import yaml
    pretrained = False
    backbone = "hrnet_w18"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(1, 3, 256, 256).to(device)

    cfg = yaml.load(open("../V2/configs/%s.yaml" % backbone), Loader=yaml.FullLoader)
    if not pretrained:
        cfg["MODEL"]["PRETRAINED"] = ""

    model = HRNetEncoder(1, config=cfg)
    model.init_weights(cfg["MODEL"]["PRETRAINED"])
    res = model.forward(input)