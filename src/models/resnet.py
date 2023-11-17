import torch
import torch.nn as nn
import timm


class NIHResNet(nn.Module):
    def __init__(self, name: str, num_classs: int = 5, out_indices=[4]):
        super().__init__()
        assert name in ['resnet50', 'resnet101']
        self.model = timm.create_model(name,
                                       pretrained=True,
                                       features_only=True,
                                       out_indices=out_indices)

        self.header = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Conv2d(2048, num_classs, 1, 1, 0)])

    def forward(self, x):
        return self.model(x)
