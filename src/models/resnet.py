import timm
import torch
import torch.nn as nn


class NIHResNet(nn.Module):
    def __init__(self, name: str, num_classs: int = 5, out_indices=[4], return_logits=False):
        super().__init__()
        assert name in ['resnet50', 'resnet101']
        self.model = timm.create_model(name,
                                       pretrained=True,
                                       features_only=True,
                                       out_indices=out_indices)

        self.header = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Conv2d(2048, num_classs, 1, 1, 0, bias=True)])

        self.return_logits = return_logits

        init_prob = 0.01
        self.initial_prob = torch.tensor((1.0 - init_prob) / init_prob)
        nn.init.constant_(self.conv[-1].bias, -torch.log(self.initial_prob))

    def forward(self, x):
        features = self.model(x)[-1]  # last feature map (B 2048 H/32 )
        logits = self.header(features)
        return logits if self.return_logits else torch.sigmoid(logits)
