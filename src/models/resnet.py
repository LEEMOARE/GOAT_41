import timm
import torch
import torch.nn as nn


class NIHResNet(nn.Module):
    def __init__(self, name: str, num_classses: int = 10, return_logits=False):
        super().__init__()
        assert name in ['resnet50', 'resnet101', 'convnext_tiny_384_in22ft1k']

        out_indices = [3] if 'convnext' in name else [4]
        self.model = timm.create_model(name,
                                       pretrained=True,
                                       features_only=True,
                                       out_indices=out_indices)
        out_channels = self.model.feature_info.channels()[0]
        self.header = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Conv2d(out_channels, num_classses, 1, 1, 0, bias=True)])

        self.return_logits = return_logits

        init_prob = 0.01
        self.initial_prob = torch.tensor((1.0 - init_prob) / init_prob)
        nn.init.constant_(self.header[-1].bias, -torch.log(self.initial_prob))

    def forward(self, x):
        # last feature map (B out_channels H/32 W/32 )
        features = self.model(x)[-1]
        logits = self.header(features)  # B, num_classes, 1, 1
        B, C, _, _ = logits.shape
        logits = logits.view(B, C)
        return logits if self.return_logits else torch.sigmoid(logits)
