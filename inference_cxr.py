from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import NIHResNet

# _WEIGHTS_PATH = '/opt/goat_41/dummy.pth'
_WEIGHTS_PATH = '/opt/goat_41/cvn_dummy.pth'


@torch.no_grad()
def inference(path: str, device: int = 0):
    path = str(path) if not isinstance(path, str) else path

    # binary_model = NIHResNet(name='resnet50', num_classses=1)
    binary_model = NIHResNet(name='convnext_tiny_384_in22ft1k', num_classses=1)
    binary_model.load_state_dict(torch.load(_WEIGHTS_PATH), strict=False)

    device = f'cuda:{int(device)}' if isinstance(device, int) else "cpu"
    binary_model.eval().to(device)

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        np.expand_dims(image, axis=-1)

    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    output: torch.Tensor = binary_model.model(image.to(device))[0]  # 1 1 H W
    output = output.sigmoid()
    output = F.interpolate(output, size=(512, 512),
                           mode='bilinear', align_corners=True)

    output = output.squeeze(0).detach().cpu().numpy()

    # for debugging
    out_path = Path('test') / Path(path).name
    # cv2.imwrite(str(out_path), output)
    return output
