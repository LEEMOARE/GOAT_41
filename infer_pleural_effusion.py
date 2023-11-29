import copy

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import src.utils.common as uc
from src.models.resnet import NIHResNet

_WEIGHTS_PATH = '/opt/goat_41/resnet50_1_12.pth'


class PleuralEffusion():
    def __init__(self, model=NIHResNet, weights=_WEIGHTS_PATH,
                 device: int = 0, image_size: int = 512, threshold: float = 0.5):
        model_name, lesions = uc.get_info_from_name(weights)
        self.model_name = model_name
        self.num_classes = len(lesions)
        self.image_size = image_size
        self.device = f'cuda:{int(device)}'
        self.model = model(name=self.model_name, num_classses=self.num_classes)
        uc.load_weights(self.model, weights)

        self.model.return_logits = False
        self.threshold = threshold
        # for GradCAM
        self.model = self.model.eval().to(self.device)
        target_layers = self.model.model.layer4
        self.cam = GradCAM(model=self.model,
                           target_layers=target_layers, use_cuda=True)
        self.targets = [ClassifierOutputTarget(0)]

    # @torch.no_grad() # for GradCAM

    def run(self, input: str):
        image_origin = uc.load_image(input, image_size=self.image_size)
        image = uc.cvt_image_to_tensor(image_origin).to(self.device)

        actmap = self.cam(input_tensor=image, targets=self.targets)[0, :]

        prob = self.model(image)[0]
        if len(prob.shape) > 0:
            prob = prob[0]
        prob = prob.item()
        draw, mask = uc.post_process_cam(
            prob, actmap, image_origin, self.threshold)
        return {'image': image_origin,
                'probabilty': prob,
                'processed_activation_map': mask,
                'overlay_image': draw}
