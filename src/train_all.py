from .componenets.nih import NIH
from .datamodules.datamodule import get_NIH_dataloader
from .models.resnet import NIHResNet

import torch
import torch.nn as nn


def train(root_dir: str, batch_size: int = 4, model_name: str = 'resnet50', device: int = 0, max_epoch: int = 100):

    # get NIH - dataset
    trainset = NIH(root_dir=root_dir, split='train',
                   image_size=512, image_channels=3)
    testset = NIH(root_dir=root_dir, split='test',
                  image_size=512, image_channels=3)
    validset = NIH(root_dir=root_dir, split='valid',
                   image_size=512, image_channels=3)

    loader_train = get_NIH_dataloader(trainset, batch_size=batch_size,
                                      use_basic=True)
    loader_valid = get_NIH_dataloader(validset, batch_size=batch_size,
                                      use_basic=True)
    loader_test = get_NIH_dataloader(testset, batch_size=batch_size,
                                     use_basic=True)

    # get model
    model = NIHResNet(name=model_name, num_classs=5,
                      out_indices=[4], return_logits=True)

    # device setting
    device = f'cuda:{device}'
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = nn.optim.RMSprop(model.parameters(), lr=0.001)

    for i in range(max_epoch):  # epoch
        for j, batch in enumerate(loader_train):
            image: torch.Tensor = batch['image']
            labels: torch.Tensor = batch['labels']
            image = image.to(device)
            labels = labels.to(device)

            logits = model(image)

            loss: torch.Tensor = criterion(logits, labels)

            print(f'iteration {i:5d} loss: {loss.item():.4f}')

            loss.backward()

            if i == 10:
                break
