import torch
import torch.nn as nn
from torchmetrics import Recall, Specificity
import os
from .componenets.nih import NIH
from .datamodules.datamodule import get_NIH_dataloader
from .models.resnet import NIHResNet


def train(root_dir: str, batch_size: int = 4, model_name: str = 'resnet50', device: int = 0, max_epoch: int = 100):

    root_dir2 = "/home/ubuntu/miniforge3/etc/GOAT_41/GOAT_41"

    # get NIH - dataset
    trainset = NIH(root_dir=root_dir, split='train',
                   image_size=512, image_channels=3)
    testset = NIH(root_dir=root_dir, split='test',
                  image_size=512, image_channels=3)
    validset = NIH(root_dir=root_dir, split='valid',
                   image_size=512, image_channels=3)

    num_classes = trainset.num_classes

    loader_train = get_NIH_dataloader(trainset, batch_size=batch_size,
                                      use_basic=True)
    loader_valid = get_NIH_dataloader(validset, batch_size=batch_size,
                                      use_basic=True)
    loader_test = get_NIH_dataloader(testset, batch_size=batch_size,
                                     use_basic=True)

    # get model
    model = NIHResNet(name=model_name, num_classs=10,
                      out_indices=[4], return_logits=True)

    # device setting
    device = f'cuda:{device}'
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50, 70],
                                                     gamma=0.1)

    sens_per_lesion = [Recall(task='binary', num_classes=1).to(device)
                       for _ in range(num_classes)]
    spec_per_lesion = [Specificity(task='binary', num_classes=1).to(device)
                       for _ in range(num_classes)]
    


    for num_epoch in range(max_epoch):  
        for num_iter, batch in enumerate(loader_train):
            image: torch.Tensor = batch['image']
            labels: torch.Tensor = batch['label']
            image = image.to(device)
            labels = labels.to(device)

            logits = model(image)

            optimizer.zero_grad()

            loss: torch.Tensor = criterion(logits, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            for idx_lesion in range(num_classes):
                sens_per_lesion[idx_lesion].update(
                    logits[:, idx_lesion], labels[:, idx_lesion])
                spec_per_lesion[idx_lesion].update(
                    logits[:, idx_lesion], labels[:, idx_lesion])

            computed_sens = [sens_per_lesion[idx_lesion].compute().item()
                             for idx_lesion in range(num_classes)]
            computed_spec = [spec_per_lesion[idx_lesion].compute().item()
                             for idx_lesion in range(num_classes)]

            if num_iter % 10 == 0:
                print(
                    f'epoch {num_epoch:03d}\titeration {num_iter:5d} {num_iter/len(loader_train)*100:.0f}\tloss: {loss.item():.4f}')
                print(["{:.6f}".format(sens) for sens in computed_sens])
                print(["{:.6f}".format(spec) for spec in computed_spec])
        model.eval()
        with torch.no_grad():
                correct = 0
                total = 0
                for num_iter, batch in enumerate(loader_valid):
                    image: torch.Tensor = batch['image']
                    labels: torch.Tensor = batch['label']
                    image = image.to(device)
                    labels = labels.to(device)
                    outputs = model(image)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    for idx_lesion in range(num_classes):
                        sens_per_lesion[idx_lesion].update(
                            outputs[:, idx_lesion], labels[:, idx_lesion])
                        spec_per_lesion[idx_lesion].update(
                            outputs[:, idx_lesion], labels[:, idx_lesion])

                    computed_sens = [sens_per_lesion[idx_lesion].compute().item()
                                    for idx_lesion in range(num_classes)]
                    computed_spec = [spec_per_lesion[idx_lesion].compute().item()
                                    for idx_lesion in range(num_classes)]

                    if num_iter % 10 == 0:
                        print(
                            f'validation iteration {num_iter:5d} {num_iter/len(loader_valid)*100:.0f}')
                        print(["{:.6f}".format(sens) for sens in computed_sens])
                        print(["{:.6f}".format(spec) for spec in computed_spec])

                if total > 0:
                    accuracy = correct / total
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), os.path.join(root_dir2, 'best_model1.pth'))
                else:
                    print("Warning: No valid data found.")

        model.train()
            
        scheduler.step()

