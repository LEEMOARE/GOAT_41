import torch
import torch.nn as nn
from torchmetrics import Recall, Specificity
import os
from src.componenets.nih import NIH
from src.datamodules.datamodule import get_NIH_dataloader
from src.models.resnet import NIHResNet


_FORMAT_PER_ITER = '''
epoch {num_epoch:03d}\titeration {num_iter:5d} {num_iter/len(loader_train)*100:.0f}\tloss: {loss.item():.4f}
'''


def forward_with_batch(batch: dict[str, torch.Tensor],
                       model: nn.Module,
                       device: str) -> tuple(torch.Tensor, torch.Tensor):
    image: torch.Tensor = batch['image']
    labels: torch.Tensor = batch['label']
    image = image.to(device)
    labels = labels.to(device)
    logits = model(image)
    return logits, labels


def backward_with_batch(logits: torch.Tensor,
                        labels: torch.Tensor,
                        optimizer: torch.optim.Optimizer,
                        criterion: torch.nn.Module) -> torch.Tensor:
    optimizer.zero_grad()
    loss: torch.Tensor = criterion(logits, labels.to(torch.float32))
    loss.backward()
    optimizer.step()
    return loss


def forward_backward_with_batch(batch: dict[str, torch.Tensor],
                                model: nn.Module,
                                optimizer: torch.optim.Optimizer,
                                criterion: torch.nn.Module,
                                device: str) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    logits, labels = forward_with_batch(batch, model, device)
    loss = backward_with_batch(logits, labels, optimizer, criterion)
    return logits, labels, loss


def train(root_dir: str,
          batch_size: int = 4,
          model_name: str = 'resnet50',
          device: int = 0,
          max_epoch: int = 100,
          save_dir: str = None):

    # get NIH - dataset
    trainset = NIH(root_dir=root_dir, split='train',
                   image_size=512, image_channels=3)
    testset = NIH(root_dir=root_dir, split='test',
                  image_size=512, image_channels=3)
    validset = NIH(root_dir=root_dir, split='val',
                   image_size=512, image_channels=3)

    num_classes = trainset.num_classes

    loader_train = get_NIH_dataloader(trainset, batch_size=batch_size,
                                      use_basic=True)
    loader_valid = get_NIH_dataloader(validset, batch_size=batch_size,
                                      use_basic=True)
    loader_test = get_NIH_dataloader(testset, batch_size=batch_size,
                                     use_basic=True)

    # device setting
    device = f'cuda:{device}'
    # get model
    model = NIHResNet(name=model_name, num_classs=num_classes,
                      out_indices=[4], return_logits=True).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50, 70],
                                                     gamma=0.1)

    sens = Recall(task='binary', num_classes=1).to(device)
    spec = Specificity(task='binary', num_classes=1).to(device)
    sens_per_lesion = [sens.clone() for _ in range(num_classes)]
    spec_per_lesion = [spec.clone() for _ in range(num_classes)]

    # epoch
    for num_epoch in range(max_epoch):
        # set model to train mode
        model.train()
        model.return_logits = True

        # iteration
        for num_iter, batch in enumerate(loader_train):
            # forward & backward with batch
            logits, labels, loss = forward_backward_with_batch(batch, model,
                                                               optimizer, criterion,
                                                               device)
            # evaluation with batch
            for idx_lesion in range(num_classes):
                p, t = logits[:, idx_lesion], labels[:, idx_lesion]
                sens_per_lesion[idx_lesion].update(p, t)
                spec_per_lesion[idx_lesion].update(p, t)

            computed_sens = [sens_per_lesion[idx_lesion].compute()
                             for idx_lesion in range(num_classes)]
            computed_spec = [spec_per_lesion[idx_lesion].compute()
                             for idx_lesion in range(num_classes)]

            if num_iter % 10 == 0:
                print(_FORMAT_PER_ITER.format(num_epoch=num_epoch,
                                              num_iter=num_iter,
                                              loss=loss.item()))

                print(["{:.4f}".format(sens.item()) for sens in computed_sens])
                print(["{:.4f}".format(spec.item()) for spec in computed_spec])

        # epoch end
        for _sens, _spec in zip(sens_per_lesion, spec_per_lesion):
            _sens.reset()
            _spec.reset()

        model.eval()
        model.return_logits = False
        best_accuracy = 0.0  # 초기 최고 정확도
        with torch.no_grad():
            correct = 0
            total = 0
            for num_iter, batch in enumerate(loader_valid):
                probs, labels = forward_with_batch(batch, model, device)
                # Generate predictions based on the threshold
                predicted = (probs.ge(0.5)).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Use probs instead of outputs
                print(f'Outputs for iteration {num_iter}: {probs}')

                for idx_lesion in range(num_classes):
                    sens_per_lesion[idx_lesion].update(
                        probs[:, idx_lesion], labels[:, idx_lesion])  # Use probs instead of outputs
                    spec_per_lesion[idx_lesion].update(
                        probs[:, idx_lesion], labels[:, idx_lesion])  # Use probs instead of outputs

                computed_sens = [sens_per_lesion[idx_lesion].compute().item()
                                 for idx_lesion in range(num_classes)]
                computed_spec = [spec_per_lesion[idx_lesion].compute().item()
                                 for idx_lesion in range(num_classes)]

                # Print additional information for each 10 iterations
                if num_iter % 10 == 0:
                    print(
                        f'Validation iteration {num_iter:5d} {num_iter/len(loader_valid)*100:.0f}')
                    print(f'Predicted: {predicted}')
                    print(f'Labels: {labels}')
                    print(["{:.6f}".format(sens) for sens in computed_sens])
                    print(["{:.6f}".format(spec) for spec in computed_spec])
                    print('')

            accuracy = correct / total
            print(f'Accuracy on validation set: {accuracy:.4f}')

            # 모델 저장
            save_dir = "/home/ubuntu/miniforge3/etc/GOAT_41/GOAT_41" if save_dir is None else save_dir
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print('Best model saved with accuracy {:.4f}'.format(
                    best_accuracy))
                model.train()

                scheduler.step()
