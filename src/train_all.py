import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Accuracy, Recall, Specificity
from tqdm import tqdm

from src.componenets.nih import NIH, SimpleNIH
from src.datamodules.datamodule import get_NIH_dataloader
from src.models.resnet import NIHResNet

_FORMAT_PER_ITER = "epoch[{num_epoch:03d}]  iter[{num_iter:5d}][{progress:3.0f}%] | loss {loss:.4f} | lr {lr:.5f}"
_FORMAT_COMPUTED = "Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}"


def forward_with_batch(batch: dict[str, torch.Tensor],
                       model: nn.Module,
                       device: str) -> tuple[torch.Tensor, torch.Tensor]:
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
                                device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, labels = forward_with_batch(batch, model, device)
    loss = backward_with_batch(logits, labels, optimizer, criterion)
    probs = torch.sigmoid(logits)
    return logits, probs, labels, loss


def update_meters(logits: torch.Tensor, targets: torch.Tensor,
                  meters: list[torchmetrics.Metric]):
    meters = meters if isinstance(meters, list) else [meters]
    for idx_lesion, meter in enumerate(meters):
        p = logits[:, idx_lesion]
        t = targets[:, idx_lesion]
        meter[idx_lesion].update(p, t)


def update_multi_meters(logits: torch.Tensor, targets: torch.Tensor,
                        *args: list[torchmetrics.Metric]):
    for meter in args:
        update_meters(logits, targets, meter)


def reset_meters(meters: list[torchmetrics.Metric]):
    meters = meters if isinstance(meters, list) else [meters]
    for meter in meters:
        meter.reset()


def reset_multi_meters(*args: torchmetrics.Metric):
    for meter in args:
        reset_meters(meter)


def compute_meters(meters: list[torchmetrics.Metric]) -> list[float]:
    meters = meters if isinstance(meters, list) else [meters]
    computed = []
    for meter in meters:
        computed.append(meter.compute().item())
    return computed


def compute_multi_meters(*args: torchmetrics.Metric) -> list[list[float]]:
    computed = []
    for meter in args:
        computed.append(compute_meters(meter))
    return computed


def save_model(model: nn.Module, save_dir: str):
    model = model.cpu().eval()
    torch.save(model.state_dict(), save_dir)


def load_model(model: nn.Module, load_dir: str):
    model.load_state_dict(torch.load(load_dir,
                                     map_location='cpu'),
                          strict=True)


@torch.no_grad()
def validate(loader,
             model: nn.Module,
             meters: list[torchmetrics.Metric],
             device: str):
    model.eval()
    model.return_logits = True
    for meter in meters:
        meter.reset()

    for batch in tqdm(loader):
        probs, labels = forward_with_batch(batch, model, device)
        update_multi_meters(probs, labels, *meters)

    computed = compute_multi_meters(*meters)
    print(
        f"Validation Acc: {computed[0]}, Sens: {computed[1]}, Spec: {computed[2]}")
    return computed


def train(root_dir: str,
          batch_size: int = 4,
          model_name: str = 'resnet50',
          device: int = 0,
          max_epoch: int = 100,
          image_size: int = 512,
          save_dir: str = None):

    # get NIH - dataset
    trainset = SimpleNIH(root_dir=root_dir, split='train',
                         image_size=image_size, image_channels=3)
    testset = SimpleNIH(root_dir=root_dir, split='test',
                        image_size=image_size, image_channels=3)
    validset = SimpleNIH(root_dir=root_dir, split='val',
                         image_size=image_size, image_channels=3)

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
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50, 70],
                                                     gamma=0.1)

    accs = Accuracy(task='binary', num_classes=1, threshold=0.5).to(device)
    sens = Recall(task='binary', num_classes=1, threshold=0.5).to(device)
    spec = Specificity(task='binary', num_classes=1, threshold=0.5).to(device)
    accs_per_lesion = [accs.clone() for _ in range(num_classes)]
    sens_per_lesion = [sens.clone() for _ in range(num_classes)]
    spec_per_lesion = [spec.clone() for _ in range(num_classes)]

    best_acc = -1

    # epoch
    for num_epoch in range(max_epoch):
        # set model to train mode
        model.train()
        model.return_logits = True
        reset_multi_meters(accs, sens, spec)

        # iteration
        for num_iter, batch in enumerate(loader_train):
            # forward & backward with batch
            _, probs, labels, loss = forward_backward_with_batch(batch, model,
                                                                 optimizer, criterion,
                                                                 device)
            # update meters
            update_multi_meters(probs, labels,
                                accs_per_lesion,
                                sens_per_lesion,
                                spec_per_lesion)

            if num_iter % 10 == 0:
                computed_all = compute_multi_meters(accs_per_lesion,
                                                    sens_per_lesion,
                                                    spec_per_lesion)
                progress = num_iter / len(loader_train) * 100
                learning_rate = optimizer.param_groups[0]['lr']
                print(_FORMAT_PER_ITER.format(num_epoch=num_epoch,
                                              num_iter=num_iter,
                                              progress=progress,
                                              loss=loss.item(),
                                              lr=learning_rate), end="  ")
                print(_FORMAT_COMPUTED.format(acc=computed_all[0][0],
                                              sens=computed_all[1][0],
                                              spec=computed_all[2][0]))

        # epoch end
        scheduler.step()

        # validate
        computed = validate(loader_valid, model, [accs, sens, spec], device)
        if best_acc < computed[0]:
            best_acc = computed[0]
            save_model(model, 'best.pth')
