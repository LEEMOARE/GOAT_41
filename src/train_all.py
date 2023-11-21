import torch
import torch.nn as nn
import torchmetrics
from .componenets.nih import NIH
from .datamodules.datamodule import get_NIH_dataloader
from .models.resnet import NIHResNet


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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    valid_losses = [] # 검증 손실값을 저장할 리스트

    accuracy = torchmetrics.Accuracy() # 정확도 계산을 위한 객체

    for i in range(max_epoch):  # epoch 
        model.train() # 학습 모드로 설정
        for j, batch in enumerate(loader_train):  
            image: torch.Tensor = batch['image'].to(device)
            labels: torch.Tensor = batch['labels'].to(device)
            optimizer.zero_grad() # 기울기 초기화
            logits = model(image) 
            loss: torch.Tensor = criterion(logits, labels)
            print(f'Training - Epoch {i:5d} iteration {j:5d} loss: {loss.item():.4f}')
            loss.backward() # 기울기 계산
            optimizer.step() # 가중치 갱신

            # if i == 10:
            #     break
        model.eval() # 검증 모드로 설정
        
        with torch.no_grad(): # 검증 단계에는 기울기를 계산하지 않음
            for batch in loader_valid:
                image: torch.Tensor = batch['image'].to(device)
                labels: torch.Tensor = batch['labels'].to(device)
                logits = model(image)
                loss = criterion(logits, labels)
                valid_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                acc = accuracy(preds, labels)
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Validation - Epoch {i:5d} valid loss: {avg_valid_loss:.4f}, accuracy: {acc:.4f}")
        scheduler.step() # 학습률 업데이트



    test_losses = [] # 테스트 손실값을 저장할 리스트
    model.eval() # 모델 테스트
    with torch.no_grad():
        for batch in loader_test:
            image: torch.Tensor = batch['image'].to(device)
            labels: torch.Tensor = batch['labels'].to(device)
            logits = model(image)
            loss = criterion(logits, labels)
            test_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, labels)
            
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f'Testing - Test Loss: {avg_test_loss:.4f}, Accuracy: {acc:.2f}')
        
