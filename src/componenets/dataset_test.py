import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from nih import NIH  # nih.py 파일에서 NIH 클래스를 import
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 크기를 224x224로 변경
    transforms.ToTensor(),  # PyTorch 텐서로 변환
])


train_dataset = NIH(csv_file="C:\\Users\\gjaischool\\Desktop\\archive (1)\\Data_Entry_2017.csv", root_dir='C:\\Users\\gjaischool\\Desktop\\archive (1)\\image',transform=transform,split='train')
test_dataset = NIH(csv_file="C:\\Users\\gjaischool\\Desktop\\archive (1)\\Data_Entry_2017.csv", root_dir='C:\\Users\\gjaischool\\Desktop\\archive (1)\\image',transform=transform,split='test')

# 첫 번째 데이터 접근
first_train_data = train_dataset[56908]
first_test_data = test_dataset[5462]

print(first_train_data['image'])  # 이미지 데이터 출력
print(first_train_data['label'])  # 레이블 출력
 
print(first_test_data['image'])  # 이미지 데이터 출력
print(first_test_data['label'])  # 레이블 출력
