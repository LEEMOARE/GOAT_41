import os
from typing import Any, Dict, Optional
from label_mapping import label_mapping
import cv2 
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from NIH import NIH


# Create training and validation datasets
train_dataset = NIH(root_dir="C:\\Users\\gjaischool\\Desktop\\archive(1)\\image",label_mapping=label_mapping, csv_file="C:\\Users\\gjaischool\\Desktop\\goat_41\\src\\componenets\\nih.csv", split='train')
val_dataset = NIH(root_dir="C:\\Users\\gjaischool\\Desktop\\archive(1)\\image",label_mapping=label_mapping, csv_file="C:\\Users\\gjaischool\\Desktop\\goat_41\\src\\componenets\\nih.csv", split='val')

first_train_data=train_dataset[1]
first_val_data=val_dataset[1]

print(first_train_data)
print(first_val_data)