import os
from typing import Any, Dict, Optional
import label_mapping
import cv2 
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class NIH(Dataset):
    """ NIH Dataset

    Args:
        csv_file (string): Path to the csv file containing the dataset.
        root_dir (string): Path to the root directory of the dataset.
        transform (Dict, optional): Optional transform to be applied on a sample.
        
    Returns:
        Dict: A dictionary containing the image, its labels, follow-up number, patient ID, patient age, patient gender, 
        view position, original image width, original image height, and original image pixel spacing. 
    """
    
    def __init__(self, csv_file:str, root_dir:str,label_mapping, transform:Optional[Dict[str,Any]]=None,split='train'):
        self.root_dir = root_dir
        self.dataframe = pd.read_csv(csv_file)
        self.split = split
        
        if transform: 
            self.transform = transform 
        else: 
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 이미지 크기를 256x256으로 조정
                transforms.ToTensor()
            ])
        self.label_mapping = label_mapping
       

    def __getitem__(self, index:int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_path = os.path.join(self.root_dir, row['Image Index'])
        image = Image.open(image_path).convert('RGB')


        if self.transform:
            image = self.transform(image)
        labels = row['Finding Labels'].split('|')
        mapped_labels = [self.label_mapping[label] for label in labels]#
        follow_up = row['Follow-up #']
        patient_id = row['Patient ID']
        Patient_Age = row['Patient Age']
        Patient_Gender = row['Patient Gender']
        View_Position = row['View Position']
        original_image_width = row['OriginalImage[Width']
        original_image_height = row['Height]']
        original_image_pixel_spacing_x = row['OriginalImagePixelSpacing[x']
        original_image_pixel_spacing_y = row['y]']

        return {
        'image': image, 
        'label': mapped_labels,
        'split': self.split,
        'follow_up': follow_up, 
        'patient_id': patient_id,
        'patient_age': Patient_Age,
        'patient_gender': Patient_Gender,
        'view_position': View_Position,
        'original_image_width': original_image_width,
        'original_image_height': original_image_height,
        'original_image_pixel_spacing_x': original_image_pixel_spacing_x,
        'original_image_pixel_spacing_y': original_image_pixel_spacing_y,
    }
    
    def __len__(self):
        return len(self.dataframe)


    
