import os
from typing import Any, Dict, Optional
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

_MAPPER = {'Effusion': 1,
           'Emphysema': 2,
           'Atelectasis': 3,
           'Edema': 4,
           'Consolidation': 5, 
           'Pleural_Thickening': 6,
           'Hernia': 7,
           'Mass': 8,
           'No Finding': 9,
           'Cardiomegaly': 10,
           'Nodule': 11,
           'Pneumothorax': 12,
           'Pneumonia': 13,
           'Fibrosis': 14,
           'Infiltration': 15}


class NIH(Dataset):
    """ NIH Dataset

    Args:
        dataframe (pandas.DataFrame): A DataFrame containing the annotations. Each row in the DataFrame represents an image and contains information such as the patient ID and the labels.
        root_dir (string): Path to the root directory of the dataset.
        label_mapping (Dict): A dictionary mapping the labels in the dataframe to the desired output labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        
    Returns:
        Dict: A dictionary containing the image, its labels, follow-up number, patient ID, patient age, patient gender, 
        view position, original image width, original image height, and original image pixel spacing. 
    """
    
    def __init__(self, dataframe:str, root_dir:str,label_mapping, transform:Optional[Dict[str,Any]]=None,split='train'):
        self.root_dir = root_dir
        self.dataframe = dataframe
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  
       


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


    
