from typing import Any, Dict, Optional,List, Union, Tuple
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from ..utils.common import get_repo_root, load_json

_MAPPER = {'effusion': 1,
           'emphysema': 2,
           'atelectasis': 3,
           'edema': 4,
           'consolidation': 5, 
           'pleural_thickening': 6,
           'hernia': 7,
           'mass': 8,
           'nofinding': 9,
           'cardiomegaly': 10,
           'nodule': 11,
           'pneumothorax': 12,
           'pneumonia': 13,
           'fibrosis': 14,
           'infiltration': 15}

class NIH(Dataset):
    """ NIH Dataset

    Args:
        root_dir (string): Path to the root directory of the dataset.
        split: (string) : train, test, or valid
        transform (Dict[str, Any]): A dictionary containing the image transformation.
        image_size (int or tuple): The size of the image. If int, a square image is returned. If tuple, 
                                    the image is resized to the size specified by the tuple.
        image_channels (int): 1 for grayscale, 3 for RGB
        
    Returns:
        Dict: A dictionary containing the image, its labels, follow-up number, patient ID, patient age, patient gender, 
        view position, original image width, original image height, and original image pixel spacing. 
    """
    
    def __init__(self, 
                 root_dir:str, 
                 split:str='train',
                 transform:Optional[Dict[str,Any]]=None,
                 image_size:Union[int, Tuple[int,int]]=(1024,1024),
                 image_channels:int=3,
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.image_channels = image_channels

        self.annots = self._load_annotations()
       

    def __getitem__(self, index:int) -> Dict[str, Any]:
        annot = self._load_data(index)
        image: np.ndarray = annot['image'] # (H,W,3 or 1)
        path:str = annot['path']
        labels:List[int] = annot['labels']

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        image = torch.Tensor(image).permute(2,0,1) # (H,W,3 or 1) -> (3 or 1, H, W)
        labels = torch.Tensor(labels).long()

        return {'image':image,
                'label': labels,
                'path': path}

    def __len__(self) -> int:
        return len(self.annots)


    def _load_annotations(self):
        path = Path(get_repo_root()) / 'data/nih.zip'
        annots = load_json(path)
        return [annot for annot in annots if annot['split'] == self.split]

    
    def _load_data(self, idx) -> Dict[str, Any]:
        annot = self.annots[idx]
        filename = annot['filename']
        gender = annot['gender']
        age = annot['age']
        view_position = annot['view_position']
        label_names = annot['label_names']
        label_indexes = annot['label_indexes']

        image_path = Path(self.root_dir) / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2) # (H,W) -> (H,W,1)
    
        if self.image_channels == 3 and image.shape[2] == 1: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # (H,W,1) -> (H,W,3)

        # min-max normalization 
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # to 8bit image 
        image = image * 255.0
        image = image.astype(np.uint8)

        return {'image':image,
                'path': image_path,
                'gender': gender, 
                'age': age,
                'view_position': view_position,
                'label_names': label_names,
                'labels': label_indexes}


        


    
