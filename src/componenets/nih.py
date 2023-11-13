from typing import Any, Dict, Optional
from pathlib import Path

import cv2
from torch.utils.data import Dataset

from ..utils.common import get_repo_root, load_json



class NIH(Dataset):
    """ NIH Dataset

    Args:
        root_dir (string): Path to the root directory of the dataset.
        split: (strinf) : train, test, or valid
        transform (callable, optional): Optional transform to be applied on a sample.
        
    Returns:
        Dict: A dictionary containing the image, its labels, follow-up number, patient ID, patient age, patient gender, 
        view position, original image width, original image height, and original image pixel spacing. 
    """
    
    def __init__(self, 
                 root_dir:str, 
                 split:str='train',
                 transform:Optional[Dict[str,Any]]=None,
                 image_channels:int=3,
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_channels = image_channels

        self.annots = self._load_annotations()
       

    def __getitem__(self, index:int) -> Dict[str, Any]:
        annot = self._load_data(index)
        
        return annot

    def __len__(self):
        return len(self.annots)


    def _load_annotations(self):
        path = Path(get_repo_root()) / 'data/nih.zip'
        return load_json(path)
    
    def _load_data(self, idx) -> Dict[str, Any]:
        annot = self.annots[idx]
        filename = annot['filename']
        gender = annot['gender']
        age = annot['age']
        view_position = annot['view_position']
        label_names = annot['label_names']
        label_indexes = annot['label_indexes']

        image_path = Path(self.root_dir) / filename
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        if self.image_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return {'image':image,
                'path': image_path,
                'gender': gender, 
                'age': age,
                'view_position': view_position,
                'label_names': label_names,
                'label_indexes': label_indexes}


        


    
