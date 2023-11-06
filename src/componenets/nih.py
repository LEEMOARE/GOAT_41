import os
from typing import Any, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

from sklearn.model_selection import train_test_split

class NIH(Dataset):
    """ NIH Dataset

    Args:
        csv_file (string): Path to the csv file containing the dataset.
        root_dir (string): Path to the root directory of the dataset.
        transform (Dict, optional): Optional transform to be applied on a sample.
        split (str): Split the dataset into train and test. (default: 'train')
        random_state (int): The seed used by the random number generator for shuffling the dataset.

    Returns:
        Dict: A dictionary containing the image, its label, and its split. 
    """
    
    def __init__(self, csv_file:str, root_dir:str, transform:Optional[Dict[str,Any]]=None, split:str='train', random_state:int=42):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.split = split
        dataframe = pd.read_csv(csv_file)

        # 데이터프레임을 훈련 데이터프레임과 테스트 데이터프레임으로 나눕니다.
        train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.3, random_state=random_state)

        # split 매개변수에 따라 self.dataframe를 설정합니다.
        if self.split == 'train':
            self.dataframe = train_dataframe
        else:
            self.dataframe = test_dataframe

    def __getitem__(self, index:int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_path = os.path.join(self.root_dir, row['Image Index'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = row['Finding Labels']
        return {'image': image, 'label': label, 'split': self.split}
    
    def __len__(self):
        return len(self.dataframe)


    