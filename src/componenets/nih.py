from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from pandas import read_csv

class NIH(Dataset):
    """ NIH Dataset

    Args:
        root_dir (string): Path to the root directory of the dataset.
        transform (Dict, optional): Optional transform to be applied on a data.
        split (str): Split the dataset into train and test. (default: 'train')

    Returns:
        Dict: A dictionary containing the data and its label. 
    """
    def __init__(self, root_dir:str, transform:Optional[Dict[str,Any]]=None, split:str='train'):
        self.root_dir = root_dir
        self.transform = transform

        self.dataframe = read_csv("/tmp/NIH.csv")


        self.train = True if split=='train' else False 
        # if split == 'train':
        #     self.train:bool = True
        # else:
        #     self.train = False

        a = [1,2,3,4,5] 
        a = [i for i in range(1,6)] # list comprehension
        

    def __getitem__(self, index:int) -> Dict[str, Any]:
        # index를 입력받아서 해당 인덱스에 대한 결과를 리턴하는 함수입니다.
        dummy = self.dataframe.iloc[index]
        path = dummy['path']
        image = cv2.imread(path)

        if self.train:
            # to do something 
            pass
        else:
            # to do something for valid or testset
            pass

        label = dummy['label']
        return {'image': image, 'label': label}
    
    def __len__(self):
        # (essential) 해당 데이터셋의 길이를 반환하는 함수입니다.
        return 0