from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from pandas import read_csv

class NIH(Dataset):
    """ NIH Dataset

    Args:
        root_dir (string): Path to the root directory of the dataset.
        transform (Dict, optional): Optional transform to be applied
            on a data.

    Returns:
        Dict: A dictionary containing the data and its label. 
    """
    def __init__(self, root_dir:str, transform:Optional[Dict[str,Any]]=None , split:str='train'):
        self.root_dir = root_dir
        self.transform = transform

        self.dataframe = read_csv("/tmp/NIH.csv")


        

    def __getitem__(self, index:int) -> Dict[str, Any]:
        # load 0번쨰 image
        dummy = self.dataframe.iloc[index]
        path = dummy['path']
        image = cv2.imread(path)

        label = dummy['label']
        return {'image': image, 'label': label}
    
    def __len__(self):
        return 0