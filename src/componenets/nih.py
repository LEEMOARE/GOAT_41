from typing import Any, Dict, Optional

from torch.utils.data import Dataset


class NIH(Dataset):
    """ NIH Dataset

    Args:
        root_dir (string): Path to the root directory of the dataset.
        transform (Dict, optional): Optional transform to be applied
            on a data.

    Returns:
        Dict: A dictionary containing the data and its label. 
    """
    def __init__(self, root_dir:str, transform:Optional[Dict[str,Any]]=None):
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index) -> Dict[str, Any]:
        return 0
    
    def __len__(self) -> int:
        return 0
