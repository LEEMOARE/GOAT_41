from functools import partial

from torch.utils.data import DataLoader

from ..componenets.nih import NIH

_TRAIN_BASIC_SETTINGS = {'shuffle': True, 'num_workers': 0,
                         'pin_memory': True, 'drop_last': True, 'collate_fn': None}
_VAL_BASIC_SETTINGS = {'shuffle': False, 'num_workers': 0,
                       'pin_memory': True, 'drop_last': False, 'collate_fn': None}
_TEST_BASIC_SETTINGS = {'shuffle': False, 'num_workers': 0,
                        'pin_memory': True, 'drop_last': False, 'collate_fn': None}


def get_NIH_dataloader(dataset: NIH, batch_size: int, use_basic: bool = True, **kwargs):
    if use_basic and dataset.split == 'train':
        return DataLoader(dataset, batch_size=batch_size, **_TRAIN_BASIC_SETTINGS)
    elif use_basic and dataset.split == 'val':
        return DataLoader(dataset, batch_size=batch_size, **_VAL_BASIC_SETTINGS)
    elif use_basic and dataset.split == 'test':
        return DataLoader(dataset, batch_size=batch_size, **_TEST_BASIC_SETTINGS)
    else:
        return DataLoader(dataset, batch_size=batch_size, **kwargs)
