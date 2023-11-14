from functools import partial

from torch.utils.data import DataLoader


DataLoaderNIHTrain = partial(DataLoader, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)