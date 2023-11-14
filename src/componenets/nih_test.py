import pytest
import logging
from pathlib import Path

import numpy as np

from .nih import NIH
from ..utils.common import get_repo_root

logger = logging.getLogger(__name__)

@pytest.fixture
def nih_train():
    return NIH(root_dir='', split='train')

@pytest.fixture
def nih_test():
    return NIH(root_dir='', split='test')

@pytest.fixture
def nih_val():
    return NIH(root_dir='', split='val')

def test_nih_train_len(nih_train:NIH):
    assert len(nih_train) == 89783, "The length of the dataset should be 89783"
    assert nih_train.annots[0]['split'] == 'train', "The split should be train"
    return 

def test_nih_test_len(nih_test:NIH):
    assert len(nih_test) == 11219, "The length of the dataset should be 11219"
    assert nih_test.annots[0]['split'] == 'test', "The split should be test"
    return

def test_nih_val_len(nih_val:NIH):
    assert len(nih_val) == 11118, "The length of the dataset should be 11118"
    assert nih_val.annots[0]['split'] == 'val', "The split should be test"
    return

def test_nih_load_data(nih_train:NIH):
    nih_train.root_dir = Path(get_repo_root()) / 'data'
    nih_train.image_channels = 3
    dummy = nih_train._load_data(0)
    image:np.ndarray = dummy['image']
    assert image.shape == (1024, 1024, 3), "The image should be 1024 x 1024 x 3"
    assert image.max() == 255.0, "The image should be normalized to 255.0"
    assert image.min() == 0.0, "The image should be normalized to 0.0"

    nih_train.image_channels = 1
    dummy = nih_train._load_data(0)
    image:np.ndarray = dummy['image']
    assert image.shape == (1024, 1024, 1), "The image should be 1024 x 1024 x 1"
    assert image.max() == 255.0, "The image should be normalized to 255.0"
    assert image.min() == 0.0, "The image should be normalized to 0.0"