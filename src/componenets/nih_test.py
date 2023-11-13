import pytest
import logging

from .nih import NIH

logger = logging.getLogger(__name__)

@pytest.fixture
def nih():
    return NIH(root_dir='', split='train')

def test_nih_len(nih:NIH):
    assert len(nih) == 112120, "The length of the dataset should be 112120"
    logger.debug(nih.annots[0])
    return 

