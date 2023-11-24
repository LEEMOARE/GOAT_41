from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

_LESIOM_TO_TRAIN_ID = {9: 0,  # nofinding train id is 0
                       1: 1,  # effusion train id is 1
                       8: 2,  # mass train id is 2
                       11: 2,  # nodule train id is 2
                       3: 3,  # atelectasis train id is 3
                       12: 4,  # pneumothorax train id is 4
                       15: 5,  # infiltration train id is 5
                       2: 6,  # emphysema train id is 6
                       4: 7,  # edema train id is 7
                       5: 8,  # consolidation train id is 8
                       6: 9,  # pleural_thickening train id is 9
                       10: 10}  # cardiomegaly train id is 10


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
                 root_dir: str,
                 split: str = 'train',
                 image_size: Union[int, Tuple[int, int]] = (1024, 1024),
                 image_channels: int = 3,
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size if isinstance(image_size, tuple) \
            else (image_size, image_size)

        if split == 'train':
            self.transform = transforms.Compose([transforms.Resize(self.image_size),  # 이미지 크기 조정
                                                 transforms.RandomHorizontalFlip(),  # 데이터 증강을 위한 무작위 수평 뒤집기
                                                 transforms.RandomVerticalFlip(),  # 데이터 증강을 위한 무작위 수직 뒤집기
                                                 # 최대 30도로 무작위 회전
                                                 transforms.RandomRotation(30),
                                                 # 색상, 대비, 밝기 조절
                                                 transforms.ColorJitter(brightness=0.2,
                                                                        contrast=0.2,
                                                                        saturation=0.2),
                                                 # 무작위로 크기를 조절하고 잘라내기
                                                 transforms.RandomResizedCrop(self.image_size,
                                                                              scale=(0.8, 1.0)),
                                                 # 가우시안 블러 적용
                                                 transforms.GaussianBlur(kernel_size=3,
                                                                         sigma=(0.1, 2.0)),
                                                 ])

        self.image_channels = image_channels

        self.annots = self._load_annotations()

        self.num_classes = max(list(_LESIOM_TO_TRAIN_ID.values()))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        annot = self._load_data(index)
        image: np.ndarray = annot['image']  # (H,W,3 or 1)
        path: str = annot['path']
        labels: np.array = annot['labels']

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)

        image = image/255.0

        return {'image': image.float(),
                'label': torch.from_numpy(labels).long(),
                'path': str(path)}

    def __len__(self) -> int:
        return len(self.annots)

    def _load_annotations(self):
        path = Path(get_repo_root()) / 'data/nih.zip'
        annots = load_json(path)

        # filter by split
        annots = [annot for annot in annots if annot['split'] == self.split]

        # filter normal class

        # do something

        return annots

    def _load_data(self, idx) -> Dict[str, Any]:
        annot = self.annots[idx]
        filename = annot['filename']
        gender = annot['gender']
        age = annot['age']
        view_position = annot['view_position']
        label_names = annot['label_names']
        label_indexes: List[int] = annot['label_indexes']

        train_ids = np.zeros((10,), dtype=np.int32)
        # convert label_indexes to multi-label one-hot vector
        if 9 in label_indexes and len(label_indexes) > 1:
            # no-finding is not the only label
            raise ValueError('no-finding is not the only label')

        for label_index in label_indexes:
            if label_index not in list(_LESIOM_TO_TRAIN_ID.keys()):
                continue

            if label_index == 9:
                continue
            train_ids[_LESIOM_TO_TRAIN_ID[label_index]-1] = 1

        image_path = Path(self.root_dir) / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        image = cv2.resize(image, self.image_size,
                           interpolation=cv2.INTER_LINEAR)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)  # (H,W) -> (H,W,1)

        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        if self.image_channels == 3 and image.shape[2] == 1:
            # (H,W,1) -> (H,W,3)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # min-max normalization
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # to 8bit image
        image = image * 255.0
        image = image.astype(np.uint8)

        return {'image': image,
                'path': image_path,
                'gender': gender,
                'age': age,
                'view_position': view_position,
                'label_names': label_names,
                'labels': train_ids}
