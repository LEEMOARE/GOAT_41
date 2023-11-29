import random
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
                       }
#    2: 6,  # emphysema train id is 6
#    4: 7,  # edema train id is 7
#    5: 8,  # consolidation train id is 8
#    6: 9,  # pleural_thickening train id is 9
#    10: 10}  # cardiomegaly train id is 10


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
                 train_lesion: Optional[List[int]] = None,
                 ratio: float = -1.0,
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size if isinstance(image_size, tuple) \
            else (image_size, image_size)

        self.transform = None
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

        if train_lesion is None:
            self.num_classes = max(list(_LESIOM_TO_TRAIN_ID.values()))
            self.mapper = None
        else:
            self.train_lesion = train_lesion
            self.num_classes = len(self.train_lesion)
            self.mapper = {lesion: i for i,
                           lesion in enumerate(self.train_lesion)}

        self.ratio = ratio  # if self.split == 'train' else -1.0
        self.annots = self._load_annotations()

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
                'label_name': '_'.join(annot['label_names']),
                'path': str(path)}

    def __len__(self) -> int:
        return self.total_length

    def _load_annotations(self):
        path = Path(get_repo_root()) / 'data/nih.zip'
        anns = load_json(path)

        # split train, test, valid
        anns = [ann for ann in anns if ann['split'] == self.split]

        if self.mapper is None:
            self.total_length = len(anns)
            return anns

        self.idx_normal = []
        self.idx_abnormal = []
        for i, ann in enumerate(anns):
            if any(label_index in self.mapper for label_index in ann['label_indexes']):
                self.idx_abnormal.append(i)
            else:
                self.idx_normal.append(i)

        self.len_normal = len(self.idx_normal)
        self.len_abnormal = len(self.idx_abnormal)
        # it dosen`t works when len_abnormal > len_normal
        self.total_length = self.len_abnormal + self.len_normal if self.len_abnormal > self.len_normal or self.ratio < 0 \
            else self.len_abnormal + int(self.len_abnormal * self.ratio)
        return anns

    def _label_to_onehot(self, label_indexes: List[int]) -> np.array:
        # create empty one-hot vector
        train_ids = np.zeros((self.num_classes,), dtype=np.int32)

        # convert label_indexes to multi-label one-hot vector
        if 9 in label_indexes:
            return train_ids  # no-finding

        if self.mapper is not None:
            for label_index in label_indexes:
                if label_index not in self.mapper:
                    continue
                train_ids[self.mapper[label_index]] = 1
        else:
            for label_index in label_indexes:
                if label_index not in list(_LESIOM_TO_TRAIN_ID.keys()):
                    continue
                train_ids[_LESIOM_TO_TRAIN_ID[label_index]-1] = 1
        return train_ids

    def _load_data(self, idx) -> Dict[str, Any]:
        if self.ratio > 0:
            if self.len_abnormal <= idx:
                idx = self.idx_normal[int(
                    random.uniform(0, self.len_normal - 1))]
            else:
                idx = self.idx_abnormal[idx]
        annot = self.annots[idx]
        filename = annot['filename']
        gender = annot['gender']
        age = annot['age']
        view_position = annot['view_position']
        label_names = annot['label_names']
        label_indexes: List[int] = annot['label_indexes']

        train_ids = self._label_to_onehot(label_indexes)

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


class SimpleNIH(NIH):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 image_size: Union[int, Tuple[int, int]] = (1024, 1024),
                 image_channels: int = 3,
                 train_lesion: Optional[List[int]] = None,
                 ratio: float = -1.0):
        super().__init__(root_dir, split, image_size,
                         image_channels, train_lesion, ratio)

        # This class take only normal and abnormal classes
        self.num_classes = 1

    def _load_data(self, idx):
        dummy = super()._load_data(idx)
        label_names: List[str] = dummy['label_names']
        labels = [0] if 'nofinding' in label_names else [1]
        labels = np.array(labels, dtype=np.int32)
        dummy['labels'] = labels
        return dummy
