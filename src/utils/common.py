import copy
import json
import zipfile
from pathlib import Path
from typing import Union

import cv2
import git
import numpy as np
import torch


def get_repo_root() -> str:
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir


def load_json(path: Union[str, Path]):
    if Path(path).suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.json'):
                    return json.load(zip_ref.open(file))
    else:
        with open(path, 'r') as f:
            return json.load(f)


def load_image(path: str, image_size: int = -1) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        np.expand_dims(image, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size > 0:
        image = cv2.resize(image, (image_size, image_size),
                           interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = min_max_norm(image)
    image = cvt_float32_to_uint8(image)
    return image


def min_max_norm(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def cvt_float32_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image * 255.0
    image = image.astype(np.uint8)
    return image


def get_info_from_name(name: str) -> tuple[str, list[int]]:
    name = Path(name).stem
    name = name.split('_')
    model_name = name[0]
    lesions = name[1:]
    lesions = [int(l) for l in lesions]
    return model_name, lesions


def cvt_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    assert len(image.shape) == 3
    image = image.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image


def post_process_cam(prob: float, cam: np.ndarray, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if prob < threshold:
        return np.zeros_like(cam)
    new_cam = copy.deepcopy(cam)
    new_cam = np.where(new_cam > 0.5, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(new_cam, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for contour in contours:
        # 컨투어 내부의 최대 값 찾기
        max_value = np.max(new_cam * (cv2.drawContours(np.zeros_like(new_cam),
                                                       [contour], 0, 1,
                                                       thickness=cv2.FILLED)))

        # 최대 값이 0.9 이상인 경우만 남기기
        if max_value >= 0.9:
            filtered_contours.append(contour)
    # 원본에 컨투어 그리기
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.drawContours(image, filtered_contours, -1, (255, 0, 255), 2)
    return image


def load_weights(model: torch.nn.Module, weights: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(weights, map_location='cpu'),
                          strict=False)
    return model
