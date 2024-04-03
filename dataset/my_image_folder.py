import os
import torch
# import torchvision
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable, Dict, List, Any
from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def find_classes(root: str):

    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())  # クラスの名前
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {root}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}  # クラスのインデックス
    return classes, class_to_idx


def mymake_dataset(
    root: str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, int]]:

    data = []

    if class_to_idx is None:
        _, class_to_idx = find_classes(root)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    for target_class in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[target_class]
        target_dir = os.path.join(root, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, filenames in sorted(os.walk(target_dir, followlinks=True)):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                _, ext = os.path.splitext(filepath)
                for img_ex in IMG_EXTENSIONS:
                    if ext == img_ex:
                        eachdata = [filepath, class_idx]
                        data.append(eachdata)
                        break

    return data

    # directory = os.path.expanduser(directory)

    # if class_to_idx is None:
    #     _, class_to_idx = find_classes(directory)
    # elif not class_to_idx:
    #     raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # instances = []
    # available_classes = set()  # 集合を作成．
    # for target_class in sorted(class_to_idx.keys()):
    #     class_idx = class_to_idx[target_class]
    #     target_dir = os.path.join(directory, target_class)
    #     if not os.path.isdir(target_dir):
    #         continue
    #     for root, _, filenames in sorted(os.walk(target_dir, followlinks=True)):
    #         for filename in filenames:
    #             filepath = os.path.join(root, filename)
    #             for img_ex in IMG_EXTENSIONS:
    #                 if img_ex in filepath:
    #                     item = filepath, class_idx
    #                     instances.append(item)
    #                     if target_class not in available_classes:  # 既にそのクラスが存在するかを判定
    #                         available_classes.add(target_class)
    #                     break

    # empty_classes = set(class_to_idx.keys()) - available_classes
    # if empty_classes:
    #     msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
    #     raise FileNotFoundError(msg)

    # return instances


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyDatasetFolder(Dataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 ):
        super().__init__(root, transforms=transforms)
        self.transforms = transforms

        initial_pathlist = Path(root).glob("**/*")
        data = []

        classes, class_to_idx = find_classes(root)

        for path in initial_pathlist:
            categoryname = path.parent.parent.name
            _, ext = os.path.splitext(path)
            for img_ex in IMG_EXTENSIONS:
                if ext == img_ex:

                    eachdata = [path, categoryname]
                    data.append(eachdata)
                    break

        # print(data)
        self.data = data

        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def mymake_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[str, int]]:

        return mymake_dataset(directory, class_to_idx)

    def find_classes(
        self,
        directory: str
    ) -> Tuple[List[str], Dict[str, int]]:

        return find_classes(directory)

    def __getitem__(self, idx: int):
        target = self.data[idx][0]
        sample = self.data[idx][1]

        pimg = Image.open(sample)
        sample = transforms.functional.to_tensor(pimg)

        return sample, target
