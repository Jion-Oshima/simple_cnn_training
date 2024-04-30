import os
import itertools
from pathlib import Path
from typing import Tuple, Any
from dataclasses import dataclass
from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    SequentialSampler,
    RandomSampler,
)

from torchvision.transforms import v2 as transforms

from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.data.clip_sampling import (
    RandomClipSampler,
    ConstantClipsPerVideoSampler,
)

class MyLabeledVideoDataset(Dataset):
    def __init__(self, video_path, transform=None) -> None:
        self.capture = cv2.VideoCapture(video_path)
        self.transform = transform

    def __getitem__(self, index) -> Any:
        ret, img = self.capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # numpy 配列を PIL Image に変換する。
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
