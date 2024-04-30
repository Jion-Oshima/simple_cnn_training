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

class MyLabeledVideoPaths(Dataset):
    def __init__(
        self,
        root: str,
    ):
