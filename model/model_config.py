from typing import Literal
from dataclasses import dataclass

# モデルを追加する
SupportedModels = Literal[
    "resnet18",
    "resnet50",
    "abn_r50",
    "vit_b",
    "x3d",
    "zero_output_dummy",
    "vgg19"
]


@dataclass
class ModelConfig:
    model_name: SupportedModels = "resnet18"
    use_pretrained: bool = True
    torch_home: str = "./"
    n_classes: int = 10
