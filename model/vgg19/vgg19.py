from torchvision.models import (
    vgg19,
    VGG19_Weights,
)

from model import ModelConfig, ClassificationBaseModel


class VGG19(ClassificationBaseModel):

    def __init__(self, model_info: ModelConfig):
        super().__init__(model_info)
        self.prepare_model()

    def prepare_model(self):
        self.model = vgg19(
            weights=VGG19_Weights.DEFAULT
            if self.model_config.use_pretrained else None
        )
