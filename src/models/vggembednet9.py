import torch
from thop import profile

from src.models.model_base import create_vgg_block, BaseModel


class VGGEmbedNet9(BaseModel):
    NAME = 'VGGEmbedNet9'
    MAX_EMB_DIM = 64

    def __init__(self, emb_dim):
        assert emb_dim <= self.MAX_EMB_DIM

        conv = torch.nn.Sequential(
            # 256x256 -> 8x8
            torch.nn.MaxPool2d(32, 32),

            create_vgg_block(1, 2, subsampling=(4, 4)),
            torch.nn.Conv2d(2, 4, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        output_layer = torch.nn.Sequential(
            torch.nn.Linear(4, emb_dim)
        )

        super().__init__(conv=conv, output_layer=output_layer)