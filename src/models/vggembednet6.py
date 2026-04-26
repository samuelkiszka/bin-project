import torch
from thop import profile

from src.models.model_base import create_vgg_block, BaseModel


class VGGEmbedNet6(BaseModel):
    NAME = 'VGGEmbedNet6'
    MAX_EMB_DIM = 64

    def __init__(self, emb_dim):
        assert emb_dim <= self.MAX_EMB_DIM

        conv = torch.nn.Sequential(
            # 256x256 -> 16x16
            torch.nn.MaxPool2d(16, 16),

            create_vgg_block(1, 16, subsampling=(8, 8)),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        output_layer = torch.nn.Sequential(
            torch.nn.Linear(32, emb_dim)
        )

        super().__init__(conv=conv, output_layer=output_layer)