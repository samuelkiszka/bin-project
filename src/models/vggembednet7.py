import torch
from thop import profile

from src.models.model_base import create_vgg_block, BaseModel


class VGGEmbedNet7(BaseModel):
    NAME = 'VGGEmbedNet7'
    MAX_EMB_DIM = 64

    def __init__(self, emb_dim):
        assert emb_dim <= self.MAX_EMB_DIM

        conv = torch.nn.Sequential(
            # 256x256 -> 8x8
            torch.nn.MaxPool2d(32, 32),

            create_vgg_block(1, 8, subsampling=(4, 4)),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        output_layer = torch.nn.Sequential(
            torch.nn.Linear(16, emb_dim)
        )

        super().__init__(conv=conv, output_layer=output_layer)