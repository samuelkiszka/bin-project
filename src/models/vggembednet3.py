import torch

from src.models.model_base import create_vgg_block, BaseModel


class VGGEmbedNet3(BaseModel):
    NAME = 'VGGEmbedNet3'
    MAX_EMB_DIM = 64
    def __init__(self, emb_dim):
        assert emb_dim <= self.MAX_EMB_DIM, f"Model {self.NAME} only supports embedding dimensions up to {self.MAX_EMB_DIM}, but got {emb_dim}."
        conv = torch.nn.Sequential(
            create_vgg_block(1, 32, subsampling=(16, 16)),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
        )

        output_layer = torch.nn.Sequential(
            torch.nn.Linear(64, emb_dim)
        )

        super().__init__(conv=conv, output_layer=output_layer)
