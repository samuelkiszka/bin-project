import torch

from src.models.model_base import create_vgg_block, BaseModel


class BaselineModel(BaseModel):
    NAME = 'baseline'
    def __init__(self, emb_dim=128):
        conv = torch.nn.Sequential(
            create_vgg_block(1, 32, subsampling=(4, 4)),
            create_vgg_block(32, 64, subsampling=(4, 4)),
            create_vgg_block(64, 128, subsampling=(2, 2)),
            torch.nn.Conv2d(128, 512, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
        )

        output_layer = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, emb_dim)
        )

        super().__init__(conv=conv, output_layer=output_layer)
