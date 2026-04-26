import torch
from thop import profile

import torch.nn.functional as F


def create_vgg_block(
    input_channels: int,
    output_channels: int,
    subsampling: tuple[int, int] = (2, 2)
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1
        ),
        torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(
            kernel_size=subsampling,
            stride=subsampling
        ),
        torch.nn.InstanceNorm2d(num_features=output_channels)
    )


class BaseModel(torch.nn.Module):
    NAME = 'model_base'
    def __init__(self, conv=None, output_layer=None):
        super().__init__()
        # Layers
        self.conv = conv
        self.output_layer = output_layer

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # [batch_size, 512, 8, 8]
        features = features.mean(dim=[2, 3])  # Global average pooling to [batch_size, 512]
        embedding = self.output_layer(features)
        return F.normalize(embedding, dim=1)  # Normalize to unit length

    def forward(self, x1, x2):
        # Embed both inputs
        emb1 = self.embed(x1)
        emb2 = self.embed(x2)

        return emb1, emb2 # [batch_size], [batch_size]

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_param_stats(self, device):
        dummy_input = torch.randn(1, 1, 256, 256).to(device)
        macs, params = profile(self, inputs=(dummy_input, dummy_input), verbose=False)
        return macs, params