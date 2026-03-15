import torch
import math

import torch.nn.functional as F


def create_vgg_block(
    input_channels: int,
    output_channels: int,
    subsampling: tuple[int, int] = (2, 2)
) -> torch.nn.Sequential:

    """
    Create one convolutional block used in the encoder

    Parameters:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        subsampling (tuple[int, int]): Pooling kernel size and stride.

    Returns:
        nn.Sequential: Convolutional block.
    """

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


class PositionalEncoding(torch.nn.Module):

    """
    Add sinusoidal positional encoding to sequence features.
    """

    def __init__(self, d_model: int, max_len: int = 60):

        """
        Initialize positional encoding.

        Parameters:
            d_model (int): Embedding dimension.
            max_len (int): Maximum supported sequence length.
        """

        super().__init__()

        # position indices: [0, 1, 2, ..., max_len - 1]
        position = torch.arange(max_len).unsqueeze(1)

        # frequency scaling term used in the sinusoidal encoding formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # compute positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer to avoid updating during training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Add positional encoding to input features.

        Parameters:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output features with positional encoding added.
        """

        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class SiameseNetwork(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.conv = torch.nn.Sequential(
            create_vgg_block(1, 32, subsampling=(4, 4)),
            create_vgg_block(32, 64, subsampling=(4, 4)),
            create_vgg_block(64, 128, subsampling=(2, 2)),
            torch.nn.Conv2d(128, 512, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
        )

        self.pos_encoding = PositionalEncoding(d_model=512, max_len=2000)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=512,
            dim_feedforward=512,
            nhead=4,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, dim)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
            Embed input image into a fixed-size vector representation.
        :param x: [batch_size, 1, 256, 256] input image tensor
        :return: [batch_size, dim] normalized embedding vector
        """
        # extract local features with CNN
        features = self.conv(x) # [batch_size, 512, 8, 8]

        # Flatten spatial dimensions and permute to [batch_size, seq_len, feature_dim]
        B, C, H, W = features.shape
        features = features.view(B, C, H * W) # [batch_size, 512, 64]

        # reshape to [batch_size, seq_len, feature_dim]
        features = features.permute(0, 2, 1) # [batch_size, 64, 512]

        # add positional encoding
        features = self.pos_encoding(features) # [batch_size, 64, 512]

        # apply transformer encoder
        encoded = self.transformer_encoder(features) # [batch_size, 64, 512]

        # project to output embedding dimension
        output = self.output_layer(encoded[:, -1, :]) # [batch_size, 256]

        return F.normalize(output, dim=1) # [batch_size, dim]

    def forward(self, x1, x2):
        # Embed both inputs
        embed1 = self.embed(x1)
        embed2 = self.embed(x2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(embed1, embed2, dim=1) # [batch_size]

        logits = (similarity + 1) / 2  # Scale cosine similarity from [-1, 1] to [0, 1]
        return logits # [batch_size]