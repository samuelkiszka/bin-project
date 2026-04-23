import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embed1, embed2, label):
        # label: 1 = same, 0 = different

        # Euclidean distance
        distances = F.pairwise_distance(embed1, embed2)

        loss_pos = label * distances.pow(2)
        loss_neg = (1 - label) * torch.clamp(self.margin - distances, min=0).pow(2)

        loss = torch.mean(loss_pos + loss_neg)
        return loss