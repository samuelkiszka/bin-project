import logging
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

class SiameseDataset(Dataset):
    def __init__(self, x, y, pairs_per_epoch=None):
        self.data = x
        self.labels = y
        self.pairs_per_epoch = pairs_per_epoch

        # Build class index dictionary
        self.class_indices = defaultdict(list)

        for idx, label in enumerate(y):
            self.class_indices[label].append(idx)

        self.classes = list(self.class_indices.keys())
        self.neg_classes = {
            c: [x for x in self.classes if x != c]
            for c in self.classes
        }

    def __len__(self):
        if self.pairs_per_epoch is not None:
            return self.pairs_per_epoch
        return len(self.data)

    def __getitem__(self, idx):
        idx1 = np.random.randint(0, len(self.data))

        x1 = self.data[idx1]
        label1 = self.labels[idx1]

        # Randomly decide to create a positive or negative pair in a 1:2 ratio
        if np.random.rand() < 0.5:
            # Positive pair
            indices = self.class_indices[label1]
            idx2 = idx1
            while idx2 == idx1:  # Ensure we don't pick the same sample
                idx2 = np.random.choice(self.class_indices[label1])
            label = 1
        else:
            # Negative pair
            label2 = np.random.choice(self.neg_classes[label1])
            idx2 = np.random.choice(self.class_indices[label2])
            label = 0

        x2 = self.data[idx2]

        return (
            np.expand_dims(x1, axis=0).astype(np.float32),
            np.expand_dims(x2, axis=0).astype(np.float32),
            np.float32(label),
        )