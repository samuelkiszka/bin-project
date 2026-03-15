import logging
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

class SiameseDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y

        # Build class index dictionary
        self.class_indices = defaultdict(list)

        for idx, label in enumerate(y):
            self.class_indices[label].append(idx)

        self.classes = list(self.class_indices.keys())

    def __len__(self):
        return len(self.data) * 10  # We will generate pairs on the fly, so we can return a larger length to allow for more pairs

    def __getitem__(self, idx):
        idx1 = np.random.randint(0, len(self.data))

        x1 = self.data[idx1]
        label1 = self.labels[idx1]

        # Randomly decide to create a positive or negative pair in a 1:2 ratio
        if np.random.rand() < 0.33:
            # Positive pair
            idx2 = np.random.choice(self.class_indices[label1])
            label = 1
        else:
            # Negative pair
            label2 = np.random.choice([l for l in self.classes if l != label1])
            idx2 = np.random.choice(self.class_indices[label2])
            label = 0

        x2 = self.data[idx2]

        return (
            np.expand_dims(x1, axis=0).astype(np.float32),
            np.expand_dims(x2, axis=0).astype(np.float32),
            np.float32(label),
        )