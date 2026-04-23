from time import perf_counter

from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import torch
import logging

import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, x1, x2, label):
        # Move data to the same device as the model
        x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)

        # Forward pass
        emb1, emb2 = self.model(x1, x2)
        loss = self.loss_fn(emb1, emb2, label.float())

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self, train_loader, num_epochs):
        self.model.train()
        times = []
        losses = []
        for epoch in range(num_epochs):
            epoch_start = perf_counter()
            total_loss = 0.0
            for batch_idx, (x, y, label) in enumerate(train_loader):
                loss = self.train_step(x, y, label)
                total_loss += loss

            epoch_end = perf_counter()
            epoch_time = epoch_end - epoch_start
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
            times.append(round(epoch_time, 5))
            losses.append(round(avg_loss, 5))

        return times, losses

    def evaluate(self, test_loader):
        self.model.eval()

        all_distances = []
        all_labels = []

        with (torch.no_grad()):
            for x1, x2, label in test_loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                emb1, emb2 = self.model(x1, x2)

                distance = F.pairwise_distance(emb1, emb2).cpu().numpy()
                labels = label.numpy()

                all_distances.extend(distance)
                all_labels.extend(labels)

        # Convert to numpy arrays
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        similarity_scores = -all_distances  # Higher similarity = more likely to be the same class

        logger.info("pos mean: %f", all_distances[all_labels == 1].mean())
        logger.info("neg mean: %f", all_distances[all_labels == 0].mean())

        # ROC-AUC
        auc = roc_auc_score(all_labels, similarity_scores)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, similarity_scores)

        # Optimal treshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        optimal_threshold_dist = -optimal_threshold  # Convert back to distance threshold

        preds = (all_distances < optimal_threshold_dist).astype(int)
        accuracy = np.mean(preds == all_labels)

        return accuracy, auc, optimal_threshold
