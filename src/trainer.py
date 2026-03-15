from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import torch
import logging


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
        predictions = self.model(x1, x2)
        loss = self.loss_fn(predictions, label)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self, train_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_idx, (x, y, label) in enumerate(train_loader):
                loss = self.train_step(x, y, label)
                total_loss += loss
                # if batch_idx == 1:
                #     break  # Limit to 10 batches per epoch to test training loop

            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    def evaluate(self, test_loader):
        self.model.eval()

        all_scores = []
        all_labels = []

        with (torch.no_grad()):
            for x1, x2, label in test_loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                logits = self.model(x1, x2)

                scores = torch.sigmoid(logits).cpu().numpy()
                labels = label.numpy()

                all_scores.extend(scores)
                all_labels.extend(labels)

        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # ROC-AUC
        auc = roc_auc_score(all_labels, all_scores)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

        # Optimal treshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        preds = (all_scores >= optimal_threshold).astype(int)
        accuracy = np.mean(preds == all_labels)

        return accuracy, auc, optimal_threshold
