import json
import logging
from datetime import datetime

import torch
from sklearn.model_selection import StratifiedKFold

from src.contrastive_loss import ContrastiveLoss
from src.env import RES_DIR
from src.dataset import SiameseDataset
from src.trainer import Trainer

logger = logging.getLogger(__name__)

def log_model_stats(model, device):
    macs, params = model.model_param_stats(device)
    logger.info(f"Model: {model.NAME} - MACs: {macs}, Parameters: {params}")


def create_dataloader(dataset, batch_size, num_workers=1, shuffle=True, drop_last=True):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


class Results:
    def __init__(self, name):
        self.experiments = []
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_name = f"{name}_{run_id}"

    def add_experiment_result(self, experiment_result):
        self.experiments.append(experiment_result)

    def save_results(self):
        results_file = f"{RES_DIR}/{self.results_name}.json"
        out_json = {
            "experiment_name": self.results_name,
            "experiments": [
                {
                    "model": experiment.model.NAME,
                    "epochs": experiment.args.num_epochs,
                    "pairs_per_epoch": experiment.args.pairs_per_epoch,
                    "batch_size": experiment.args.batch_size,
                    "learning_rate": experiment.args.lr,
                    "embedding_dim": experiment.args.emb_dim,
                    "experiment_results": experiment.to_json()
                } for experiment in self.experiments
            ]
        }

        with open(results_file, "w") as f:
            json.dump(out_json, f, indent=4)

        logger.info(f"Saved results to {results_file}")


class RunResult:
    def __init__(self, experiment, fold_results):
        self.experiment = experiment
        self.fold_results = fold_results

    def to_json(self):
        avg_run_acc = sum(fold_result.test_acc for fold_result in self.fold_results) / len(self.fold_results)
        return {
            "avg_run_acc": round(avg_run_acc, 5),
            "folds": len(self.fold_results),
            "fold_results": [
                {
                    "fold": i + 1,
                    "results": fold_result.to_json()
                } for i, fold_result in enumerate(self.fold_results)
            ]
        }


class Experiment:
    def __init__(self, model, data, labels, args, device):
        self.model = model
        self.data = data
        self.labels = labels
        self.args = args
        self.device = device
        self.run_results = []

    def run(self):
        runs = self.args.runs_per_model
        for run in range(runs):
            logger.info(f"Starting run {run + 1}/{runs} for model {self.model.NAME}")
            result = self._run_single()
            self.run_results.append(result)

    def _run_single(self):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        loss_fn = ContrastiveLoss()

        data, labels = self.data, self.labels
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
            # prepare model
            model = self.model(emb_dim=self.args.emb_dim).to(self.device)
            log_model_stats(model, self.device)
            # prepare optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

            trainer = Trainer(model, optimizer, loss_fn, self.device)

            train_dataset = SiameseDataset(data[train_idx], labels[train_idx], pairs_per_epoch=self.args.pairs_per_epoch)
            test_dataset = SiameseDataset(data[test_idx], labels[test_idx])

            train_loader = create_dataloader(train_dataset, self.args.batch_size, self.args.num_workers, shuffle=True)
            test_loader = create_dataloader(test_dataset, self.args.batch_size, self.args.num_workers, shuffle=False,
                                            drop_last=False)

            epoch_times, epoch_losses = trainer.train(train_loader, num_epochs=self.args.num_epochs)
            acc, auc, threshold = trainer.evaluate(test_loader)

            logger.info(f"Fold {fold + 1} - Test Accuracy: {acc:.4f}, AUC: {auc:.4f}, Threshold: {-threshold:.4f}")

            fold_results.append(FoldResult(
                epoch_times=epoch_times,
                epoch_losses=epoch_losses,
                test_acc=round(acc, 5),
                test_auc=round(auc, 5)
            ))

        avg_acc = sum(fold_result.test_acc for fold_result in fold_results) / len(fold_results)
        logger.info(f"Average Train Accuracy: {avg_acc:.4f}")

        return RunResult(
            experiment=self,
            fold_results=fold_results,
        )

    def to_json(self):
        model = self.model(emb_dim=self.args.emb_dim).to(self.device)
        mac, params = model.model_param_stats(self.device)
        return {
            "macs": int(mac),
            "params": int(params),
            "runs": len(self.run_results),
            "run_results": [
                {
                    "run": i + 1,
                    "run_results": run_result.to_json()
                } for i, run_result in enumerate(self.run_results)
            ]
        }

class FoldResult:
    def __init__(self, epoch_times, epoch_losses, test_acc, test_auc):
        self.epoch_times = epoch_times
        self.epoch_losses = epoch_losses
        self.test_acc = test_acc
        self.test_auc = test_auc

    def to_json(self):
        return {
            "epoch_times": self.epoch_times,
            "epoch_losses": self.epoch_losses,
            "test_acc": self.test_acc,
            "test_auc": self.test_auc,
        }
