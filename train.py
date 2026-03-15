import torch
import argparse
from datetime import datetime
import logging
from sklearn.model_selection import StratifiedKFold

from src.dataset import SiameseDataset
from src.model import SiameseNetwork
from src.trainer import Trainer
from env import DATA_PATH, LOG_DIR
from utils import load_data, ensure_directory_exists

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese Network for sequence similarity.")

    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension for the Siamese Network.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")

    return parser.parse_args()

def configure_logging():
    ensure_directory_exists(LOG_DIR)
    # get current timestamp for log file naming
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logg_file = f"{LOG_DIR}/train_{run_id}.log"
    # log both to console and to a file with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(logg_file),
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )


def create_model(args, device):
    model = SiameseNetwork(args.emb_dim).to(device)
    return model


def create_dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.multiprocessing.set_start_method('spawn')
    configure_logging()

    data, labels = load_data(DATA_PATH)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        train_dataset = SiameseDataset(data[train_idx], labels[train_idx])
        test_dataset = SiameseDataset(data[test_idx], labels[test_idx])

        train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
        test_loader = create_dataloader(test_dataset, args.batch_size, shuffle=False, drop_last=False)

        model = create_model(args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        trainer = Trainer(model, optimizer, loss_fn, device)
        trainer.train(train_loader, num_epochs=args.num_epochs)

        acc, auc, threshold = trainer.evaluate(test_loader)

        logger.info(f"Fold {fold+1} - Test Accuracy: {acc:.4f}, AUC: {auc:.4f}, Threshold: {threshold:.4f}")

        fold_results.append(acc)

    avg_acc = sum(fold_results) / len(fold_results)
    logger.info(f"Average Train Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()