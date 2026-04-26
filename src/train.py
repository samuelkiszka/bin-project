import json

import torch
import argparse
from datetime import datetime
import logging

from src.models.models import get_models
from src.env import DATA_PATH, LOG_DIR, RES_DIR
from src.utils import load_data, ensure_directory_exists
from src.experiment import Experiment, Results

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Siamese Network for sequence similarity.")

    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension for the Siamese Network.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for data loading.")
    parser.add_argument("--pairs_per_epoch", type=int, default=1, help="Number of pairs to generate per epoch.")
    parser.add_argument("--model", type=str, default="baseline", choices=get_models().keys(), help="Model architecture to use.")
    parser.add_argument("--runs_per_model", type=int, default=1, help="Number of runs per model to use.")
    parser.add_argument("--note", type=str, default="", help="Additional note to include in the result file name.")

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

def log_args(args):
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")


def get_trained_model(model_name):
    if model_name == "all":
        return get_models().values()
    return [get_models()[model_name]]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.multiprocessing.set_start_method('spawn')
    configure_logging()
    log_args(args)

    ensure_directory_exists(RES_DIR)
    ensure_directory_exists(LOG_DIR)

    results = Results(args.model, args.note)
    data, labels = load_data(DATA_PATH)

    for model in get_trained_model(args.model):
        experiment = Experiment(
            model=model,
            data=data,
            labels=labels,
            args=args,
            device=device
        )
        logger.info("-------------------------------")
        logger.info(f"Starting experiment for model: {model.NAME}")
        logger.info(f"Results will be saved under: {results.results_name}")
        experiment.run()
        results.add_experiment_result(experiment)
        results.save_results()


if __name__ == "__main__":
    main()