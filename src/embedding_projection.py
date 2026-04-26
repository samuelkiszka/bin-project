import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.models import get_models
from src.dataset import SiameseDataset
from src.env import DATA_PATH
from src.dataset import create_dataloader
from src.utils import load_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--emb_dims", type=int, nargs="+", required=True)

    parser.add_argument("--save_path", type=str, default="comparison.png")

    return parser.parse_args()


def load_model(name, path, emb_dim, device):
    model_class = get_models()[name]
    model = model_class(emb_dim=emb_dim)

    checkpoint = torch.load(path, map_location="cpu")

    state = {
        k: v for k, v in checkpoint.items()
        if torch.is_tensor(v)
    }

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model


def extract_embeddings(model, loader, device):
    embs = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch

            x = x.to(device).float()
            emb = model.embed(x)

            embs.append(emb.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(embs), np.concatenate(labels)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, labels = load_data(DATA_PATH)

    dataset = SiameseDataset(data, labels, pairs_per_epoch=None, all=True)
    loader = create_dataloader(dataset, batch_size=64, shuffle=False, drop_last=False)

    models = [
        load_model(n, p, d, device)
        for n, p, d in zip(args.models, args.model_paths, args.emb_dims)
    ]

    n_models = len(models)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    if n_models == 1:
        axes = [axes]

    cmap = plt.colormaps.get_cmap("tab20")
    num_classes = len(np.unique(labels))

    for i, (model, name) in enumerate(zip(models, args.models)):
        emb, y = extract_embeddings(model, loader, device)

        ax = axes[i//2][i%2]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(emb)

        for cls in range(num_classes):
            mask = y == cls

            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=10,
                color=cmap(cls % 20),
                alpha=0.7
            )

        ax.set_title(name)

        # REMOVE ALL AXIS INFORMATION
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()