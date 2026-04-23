import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(data_path):
    with np.load(data_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
    return x_train, y_train

def save_all_class_grids(x_train, y_train, num_classes=30, out_dir="data/class_grids"):
    os.makedirs(out_dir, exist_ok=True)

    for c in range(num_classes):
        class_imgs = x_train[y_train == c]
        n = len(class_imgs)

        if n == 0:
            print(f"Třída {c} je prázdná.")
            continue

        # mřížka (přibližně čtvercová)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

        # když je jen 1 řádek/sloupec, axes nemusí být 2D
        axes = np.array(axes).reshape(rows, cols)

        idx = 0
        for r in range(rows):
            for col in range(cols):
                ax = axes[r, col]
                ax.axis("off")

                if idx < n:
                    ax.imshow(class_imgs[idx], cmap="gray")
                else:
                    ax.set_visible(False)

                idx += 1

        plt.suptitle(f"Třída {c}", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"class_{c}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Uloženo: {out_path}")

def plot_class_means(x_train, y_train, num_classes=30, output_path="data/class_means.png"):
    img_h, img_w = x_train.shape[1], x_train.shape[2]

    mean_images = []

    for c in range(num_classes):
        class_imgs = x_train[y_train == c]

        if len(class_imgs) == 0:
            mean_img = np.zeros((img_h, img_w))
        else:
            mean_img = np.mean(class_imgs, axis=0)

        mean_images.append(mean_img)

    mean_images = np.array(mean_images)

    # vykreslení
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(num_classes):
        axes[i].imshow(mean_images[i], cmap="gray")
        axes[i].set_title(f"Class {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    data_path = "data/siam_xkiszk00.npz"

    x_train, y_train = load_data(data_path)
    plot_class_means(x_train, y_train)
    save_all_class_grids(x_train, y_train)