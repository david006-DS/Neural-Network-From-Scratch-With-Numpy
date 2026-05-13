"""
Training script for the NumPy MLP on MNIST.

Usage
-----
    python train.py
    python train.py --epochs 30 --lr 0.01 --batch_size 128

Requirements: numpy, matplotlib  (no deep-learning frameworks)
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import numpy as np

from src.data_loader import load_mnist
from src.neural_net import NeuralNetwork


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST")
    p.add_argument("--epochs",     type=int,   default=20,   help="Training epochs")
    p.add_argument("--lr",         type=float, default=0.01, help="Learning rate")
    p.add_argument("--batch_size", type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--hidden",     type=int,   nargs="+",    default=[128, 64],
                   help="Hidden layer sizes, e.g. --hidden 256 128")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: dict) -> None:
    """Save and display training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Training History — NumPy MLP on MNIST", fontsize=13)

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Val loss",  linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train acc")
    ax.plot(epochs, history["val_acc"],   label="Val acc",  linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/training_history.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Animated Training Visualization
# ---------------------------------------------------------------------------

def animate_training(history: dict) -> None:
    """Create animated training curves."""

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    line1, = ax.plot([], [], label="Train Loss")
    line2, = ax.plot([], [], label="Validation Loss")

    ax.set_xlim(1, len(epochs))

    all_losses = history["train_loss"] + history["val_loss"]
    ax.set_ylim(min(all_losses) * 0.9, max(all_losses) * 1.1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Animation")

    ax.legend()
    ax.grid(True, alpha=0.3)

    def update(frame):
        x = epochs[:frame + 1]

        y1 = history["train_loss"][:frame + 1]
        y2 = history["val_loss"][:frame + 1]

        line1.set_data(x, y1)
        line2.set_data(x, y2)

        return line1, line2

    ani = FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=500,
        blit=True,
        repeat=False,
    )

    out_path = "outputs/training_animation.gif"

    ani.save(out_path, writer="pillow", fps=2)

    print(f"Animated training visualization saved to {out_path}")

    plt.show()

def show_predictions(model: NeuralNetwork, X_test: np.ndarray, y_test_int: np.ndarray,
                     n: int = 20) -> None:
    """Display a grid of test images with predicted vs true labels."""
    indices = np.random.choice(len(X_test), n, replace=False)
    preds = model.predict(X_test[indices])
    true  = y_test_int[indices]

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))
    fig.suptitle("Predictions on test set", fontsize=12)
    for ax, pred, truth, img in zip(axes.flat, preds, true, X_test[indices]):
        ax.imshow(img.reshape(28, 28), cmap="gray", interpolation="nearest")
        color = "green" if pred == truth else "red"
        ax.set_title(f"p={pred} t={truth}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path = "outputs/sample_predictions.png"
    plt.savefig(out_path, dpi=120)
    print(f"Prediction grid saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Neural Network Architecture Diagram
# ---------------------------------------------------------------------------

def draw_network(layer_sizes: list[int]) -> None:
    """Draw and save a neural network architecture diagram."""

    fig, ax = plt.subplots(figsize=(14, 8))

    n_layers = len(layer_sizes)
    v_spacing = 1
    h_spacing = 3

    neuron_positions = []

    # Draw neurons
    for i, layer_size in enumerate(layer_sizes):

        layer_positions = []

        y_offset = (layer_size - 1) / 2

        for j in range(layer_size):

            x = i * h_spacing
            y = -j * v_spacing + y_offset

            circle = Circle((x, y), radius=0.12, fill=True)

            ax.add_patch(circle)

            layer_positions.append((x, y))

        neuron_positions.append(layer_positions)

    # Draw connections
    for i in range(n_layers - 1):

        for (x1, y1) in neuron_positions[i]:

            for (x2, y2) in neuron_positions[i + 1]:

                ax.plot([x1, x2], [y1, y2], linewidth=0.3)

    # Labels
    labels = ["Input"]

    for i in range(1, n_layers - 1):
        labels.append(f"Hidden {i}")

    labels.append("Output")

    for i, label in enumerate(labels):

        ax.text(
            i * h_spacing,
            max(layer_sizes) / 2 + 2,
            label,
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title("Neural Network Architecture", fontsize=18, pad=30)
    ax.axis("off")

    out_path = "outputs/network_architecture.png"

    plt.tight_layout(pad=3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Network diagram saved to {out_path}")

    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Data
    X_train, y_train, X_test, y_test, _, y_test_int = load_mnist()

    # Use 10 k of training data as a validation split
    val_size  = 10_000
    X_val     = X_train[:val_size]
    y_val     = y_train[:val_size]
    X_tr      = X_train[val_size:]
    y_tr      = y_train[val_size:]

    # 2. Model
    layer_sizes = [784] + args.hidden + [10]
    print(f"\nNetwork architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Epochs: {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}\n")

    model = NeuralNetwork(layer_sizes, seed=42)
    draw_network([32, 16, 8, 10])


    # 3. Train
    history = model.train(
        X_tr, y_tr,
        X_val, y_val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # 4. Final test evaluation
    test_acc = np.mean(model.predict(X_test) == y_test_int)
    print(f"\nFinal test accuracy: {test_acc * 100:.2f}%")

    # 5. Plots
    plot_history(history)
    animate_training(history)
    show_predictions(model, X_test, y_test_int, n=50)


if __name__ == "__main__":
    main()
