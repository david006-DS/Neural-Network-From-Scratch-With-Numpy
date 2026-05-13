"""
Pure NumPy Feedforward Neural Network (MLP)
Architecture: configurable layers with ReLU hidden activations and Softmax output.
Trained with mini-batch SGD and categorical cross-entropy loss.
"""

import numpy as np


class NeuralNetwork:
    """
    Multi-layer perceptron built from scratch with NumPy.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons per layer, including input and output.
        Example: [784, 128, 64, 10] for MNIST.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and output layer.")
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight matrices

        # He initialization for ReLU layers
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He init: scale = sqrt(2 / fan_in)
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 where z > 0, else 0."""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (subtract row-wise max)."""
        shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute the forward pass.

        Returns
        -------
        y_pred : np.ndarray, shape (batch, num_classes)
            Softmax probabilities.
        cache : dict
            Intermediate values needed for backpropagation.
        """
        cache = {"A": [X]}  # A[0] = input
        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            if i < self.num_layers - 1:
                A = self.relu(Z)
            else:
                A = self.softmax(Z)  # output layer
            cache.setdefault("Z", []).append(Z)
            cache["A"].append(A)
        return A, cache

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Categorical cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray, shape (batch, classes)  — softmax probabilities
        y_true : np.ndarray, shape (batch, classes)  — one-hot encoded labels
        """
        batch = y_pred.shape[0]
        # Clip to avoid log(0)
        log_probs = -np.log(np.clip(y_pred, 1e-12, 1.0))
        return float(np.sum(y_true * log_probs) / batch)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(
        self,
        y_true: np.ndarray,
        cache: dict,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute gradients via backpropagation.

        Returns
        -------
        dW : list of weight gradients (same order as self.weights)
        db : list of bias gradients
        """
        batch = y_true.shape[0]
        dW = [None] * self.num_layers
        db = [None] * self.num_layers

        # Gradient of cross-entropy + softmax combined: dZ_out = y_pred - y_true
        dA = cache["A"][-1] - y_true  # (batch, out)

        for i in reversed(range(self.num_layers)):
            A_prev = cache["A"][i]          # (batch, fan_in)
            Z      = cache["Z"][i]          # (batch, fan_out)

            if i < self.num_layers - 1:
                # ReLU gradient
                dZ = dA * self.relu_grad(Z)
            else:
                # Softmax gradient already folded in above
                dZ = dA

            dW[i] = (A_prev.T @ dZ) / batch
            db[i] = dZ.mean(axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T     # propagate upstream

        return dW, db

    # ------------------------------------------------------------------
    # Parameter update (SGD)
    # ------------------------------------------------------------------

    def update_params(
        self,
        dW: list[np.ndarray],
        db: list[np.ndarray],
        lr: float,
    ) -> None:
        for i in range(self.num_layers):
            self.weights[i] -= lr * dW[i]
            self.biases[i]  -= lr * db[i]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 64,
    ) -> dict:
        """
        Train the network and return history.

        Parameters
        ----------
        X_train, y_train : training data (X: float32, y: one-hot)
        X_val, y_val     : validation data
        epochs           : number of full passes over training data
        lr               : learning rate
        batch_size       : mini-batch size

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss',
                  'train_acc', 'val_acc'
        """
        history = {
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
        }
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            # Mini-batch SGD
            for start in range(0, n, batch_size):
                Xb = X_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                y_pred, cache = self.forward(Xb)
                dW, db = self.backward(yb, cache)
                self.update_params(dW, db, lr)

            # --- Metrics (computed on full sets) ---
            train_pred, _ = self.forward(X_train)
            val_pred,   _ = self.forward(X_val)

            t_loss = self.cross_entropy_loss(train_pred, y_train)
            v_loss = self.cross_entropy_loss(val_pred,   y_val)
            t_acc  = self._accuracy(train_pred, y_train)
            v_acc  = self._accuracy(val_pred,   y_val)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss: {t_loss:.4f}  val_loss: {v_loss:.4f}  "
                f"acc: {t_acc:.4f}  val_acc: {v_acc:.4f}"
            )

        return history

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))
