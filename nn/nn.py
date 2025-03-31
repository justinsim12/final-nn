# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if activation.lower() == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation.lower() == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Unsupported activation function: " + activation)
        return A_curr, Z_curr


    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        # Transpose X: input shape [batch_size, features] -> [features, batch_size]
        A_prev = X.T
        cache["A0"] = A_prev
        L = len(self.arch)
        for i, layer in enumerate(self.arch):
            layer_idx = i + 1
            W_curr = self._param_dict["W" + str(layer_idx)]
            b_curr = self._param_dict["b" + str(layer_idx)]
            activation = layer['activation']
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            cache["Z" + str(layer_idx)] = Z_curr
            cache["A" + str(layer_idx)] = A_curr
            A_prev = A_curr
        return A_prev, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation_curr.lower() == 'sigmoid':
            dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() == 'relu':
            dZ = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError("Unsupported activation function: " + activation_curr)
        m = A_prev.shape[1]
        dW_curr = np.dot(dZ, A_prev.T) / m
        db_curr = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ)
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        if self._loss_func == "binary_cross_entropy":
            dA = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func in ["mse", "mean_squared_error"]:
            dA = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError("Unsupported loss function: " + self._loss_func)
        grad_dict = {}
        L = len(self.arch)
        for i in reversed(range(L)):
            layer_idx = i + 1
            A_prev = cache["A" + str(i)]
            Z_curr = cache["Z" + str(layer_idx)]
            W_curr = self._param_dict["W" + str(layer_idx)]
            activation_curr = self.arch[i]['activation']
            dA, dW, db = self._single_backprop(W_curr,
                                               self._param_dict["b" + str(layer_idx)],
                                               Z_curr,
                                               A_prev,
                                               dA,
                                               activation_curr)
            grad_dict["dW" + str(layer_idx)] = dW
            grad_dict["db" + str(layer_idx)] = db
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        L = len(self.arch)
        for i in range(1, L+1):
            self._param_dict["W" + str(i)] -= self._lr * grad_dict["dW" + str(i)]
            self._param_dict["b" + str(i)] -= self._lr * grad_dict["db" + str(i)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Transpose y_train and y_val to match the expected shape (features, n_samples)
        y_train = y_train.T
        y_val = y_val.T

        train_losses = []
        val_losses = []
        m = X_train.shape[0]
        for epoch in range(self._epochs):
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[:, permutation]
            epoch_loss = 0
            num_batches = int(np.ceil(m / self._batch_size))
            for i in range(num_batches):
                start = i * self._batch_size
                end = min((i+1) * self._batch_size, m)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[:, start:end]
                y_hat, cache = self.forward(X_batch)
                if self._loss_func == "binary_cross_entropy":
                    loss = self._binary_cross_entropy(y_batch, y_hat)
                elif self._loss_func in ["mse", "mean_squared_error"]:
                    loss = self._mean_squared_error(y_batch, y_hat)
                else:
                    raise ValueError("Unsupported loss function: " + self._loss_func)
                epoch_loss += loss
                grad_dict = self.backprop(y_batch, y_hat, cache)
                self._update_params(grad_dict)
            epoch_loss /= num_batches
            train_losses.append(epoch_loss)
            # Evaluate on validation set
            y_hat_val, _ = self.forward(X_val)
            if self._loss_func == "binary_cross_entropy":
                val_loss = self._binary_cross_entropy(y_val, y_hat_val)
            elif self._loss_func in ["mse", "mean_squared_error"]:
                val_loss = self._mean_squared_error(y_val, y_hat_val)
            else:
                raise ValueError("Unsupported loss function: " + self._loss_func)
            val_losses.append(val_loss)
        return train_losses, val_losses

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat
    
    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig = self._sigmoid(Z)
        return dA * sig * (1 - sig)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
        m = y.shape[1]
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
        return -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean((y - y_hat) ** 2)


    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[1]
        return 2 * (y_hat - y) / m