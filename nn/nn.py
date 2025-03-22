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
        nn_arch: List[Dict[str, Union(int, str)]],
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
        #perform single forward pass on a single layer
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        A_curr = self._relu(Z_curr) if activation == 'relu' else self._sigmoid(Z_curr)
        
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
        for layer in self.arch:
            A_curr, Z_curr = self._single_forward(
                self._param_dict['W' + str(layer_idx)],
                self._param_dict['b' + str(layer_idx)],
                A_prev,
                layer['activation']
            )
            A_prev = A_curr
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr
        
        return output, cache

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
        #perform single backprop pass on a single layer
        dZ_curr = self._relu_backprop(dA_curr, Z_curr) if activation_curr == 'relu' else self._sigmoid_backprop(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / self._batch_size
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / self._batch_size
        dA_prev = np.dot(W_curr.T, dZ_curr)
        
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
        # Determine initial gradient from the loss function based on self._loss_func
        if self._loss_func == 'binary_cross_entropy':
            dA = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError('Unsupported loss function: ' + self._loss_func)

        grad_dict = {}
        L = len(self.arch)
        # Backpropagate through layers in reverse order
        for i in reversed(range(1, L + 1)):
            # For the first layer, assume the input is stored as 'A0' in cache
            if i == 1:
                A_prev = cache.get('A0')
            else:
                A_prev = cache['A' + str(i - 1)]

            Z_curr = cache['Z' + str(i)]
            W_curr = self._param_dict['W' + str(i)]
            b_curr = self._param_dict['b' + str(i)]
            activation_curr = self.arch[i - 1]['activation']

            # Compute gradients for the current layer using the single layer backprop method
            dA, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA, activation_curr)

            grad_dict['dW' + str(i)] = dW_curr
            grad_dict['db' + str(i)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for i in range(1, len(self.arch) + 1):
            self._param_dict['W' + str(i)] -= self._lr * grad_dict['dW' + str(i)]
            self._param_dict['b' + str(i)] -= self._lr * grad_dict['db' + str(i)]

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
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        m = X_train.shape[0]

        for epoch in range(self._epochs):
            # Shuffle training data
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            epoch_loss = 0.0
            num_batches = int(np.ceil(m / self._batch_size))

            for batch in range(num_batches):
                start = batch * self._batch_size
                end = min(start + self._batch_size, m)

                # Transpose to match expected input shape (features, batch_size)
                X_batch = X_train_shuffled[start:end].T
                y_batch = y_train_shuffled[start:end].T

                # Store the input as 'A0' in the cache for backpropagation
                cache = {'A0': X_batch}
                # Forward pass
                y_hat, cache_forward = self.forward(X_batch)
                # Merge caches
                cache.update(cache_forward)

                # Compute loss
                if self._loss_func == 'binary_cross_entropy':
                    loss = self._binary_cross_entropy(y_batch, y_hat)
                elif self._loss_func == 'mean_squared_error':
                    loss = self._mean_squared_error(y_batch, y_hat)
                else:
                    raise ValueError('Unsupported loss function: ' + self._loss_func)

                epoch_loss += loss

                # Backpropagation
                grad_dict = self.backprop(y_batch, y_hat, cache)

                # Update parameters
                self._update_params(grad_dict)

            avg_train_loss = epoch_loss / num_batches
            per_epoch_loss_train.append(avg_train_loss)

            # Validation loss computation
            y_val_hat, _ = self.forward(X_val.T)
            if self._loss_func == 'binary_cross_entropy':
                val_loss = self._binary_cross_entropy(y_val.T, y_val_hat)
            elif self._loss_func == 'mean_squared_error':
                val_loss = self._mean_squared_error(y_val.T, y_val_hat)
            per_epoch_loss_val.append(val_loss)

            print(f"Epoch {epoch+1}/{self._epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return per_epoch_loss_train, per_epoch_loss_val

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
        # Transpose input to match expected shape (features, samples)
        y_hat, _ = self.forward(X.T)
        # Transpose output back to (samples, output_dim)
        return y_hat.T

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
        A = self._sigmoid(Z)
        return dA * A * (1 - A)

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
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

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
        return -np.divide(y, y_hat) + np.divide(1 - y, 1 - y_hat)

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
        return np.mean(np.square(y - y_hat))

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
        return y_hat - y