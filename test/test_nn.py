import pytest
import numpy as np
from nn.nn import NeuralNetwork

def test_single_forward():
    nn_arch = [{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    W = nn._param_dict['W1']
    b = nn._param_dict['b1']
    A_prev = np.array([[1], [2], [3], [4]])
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'relu')
    assert A_curr.shape == (3, 1)
    assert Z_curr.shape == (3, 1)

def test_forward():
    nn_arch = [{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'},
               {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    X = np.array([[1, 2, 3, 4]])
    output, cache = nn.forward(X.T)
    assert output.shape == (1, 1)
    assert 'A1' in cache and 'Z1' in cache

def test_single_backprop():
    nn_arch = [{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    W = nn._param_dict['W1']
    b = nn._param_dict['b1']
    Z_curr = np.random.randn(3, 1)
    A_prev = np.random.randn(4, 1)
    dA_curr = np.random.randn(3, 1)
    dA_prev, dW_curr, db_curr = nn._single_backprop(W, b, Z_curr, A_prev, dA_curr, 'relu')
    assert dA_prev.shape == (4, 1)
    assert dW_curr.shape == (3, 4)
    assert db_curr.shape == (3, 1)

def test_predict():
    nn_arch = [{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'},
               {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    X = np.array([[1, 2, 3, 4]])
    y_hat = nn.predict(X)
    assert y_hat.shape == (1, 1)

def test_binary_cross_entropy():
    nn = NeuralNetwork([], lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    loss = nn._binary_cross_entropy(y, y_hat)
    assert loss > 0

def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork([], lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    assert dA.shape == y.shape

def test_mean_squared_error():
    nn = NeuralNetwork([], lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    loss = nn._mean_squared_error(y, y_hat)
    assert loss > 0

def test_mean_squared_error_backprop():
    nn = NeuralNetwork([], lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    assert dA.shape == y.shape

def test_sample_seqs():
    # Placeholder for sampling sequences test
    pass

def test_one_hot_encode_seqs():
    # Placeholder for one-hot encoding sequences test
    pass