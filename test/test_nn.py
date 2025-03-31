import pytest
import numpy as np
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

def test_single_forward():
# Define a minimal network architecture for one layer.
    nn_arch = [{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'}]
    lr = 0.01
    seed = 42
    batch_size = 2
    epochs = 1
    loss_function = "mse"  # Loss function is irrelevant here

    # Instantiate the neural network (assuming NeuralNetwork is already defined/imported)
    nn = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

    # Override parameters with known values.
    # W1: shape (2, 3) and b1: shape (2, 1)
    nn._param_dict["W1"] = np.array([[1, 2, 3],
                                     [4, 5, 6]], dtype=float)
    nn._param_dict["b1"] = np.array([[1],
                                     [1]], dtype=float)

    # Define A_prev (the input activation) with shape (3, 2)
    # For example, two samples: first sample [1, 3, 5] and second sample [2, 4, 6]
    A_prev = np.array([[1, 2],
                       [3, 4],
                       [5, 6]], dtype=float)

    # Expected computation:
    # For sample 1:
    #   Z1 = W1 dot [1,3,5] + b1
    #      = [1*1 + 2*3 + 3*5, 4*1 + 5*3 + 6*5] + [1, 1]
    #      = [1+6+15, 4+15+30] + [1, 1]
    #      = [22, 49] + [1, 1] = [23, 50]
    #
    # For sample 2:
    #   Z2 = W1 dot [2,4,6] + b1
    #      = [1*2 + 2*4 + 3*6, 4*2 + 5*4 + 6*6] + [1, 1]
    #      = [2+8+18, 8+20+36] + [1, 1]
    #      = [28, 64] + [1, 1] = [29, 65]
    #
    # Since we are using ReLU and all computed values are positive,
    # A should be identical to Z.
    expected_Z = np.array([[23, 29],
                           [50, 65]], dtype=float)
    expected_A = expected_Z.copy()

    # Call the _single_forward function
    A, Z = nn._single_forward(nn._param_dict["W1"], nn._param_dict["b1"], A_prev, "relu")

    # Compare the outputs to expected values
    assert np.allclose(A, expected_A), f"Activation mismatch. Expected: {expected_A}, Got: {A}"
    assert np.allclose(Z, expected_Z), f"Z mismatch. Expected: {expected_Z}, Got: {Z}"

def test_forward():
    import numpy as np

    # Define a two-layer architecture:
    # Layer 1: input_dim=3, output_dim=2, activation=ReLU
    # Layer 2: input_dim=2, output_dim=1, activation=Sigmoid
    nn_arch = [
        {'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    lr = 0.01
    seed = 42
    batch_size = 2
    epochs = 1
    loss_function = "mse"  # not used in forward, so any loss will do

    # Create the network instance (assumes NeuralNetwork is imported)
    nn = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

    # Manually set known parameters for reproducibility.
    # Layer 1 parameters: W1 (2x3) and b1 (2x1)
    nn._param_dict["W1"] = np.array([[1, 2, 3],
                                     [4, 5, 6]], dtype=float)
    nn._param_dict["b1"] = np.array([[1],
                                     [1]], dtype=float)
    # Layer 2 parameters: W2 (1x2) and b2 (1x1)
    nn._param_dict["W2"] = np.array([[1, -1]], dtype=float)
    nn._param_dict["b2"] = np.array([[0.5]], dtype=float)

    # Define input X with shape (batch_size, features) = (2, 3)
    # Two samples: sample1: [1, 2, 3] and sample2: [4, 5, 6]
    X = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)

    # --- Expected Calculation ---
    # The forward pass transposes X, so:
    # A0 = X.T = [[1, 4],
    #             [2, 5],
    #             [3, 6]]
    #
    # For Layer 1:
    #   Z1 = W1 dot A0 + b1
    # For sample 1 (first column):
    #   Z1_sample1 = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] + [1, 1]
    #              = [1+4+9, 4+10+18] + [1, 1] = [14, 32] + [1, 1] = [15, 33]
    # For sample 2 (second column):
    #   Z1_sample2 = [1*4 + 2*5 + 3*6, 4*4 + 5*5 + 6*6] + [1, 1]
    #              = [4+10+18, 16+25+36] + [1, 1] = [32, 77] + [1, 1] = [33, 78]
    #
    # Since Layer 1 uses ReLU, and all values are positive, A1 = Z1:
    #   A1 = [[15, 33],
    #         [33, 78]]
    #
    # For Layer 2:
    #   Z2 = W2 dot A1 + b2, where W2 = [[1, -1]] and b2 = [[0.5]]
    # For sample 1:
    #   Z2_sample1 = 1*15 + (-1)*33 + 0.5 = 15 - 33 + 0.5 = -17.5
    # For sample 2:
    #   Z2_sample2 = 1*33 + (-1)*78 + 0.5 = 33 - 78 + 0.5 = -44.5
    #
    # Finally, Layer 2 uses sigmoid activation:
    #   A2 = sigmoid(Z2) = 1 / (1 + exp(-Z2))
    # So, expected output:
    expected_A2 = np.array([
        [1 / (1 + np.exp(17.5)), 1 / (1 + np.exp(44.5))]
    ])

    # Run the forward pass
    output, cache = nn.forward(X)

    # Check that the output matches the expected result (with an appropriate tolerance)
    assert np.allclose(output, expected_A2, atol=1e-7), f"Forward pass output mismatch. Expected: {expected_A2}, Got: {output}"

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
    import numpy as np

    # Define a simple one-layer architecture: input_dim=3, output_dim=1, activation=sigmoid.
    nn_arch = [{'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.01
    seed = 42
    batch_size = 2
    epochs = 1
    loss_function = "binary_cross_entropy"  # Loss function is not used in predict

    # Instantiate the network (assuming NeuralNetwork is already defined/imported)
    nn = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

    # Manually set the parameters for reproducibility.
    # For layer 1: W1 is shape (1, 3) and b1 is shape (1, 1)
    nn._param_dict["W1"] = np.array([[1, -1, 2]], dtype=float)
    nn._param_dict["b1"] = np.array([[0.5]], dtype=float)

    # Define the input X with shape (batch_size, features) = (2, 3)
    # Two samples: sample 1: [1, 2, 3], sample 2: [4, 5, 6]
    X = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)

    # --- Expected Calculation ---
    # The forward pass transposes X, so A0 = X.T of shape (3, 2)
    # For sample 1 (column [1, 2, 3]):
    #   Z = 1*1 + (-1)*2 + 2*3 + 0.5 = 1 - 2 + 6 + 0.5 = 5.5
    # For sample 2 (column [4, 5, 6]):
    #   Z = 1*4 + (-1)*5 + 2*6 + 0.5 = 4 - 5 + 12 + 0.5 = 11.5
    # Apply the sigmoid activation: sigmoid(z) = 1 / (1 + exp(-z))
    expected_output = np.array([[1/(1 + np.exp(-5.5)), 1/(1 + np.exp(-11.5))]])

    # Call predict to obtain the network's output.
    y_pred = nn.predict(X)

    # Check that the predicted output matches the expected values.
    assert np.allclose(y_pred, expected_output, atol=1e-7), f"Prediction mismatch. Expected: {expected_output}, Got: {y_pred}"

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
    import numpy as np

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Define a small dataset with an imbalance:
    # Positive sequences: "seq1", "seq2"
    # Negative sequences: "seq3", "seq4", "seq5"
    seqs = ["seq1", "seq2", "seq3", "seq4", "seq5"]
    labels = [True, True, False, False, False]

    # Call the sample_seqs function (assumes it's imported from preprocess)
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # Calculate counts for each class in the sampled data
    count_pos = sum(1 for lab in sampled_labels if lab)
    count_neg = sum(1 for lab in sampled_labels if not lab)

    # The sampling should balance the classes:
    assert count_pos == count_neg, f"Balanced classes expected, got {count_pos} positives and {count_neg} negatives"

    # The expected number per class is the maximum of the original counts.
    expected_per_class = max(labels.count(True), labels.count(False))
    expected_total = 2 * expected_per_class
    assert len(sampled_seqs) == expected_total, f"Expected {expected_total} samples, got {len(sampled_seqs)}"

    # Verify that every sampled sequence is from the original list
    for seq in sampled_seqs:
        assert seq in seqs, f"Sampled sequence {seq} not in original sequences"

def test_one_hot_encode_seqs():
    import numpy as np

    # Define a small set of DNA sequences to encode.
    seq_arr = ["AG", "CT"]
    # Expected one-hot encoding:
    # For "AG":
    #   A -> [1, 0, 0, 0]
    #   G -> [0, 0, 0, 1]
    # => Combined: [1, 0, 0, 0, 0, 0, 0, 1]
    # For "CT":
    #   C -> [0, 0, 1, 0]
    #   T -> [0, 1, 0, 0]
    # => Combined: [0, 0, 1, 0, 0, 1, 0, 0]
    expected_output = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0]
    ])

    # Call the one_hot_encode_seqs function (assumes it's imported from preprocess)
    encodings = one_hot_encode_seqs(seq_arr)

    # Verify the shape: should be (number of sequences, sequence_length * 4)
    expected_shape = (len(seq_arr), len(seq_arr[0]) * 4)
    assert encodings.shape == expected_shape, f"Expected shape {expected_shape}, got {encodings.shape}"

    # Compare the output to the expected output.
    assert np.allclose(encodings, expected_output), f"Expected encoding: {expected_output}, Got: {encodings}"