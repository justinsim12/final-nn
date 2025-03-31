# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate sequences by label
    pos_seqs = [s for s, lab in zip(seqs, labels) if lab]
    neg_seqs = [s for s, lab in zip(seqs, labels) if not lab]
    
    # Determine target size as the maximum of both classes
    target = max(len(pos_seqs), len(neg_seqs))
    
    # Oversample with replacement if needed
    if len(pos_seqs) < target:
        pos_seqs = list(np.random.choice(pos_seqs, size=target, replace=True))
    if len(neg_seqs) < target:
        neg_seqs = list(np.random.choice(neg_seqs, size=target, replace=True))
    
    # Create corresponding labels
    pos_labels = [True] * target
    neg_labels = [False] * target
    
    # Combine the sequences and labels from both classes
    sampled_seqs = pos_seqs + neg_seqs
    sampled_labels = pos_labels + neg_labels
    
    # Shuffle the combined lists to mix the classes
    indices = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in indices]
    sampled_labels = [sampled_labels[i] for i in indices]
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define the mapping for nucleotides to one-hot encoding
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }
    
    encoded_seqs = []
    for seq in seq_arr:
        one_hot = []
        for char in seq.upper():
            # Use mapping if the nucleotide is known; else, use [0, 0, 0, 0]
            one_hot.extend(mapping.get(char, [0, 0, 0, 0]))
        encoded_seqs.append(one_hot)
    
    return np.array(encoded_seqs)