"""
Utilities for creating the training and test data.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
DTYPE = np.float32


def number_noisy_inputs(seq):
    N = len(seq)
    if N < 3:
        return 1
    if N < 5:
        return 2
    if N < 10:
        return 2
    if N < 15:
        return 3
    else:
        return 3


def create_noisy_train_data(tokenizer, train):
    """
    Creates the training data. Each input patient case is repeated
    a number of times given by the function number_noisy_inputs.
    For each replication, one icd code is dropped out.
    """
    print('Creating training data')
    cases = list(train.values())

    seqs = tokenizer.texts_to_sequences(cases)

    size = sum(number_noisy_inputs(seq) for seq in seqs)
    number_icds = len(tokenizer.word_index) + 1

    shape = (size, number_icds)
    X_corrupted = np.zeros(shape, dtype=DTYPE)
    X_true = np.zeros(shape, dtype=DTYPE)

    i = 0
    for seq in seqs:
        n = number_noisy_inputs(seq)
        dropouts = np.random.choice(seq, n, replace=False)
        for dropout in dropouts:
            noisy = seq[:]
            noisy.remove(dropout)

            X_corrupted[i, noisy] = 1
            X_true[i, seq] = 1
            i += 1

    return X_corrupted, X_true


def create_test_data(tokenizer, test):
    print('Creating test samples')
    cases = list(test.values())

    number_icds = len(tokenizer.word_index) + 1
    number_cases = len(cases)
    seqs = tokenizer.texts_to_sequences(cases)

    X = np.zeros((number_cases, number_icds), dtype=DTYPE)
    for n, seq in enumerate(seqs):
        X[n, tuple(seq)] = 1
    return X


def create_tokenizer(train, test):

    train_cases = list(train.values())
    test_cases = list(test.values())

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(train_cases + test_cases)
    return tokenizer
