import numpy as np

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
    '''
    Creates the training data. Each input patient case is repeated
    a number of times given by the function number_noisy_inputs.
    For each replication, one icd code is dropped out.
    '''
    cases = list(train.values())

    seqs = tokenizer.texts_to_sequences(cases)

    size = sum(number_noisy_inputs(seq) for seq in seqs)
    number_icds = len(tokenizer.word_index) + 1

    shape = (size, number_icds)
    X = np.zeros(shape)

    i=0
    for seq in seqs:
        n = number_noisy_inputs(seq)
        dropouts = np.random.choice(seq, n, replace=False)
        for dropout in dropouts:
            x = np.zeros(shape)
            noisy = seq[:]
            noisy.remove(dropout)

            X[i, noisy] = 1
            i+=1

    return X


def create_test_data(tokenizer, test):
    cases = list(test.values())

    number_icds = len(tokenizer.word_index) + 1
    number_cases = len(cases)
    seqs = tokenizer.texts_to_sequences(cases)

    X = np.zeros((number_cases, number_icds))
    for n, seq in enumerate(seqs):
        X[n, tuple(seq)] = 1
    return X


