import csv
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
import numpy as np

import time
from data import csv_utils

np.set_printoptions(linewidth=120)


def build_autoencoder(input_shape, nb_encoding_features):
    model = models.Sequential()

    #model.add(layers.Dense(input_shape[0], input_shape=input_shape, activation='sigmoid'))
    model.add(layers.Dense(nb_encoding_features, input_shape=input_shape, activation='sigmoid'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(nb_encoding_features, activation='sigmoid'))

    encoding_layer = model.layers[-1].name
    model.add(layers.Dense(input_shape[0], activation='sigmoid'))

    model.compile(optimizer='adam',
              #loss=tensorflow.keras.losses.CategoricalCrossentropy(),
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              #loss='mse',
              metrics=['accuracy'],
    )

    encoding_layer = 'dense'
    print(model.summary())
    return model, encoding_layer

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

def create_noisy_train_data(tokenizer, cases):
    '''
    Creates the training data. Each input patient cases is repeated a number of times given by the function number_noisy_inputs. For each replication, one icd code is dropped out.'''
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

def create_test_data(tokenizer, cases):
    number_icds = len(tokenizer.word_index) + 1
    number_cases = len(cases)
    seqs = tokenizer.texts_to_sequences(cases)

    X = np.zeros((number_cases, number_icds))
    for n, seq in enumerate(seqs):
        X[n, tuple(seq)] = 1
    return X



if __name__ == '__main__':

    train = csv_utils.parse_csv('train.csv', skip_noninformative_icds=True)
    test = csv_utils.parse_csv('test.csv', skip_noninformative_icds=False)

    #train = _parse_csv('train.csv', skip_noninformative_icds=True)
   # test = _parse_csv('test.csv', skip_noninformative_icds=False)

    train_cases = list(train.values())
    test_cases = list(test.values())

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(train_cases + test_cases)
    number_icds = len(tokenizer.word_index) + 1

    X = create_noisy_train_data(tokenizer, train_cases)
    X_test = create_test_data(tokenizer, test_cases)

    input_shape = (number_icds, )
    #nb_encoding_features = int(number_icds ** (1/4) )
    nb_encoding_features = 128

    print(f'Nmuber of encding features: {nb_encoding_features}')

    model, encoding_layer = build_autoencoder(input_shape, nb_encoding_features)


    feature_layer = tensorflow.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(encoding_layer).output
    )


    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0.01,
        mode='max',
        restore_best_weights=True
    )

    model.fit(x=X,
              y=X,
              epochs=3,
              batch_size=256,
              validation_data=(X_test, X_test),
              callbacks=[early_stopping],
              verbose=True,
    )

    test_icds = model(X_test).numpy()

    nb_recommendations = 5

    t = tokenizer
    recommendations = []

    for pred, case in zip(test_icds, X_test):

        icds = np.where(case == 1)[0]

        # since we want new icd codes, remove the known codes from the prediction
        pred[icds] = 0

        # get the 5 best recommendations
        icd_recs = np.argsort(pred)[-nb_recommendations:]

        # transform them back into icd codes
        icd_recs_list = [tokenizer.index_word[rec] for rec in icd_recs]
        icd_recs_str = ','.join(icd_recs_list)

        recommendations.append(icd_recs_str)

    csv_utils.write_recommendations(
    #_write_recommendations(
        patients=test.keys(),
        recommendations=recommendations
    )

