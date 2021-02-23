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
import data_utils

np.set_printoptions(linewidth=120)


def create_autoencoder(input_shape, nb_encoding_features):
    model = models.Sequential()

    #model.add(layers.Dense(512, input_shape=input_shape, activation='sigmoid'))
    model.add(layers.Dense(nb_encoding_features, input_shape=input_shape, activation='sigmoid'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(nb_encoding_features, activation='sigmoid'))

    model.add(layers.Dense(input_shape[0], activation='sigmoid'))
    #model.add(layers.Dense(input_shape[0], activation='relu'))


    adam = tensorflow.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
    )

    nadam = tensorflow.keras.optimizers.Nadam(
        learning_rate=0.0004,
        beta_1=0.9,
        beta_2=0.999,
    )


    model.compile(optimizer=adam,
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              #loss='mse',
              metrics=['accuracy'],
    )

    return model


def create_tokenizer(train, test):

    train_cases = list(train.values())
    test_cases = list(test.values())

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(train_cases + test_cases)
    return tokenizer


def get_recommendations(test_icds, nb_recommendations = 5):

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
    return recommendations


if __name__ == '__main__':

    # data parsing
    train = csv_utils.parse_csv('train.csv', skip_noninformative_icds=True)
    test = csv_utils.parse_csv('test.csv', skip_noninformative_icds=False)

    # create training and test set based on input data
    tokenizer = create_tokenizer(train, test)

    X = data_utils.create_noisy_train_data(tokenizer, train)
    X_test = data_utils.create_test_data(tokenizer, test)

    # build autoencoder neural network with 1 hidden layer
    model = create_autoencoder(
        input_shape=(len(tokenizer.word_index) + 1,),
        nb_encoding_features=256,
    )
    print(model.summary())
    print('optimizer:', model.optimizer.get_config())

    # fit the model, use early stopping for regularization
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        min_delta=0.01,
        mode='max',
        restore_best_weights=True
    )

    model.fit(x=X,
              y=X,
              epochs=100,
              batch_size=256,
              validation_data=(X_test, X_test),
              callbacks=[early_stopping],
              verbose=True,
    )


    # use the model to create the recommendations
    test_icds = model(X_test).numpy()
    recommendations = get_recommendations(test_icds)

    csv_utils.write_recommendations(
        patients=test.keys(),
        recommendations=recommendations
    )

