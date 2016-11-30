# -*- coding: utf-8 -*-
'''An implementation of sentence to sentence learning
Input: "I see dead people"
Output: "I see dead people"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be inverted, shown to increase performance in many tasks

'''

from __future__ import print_function

import numpy as np
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, RepeatVector, recurrent
from keras.layers import Dense, Masking
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.visualize_util import plot
from six.moves import range

MAX_NB_WORDS = 400000
MAX_SEQUENCE_LENGTH = 10
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

TRAINING_SIZE = 100000
DIGITS = 3
INVERT = False

RNN = recurrent.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 256
LAYERS = 1


class WordTable(object):
    def __init__(self, word_index, maxlen):

        self.word_index_reverse = dict((value, key) for key, value in word_index.items())
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(word_index) + 1))
        for i, c in enumerate(C):
            X[i, c] = 1
        return X

    def decode(self, X, calc_argmax=False):

        if calc_argmax:
            X = X.argmax(axis=-1)

        return ' '.join(self.word_index_reverse[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


print('Generating data...')

f = open('/home/erincmer/new_sentences2.txt', "r")
lines = f.readlines()
texts = []
for x in lines:
    texts.append(x)
f.close()

texts = texts[:TRAINING_SIZE]
questions = texts
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
word_index[" "] = 0
print('Found %s unique tokens.' % len(word_index))

wtable = WordTable(word_index, MAX_SEQUENCE_LENGTH)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

EMBEDDING_DIM = 50
embeddings_index = {}
f = open('/home/erincmer/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix[0] = np.zeros(EMBEDDING_DIM)

print('Total addition questions:', len(questions))

print('Vectorization...')

X = np.zeros((len(questions), MAX_SEQUENCE_LENGTH), dtype=np.bool)
y = np.zeros((len(questions), MAX_SEQUENCE_LENGTH), dtype=np.bool)
X = data
y = data

indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
print(X[0])
# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10

y = y[:, :, np.newaxis]

(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()

model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))

model.add(Masking(mask_value=0.))
model.add(RNN(HIDDEN_SIZE))

model.add(RepeatVector(MAX_SEQUENCE_LENGTH))

# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(TimeDistributed(Dense(len(word_index) + 1)))

model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

plot(model, to_file='modelRNN.png', show_shapes="true")

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                        validation_data=(X_val, y_val))

    model.save('my_model_256.h5')
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = wtable.decode(rowX[0])
        outputY = np.squeeze(rowy[0])
        correct = wtable.decode(outputY)
        guess = wtable.decode(preds[0], calc_argmax=False)

        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
