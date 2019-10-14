import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import model_fn
from utils import load_data

seed = 0
np.random.seed(seed)

# model
filename = 'model.h5'

# epochs
epochs = 5

# pre-trained word embedding dimensionality
word_embedding_dim = 50

# number of samples to use for each gradient update
batch_size = 256

# maximum amount of words we will keep in the vocabulary
max_fatures = 5000

# Embedding layer output dimension
embed_dim = 128

# maximum length of all sentences
max_sequence_length = 300

X_train, X_test, Y_train, Y_test, word_index, tokenizer = load_data(
    'imdb.xlsx',
    max_fatures,
    max_sequence_length)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = model_fn(max_sequence_length, max_fatures, embed_dim)

if not os.path.exists('./{}'.format(filename)):

    model.fit(X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1)

    model.save_weights(filename)
else:
    model.load_weights('./{}'.format(filename))

scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))

while True:
    sentence = input(">>")

    if sentence == "exit":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text,
                             maxlen=max_sequence_length,
                             dtype='int32',
                             value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]

    if np.argmax(sentiment) == 0:
        pred_proba = "%.2f%%" % (sentiment[0] * 100)
        print("negative => ", pred_proba)
    elif np.argmax(sentiment) == 1:
        pred_proba = "%.2f%%" % (sentiment[1] * 100)
        print("positive => ", pred_proba)