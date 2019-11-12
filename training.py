import os

import numpy as np

import settings
from model import model_fn
from utils import load_data
import pickle

seed = 0
np.random.seed(seed)

# epochs
epochs = 20

# number of samples to use for each gradient update
batch_size = 512

X_train, X_test, Y_train, Y_test, word_index, tokenizer = load_data(
    'data/imdb.xlsx',
    settings.max_sequence_length)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# saving tokenizer
with open(settings.tokenizer_file, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocabulary_size = len(tokenizer.word_counts.keys())+1

model = model_fn(vocabulary_size=vocabulary_size,
                 embedding_size=settings.embedding_size,
                 max_sequence_length=settings.max_sequence_length)

# loadin saved model
if os.path.exists(settings.filename):
    model.load_weights('./{}'.format(settings.filename))

for i in range(batch_size):
    print(X_train.shape)
    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              epochs=1,
              batch_size=batch_size,
              shuffle=True,
              verbose=1)

    # saving model
    model.save_weights(settings.filename)

# evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))