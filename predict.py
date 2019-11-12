import os
import pickle

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import settings
from model import model_fn

seed = 0
np.random.seed(seed)

# loading saved tokenizer
with open(settings.tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)

vocabulary_size = len(tokenizer.word_counts.keys())+1

model = model_fn(vocabulary_size=vocabulary_size,
                 embedding_size=settings.embedding_size,
                 max_sequence_length=settings.max_sequence_length)

# loading saved model
if os.path.exists(settings.filename):
    model.load_weights('./{}'.format(settings.filename))

while True:
    sentence = input(">>")

    if sentence == "exit":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text,
                             maxlen=settings.max_words,
                             dtype='int32',
                             value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0][0]
    if sentiment > 0.5:
        pred_proba = "%.2f%%" % (sentiment * 100)
        print("positive => ", pred_proba)
    else:
        pred_proba = "%.2f%%" % (sentiment * 100)
        print("negative => ", pred_proba)


