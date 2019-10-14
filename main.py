import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import model_fn
from utils import load_data

seed = 0
np.random.seed(seed)

# O model será exportado para este arquivo
filename = 'model.h5'

epochs = 5

# dimensionalidade do word embedding pré-treinado
word_embedding_dim = 50

# número de amostras a serem utilizadas em cada atualização do gradiente
batch_size = 256

# Reflete a quantidade máxima de palavras que iremos manter no vocabulário
max_fatures = 5000

# dimensão de saída da camada Embedding
embed_dim = 128

# limitamos o tamanho máximo de todas as sentenças
max_sequence_length = 300

X_train, X_test, Y_train, Y_test, word_index, tokenizer = load_data(
    'imdb.xlsx',
    max_fatures,
    max_sequence_length)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = model_fn(max_sequence_length, max_fatures, embed_dim)

if not os.path.exists('./{}'.format(filename)):

    hist = model.fit(
        X_train,
        Y_train,
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
    sentence = input(">")

    if sentence == "exit":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text,
                             maxlen=max_sequence_length,
                             dtype='int32',
                             value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        pred_proba = "%.2f%%" % (sentiment[0] * 100)
        print("negativo => ", pred_proba)
    elif (np.argmax(sentiment) == 1):
        pred_proba = "%.2f%%" % (sentiment[1] * 100)
        print("positivo => ", pred_proba)
