from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


def model_fn(max_sequence_length, max_fatures, embed_dim):
    model = Sequential()

    model.add(Embedding(max_fatures,
                        embed_dim,
                        input_length=max_sequence_length,
                        name="embedding"))

    model.add(LSTM(embed_dim,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   name="lstm"))

    model.add(Dense(2, activation='softmax', name="softmax"))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model
