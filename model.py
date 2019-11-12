from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.models import Sequential


def model_fn(vocabulary_size, embedding_size, max_sequence_length):
    model = Sequential()

    model.add(Embedding(vocabulary_size,
                        embedding_size,
                        input_length=max_sequence_length,
                        name="embedding"))

    model.add(LSTM(200, name="lstm"))

    model.add(Dense(1, activation='sigmoid', name="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model
