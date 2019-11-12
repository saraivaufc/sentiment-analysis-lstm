import re

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

import settings


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

    return string.strip().lower()


def load_data(file, max_sequence_length):
    data = pd.read_excel(file)

    data = data[['text', 'sentiment']]

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: clean_str(x))
    data['text'] = data['text'].apply(
        (lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    stop_words = set(stopwords.words('english'))
    text = []
    for row in data['text'].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)

    tokenizer = Tokenizer(lower=True, split=' ')

    tokenizer.fit_on_texts(text)

    X = tokenizer.texts_to_sequences(text)

    X = pad_sequences(X, maxlen=max_sequence_length)

    word_index = tokenizer.word_index

    Y = data['sentiment']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,
                                                        random_state=42)

    return X_train, X_test, Y_train, Y_test, word_index, tokenizer
