from flask import Flask
import pandas as pd
import numpy as np
import translatepy as translate
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import keras

tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from gensim.models.doc2vec import TaggedDocument
import re
from bs4 import BeautifulSoup
import nltk

app = Flask(__name__)


def peredict(message):
    nltk.download('punkt')

    df = pd.read_csv('all-data.csv', delimiter=',', encoding='latin-1')

    df = df.rename(columns={'neutral': 'sentiment',
                            'According to Gran , the company has no plans to move all production to Russia , '
                            'although that is where the company is growing .': 'Message'})

    df.index = range(4845)
    df['Message'].apply(lambda x: len(x.split(' '))).sum()

    sentiment = {'positive': 0, 'neutral': 1, 'negative': 2}

    df.sentiment = [sentiment[item] for item in df.sentiment]

    def cleanText(text):
        text = BeautifulSoup(text, "lxml").text
        text = re.sub(r'\|\|\|', r' ', text)
        text = re.sub(r'http\S+', r'<URL>', text)
        text = text.lower()
        text = text.replace('x', '')
        return text

    df['Message'] = df['Message'].apply(cleanText)
    train, test = train_test_split(df, test_size=0.000001, random_state=42)
    from nltk.corpus import stopwords

    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) <= 0:
                    continue
                tokens.append(word.lower())
        return tokens

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

    # The maximum number of words to be used. (most frequent)
    max_fatures = 500000

    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 50

    tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['Message'].values)
    X = tokenizer.texts_to_sequences(df['Message'].values)
    X = pad_sequences(X)
    print('Found %s unique tokens.' % len(X))

    X = tokenizer.texts_to_sequences(df['Message'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    model = keras.models.load_model('model.h5')
    seq = tokenizer.texts_to_sequences(message)

    padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)

    pred = model.predict(padded)

    labels = ['0', '1', '2']
    print(labels[np.argmax(pred)])

    return labels[np.argmax(pred)]


@app.route('/')
def welcome():
    return 'Server is running'


@app.route('/predict/<text>', methods=['GET'])
def predict_news(text):
    labels = ['Positive', 'Neutral', 'Negative']
    translator = translate.Translator()
    text_eng = str(translator.translate(text, "english"))
    print(text_eng)

    return labels[int(peredict([text_eng]))]


if __name__ == '__main__':
    app.run()
