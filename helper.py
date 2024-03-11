import pandas as pd
import numpy as np
import torch
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora
from collections import Counter
pd.options.mode.chained_assignment = None 

device = "cuda" if torch.cuda.is_available() else "cpu"
stopwords_eng = stopwords.words('english')

def preprocess_text(text):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    punct = "[^\w\s]+"
    text = re.sub(link_re_pattern, "", text)
    text = re.sub(punct, "", text)
    return text.lower()

def build_tweet_corpus(data):

    word_counter = build_tweet_counter(data)
    ls = []
    for key in word_counter:
        if word_counter[key] < 5:
            ls.append(key)
    
    stopwords = set(stopwords_eng + ls)
    tweet_corpus = set()
    for x in data:
        for word in x:
            if word not in stopwords:
                tweet_corpus.add(word)
    return list(tweet_corpus)

def build_tweet_counter(data):
    
    tweet_corpus = []
    for x in data:
        for word in x:
            if word not in stopwords_eng:
                tweet_corpus.append(word)
    return Counter(tweet_corpus)


def process_data(df):

    tokenizer = TweetTokenizer()

    df["Text"] = df["Text"].apply(preprocess_text)
    df["Tokens"] = df["Text"].apply(tokenizer.tokenize)
    df["Tokens"] = df["Tokens"].apply(lambda x: [word for word in x if word not in stopwords_eng])

    return df

def tokenize(df, tweet_corpus, max_len):
    corpus_dict = corpora.Dictionary([tweet_corpus]).token2id
    
    def to_tokenids(text):
        tokens = [corpus_dict[x] for x in text if x in corpus_dict]
        if len(tokens) <= 1:
            return "NA"
        else:
            return np.array(tokens)

    df["Tokens"] = df["Tokens"].apply(to_tokenids)
    df = df[df["Tokens"] != "NA"]
    lens = torch.LongTensor([len(x) for x in df["Tokens"]])

    def pad(x):
        if len(x) < max_len:
            x = np.append(x, [0]*(max_len - len(x)))
        return x[0:max_len]

    df['Tokens'] = df["Tokens"].apply(pad)
    return df, lens

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def predict(sentence, model, tweet_corpus, max_len):
    sentence = sentence.lower()
    words = sentence.split(" ")
    corpus_dict = corpora.Dictionary([tweet_corpus]).token2id
    tokens = [corpus_dict[x] for x in words if x in corpus_dict]
    if len(tokens) <= 1:
        print("No Valid Strings!")
        return None
    else:
        tokens = np.array(tokens)

    if len(tokens) < max_len:
        tokens = np.append(tokens, [0]*(max_len - len(tokens)))
        tokens = tokens[0:max_len]
    
    tokens = torch.LongTensor(tokens[None, :])
    output = model(tokens)
    preds = torch.argmax(output, 1)
    return preds