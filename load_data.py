import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from helper import *

max_len = 35

def preprocess_data(data_path):

    df = pd.read_csv(data_path, header = None)
    df = df.drop([0, 1], axis = 1)
    df = df.applymap(str)
    df.columns = ["Sentiment", "Text"]
    df["Target"] = df["Sentiment"].map({"Negative" : 0, "Positive" : 1, "Neutral" : 2, "Irrelevant" : 3})
    df = process_data(df)
    return df

def load_train_corpus(df):
    train_corpus = build_tweet_corpus(df["Tokens"])
    return train_corpus

def load_train_dataloaders(traindf, train_corpus, batch_size):

    traindata, valdata = train_test_split(traindf, test_size = 0.2, random_state= 321)

    traindata, trainlens = tokenize(traindata, train_corpus, max_len)
    valdata, vallens = tokenize(valdata, train_corpus, max_len)

    trainX = np.stack(traindata["Tokens"])
    trainY = np.array(traindata["Target"])
    validX = np.stack(valdata["Tokens"])
    validY = np.array(valdata["Target"])

    torchtrain = TensorDataset(torch.from_numpy(trainX).to(torch.int64), torch.from_numpy(trainY).to(torch.int64))
    torchtval = TensorDataset(torch.from_numpy(validX).to(torch.int64), torch.from_numpy(validY).to(torch.int64))

    trainloader = DataLoader(torchtrain, shuffle=True, batch_size=batch_size)
    validloader = DataLoader(torchtval, shuffle=True, batch_size=batch_size)
    return trainloader, validloader

def load_test_dataloader(testdf, train_corpus, batch_size):

    testdata, testlens = tokenize(testdf, train_corpus, max_len)

    testX = np.stack(testdata["Tokens"])
    testY = np.array(testdata["Target"])
    torchtest = TensorDataset(torch.from_numpy(testX).to(torch.int64), torch.from_numpy(testY).to(torch.int64))

    testloader = DataLoader(torchtest, shuffle=True, batch_size=batch_size)
    return testloader

