import pandas as pd
import numpy as np
import torch
from torchtext import data
import re
from torch import nn, optim
from gensim import corpora
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_packed_sequence
from helper import *
from load_data import *
from models.CNN import load_cnn_model
from models.RNN import load_rnn_model
from models.LSTM import load_lstm_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_modeltype(modeltype, vocab_size):
    modelfunc = {
        "CNN" : load_cnn_model,
        "RNN" : load_rnn_model,
        "LSTM" : load_lstm_model
    }

    # load model
    model = modelfunc[modeltype](vocab_size)
    model.to(device)
    return model


def train_model(model_type, train_data, train_corpus, epochs, batch_size = 16, lr=0.002):
    
    savefolder = {"CNN" : "working/CNN/state_dict.pt",
                  "RNN" : "working/RNN/state_dict.pt",
                  "LSTM" : "working/LSTM/state_dict.pt"}
    
    model_type = model_type.upper()

    print("Generating Dataloaders...")
    trainloader, validloader = load_train_dataloaders(train_data, train_corpus, batch_size)
    vocab_size = len(train_corpus)

    print("loading model...")
    model = load_modeltype(model_type, vocab_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    valid_loss_min = np.Inf

    epoch_tr_loss, epoch_vl_loss = [],[]
    epoch_tr_acc, epoch_vl_acc = [],[]

    print("beginning training...")
    for epoch in range(epochs):
        train_losses = []
        val_losses = []
        train_acc = 0.0
        corr = 0
        tot = 0

        corrval = 0
        totval = 0
        model.train()
        # initialize hidden state 
        for inputs, labels in trainloader:
            labels = labels.to(torch.int64)
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()

            output = model(inputs)
            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            preds = torch.argmax(output, 1)
            corr += (preds == labels).sum().item()
            tot += 16

        for val_inputs, val_labels in validloader:
            val_labels = val_labels.to(torch.int64)
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_output = model(val_inputs)
            val_loss = criterion(val_output, val_labels)
            val_losses.append(val_loss.item())
            # calculate the loss and perform backprop
            val_preds = torch.argmax(val_output, 1)
            corrval += (val_preds == val_labels).sum().item()
            totval += 16

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss}')
        print(f'val_loss: {epoch_val_loss}')
        print(f'train_accuracy : {corr/tot*100}')
        print(f"valid accuracy: {corrval/totval*100}")
        
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)

        epoch_tr_acc.append(corr/tot*100)
        epoch_vl_acc.append(corrval/totval*100)

        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), savefolder[model_type])
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
            valid_loss_min = epoch_val_loss
    
    results_dataframe = pd.DataFrame({"Epoch" : range(0,epochs),
                                      "Train Loss" : epoch_tr_loss,
                                      "Validation Loss": epoch_vl_loss,
                                      "Train Accuracy" : epoch_tr_acc,
                                      "Validation Accuracy" : epoch_vl_acc
                                      })
    return results_dataframe

def test_model(model_type, test_data, train_corpus, batch_size = 16):

    model_type = model_type.upper()
    testloader = load_test_dataloader(test_data, train_corpus, batch_size)
    
    vocab_size = len(train_corpus)
    model = load_modeltype(model_type, vocab_size)

    corrval = 0
    totval = 0

    modelload = {
        "CNN" : 'working/CNN/state_dict.pt',
        "RNN" : 'working/RNN/state_dict.pt',
        "LSTM" : 'working/LSTM/state_dict.pt'
    }

    model.load_state_dict(torch.load(modelload[model_type]))
    model.to(device)

    for inputs, labels in testloader:
        labels = labels.to(torch.int64)
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        # calculate the loss and perform backprop
        preds = torch.argmax(output, 1)
        corrval += (preds == labels).sum().item()
        totval += 16

    print(f"The test accuacy is: {corrval/totval}")
