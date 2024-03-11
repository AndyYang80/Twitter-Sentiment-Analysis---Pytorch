
import torch
from torch import nn

class SentimentLSTM(nn.Module):
    def __init__(self, voab_size, embedding_dim, hidden_dim, layers):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(voab_size, embedding_dim)
        self.out_dim = 1
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = layers, batch_first = True)
    
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim, 4)
    
    def forward(self, input):
        batch_size = input.size(0)
        lengths = 35 - (input == 0).sum(dim=1)
        lengths.to(self.device)
        hidden = self.init_hidden(batch_size)
        embeds = self.embedding(input)
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, list(lengths), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(embeds)
        out = self.linear(h[-1])
        
        return out
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        hidden = torch.zeros((self.layers, batch_size, self.hidden_dim)).to(self.device)
        return hidden
    
def load_lstm_model(vocab_size):

    no_layers = 2
    embedding_dim = 64
    hidden_dim = 128

    return SentimentLSTM(vocab_size, embedding_dim, hidden_dim, no_layers)