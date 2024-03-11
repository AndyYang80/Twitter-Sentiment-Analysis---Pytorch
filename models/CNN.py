import torch
from torch import nn

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SentimentCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out_dim = 1

        self.convs = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(2,2), padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(3, 5, kernel_size=(2,2), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten()
        )
    
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2970, 4)
    
    def forward(self, input):
        
        batch_size = input.size(0)
        embeds = self.embedding(input)
        embeds = torch.unsqueeze(embeds, 1)
        out = self.convs(embeds)
        out = self.linear(out)
        
        return out
    
def load_cnn_model(vocab_size):

    embedding_dim = 64

    return SentimentCNN(vocab_size, embedding_dim)