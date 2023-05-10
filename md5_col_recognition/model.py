import torch.nn as nn
import torch
import math
class LSTM(nn.Module):

    def __init__(self, weight_matrix, TEXT):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # torch.Size([200, 8, 300])
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2 
        return y