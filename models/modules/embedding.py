
import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('positional_encoding', self.get_positional_encoding(max_seq_len, d_model))

    def get_positional_encoding(self, max_seq_len, d_model):
        positional_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :seq_len]
        return x