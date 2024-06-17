
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.attention import MultiHeadAttention
from models.modules.embedding import PositionalEmbedding
from models.modules.feedforward import PositionWiseFeedForward

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, permutation, src_mask=None, tgt_mask=None):
        # Apply the permutation to the target sequence
        tgt = tgt[permutation]

        attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.cross_attn(tgt, memory, memory, mask=src_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)

        # Revert the permutation
        tgt = tgt[torch.argsort(permutation)]

        return tgt

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.src_embedding = PositionalEmbedding(config.vocab_size, config.d_model, config.max_seq_len)
        self.tgt_embedding = PositionalEmbedding(config.vocab_size, config.d_model, config.max_seq_len)
        self.encoder = TransformerEncoder(config.d_model, config.num_heads, config.d_ff, config.dropout)
        self.decoder = TransformerDecoder(config.d_model, config.num_heads, config.d_ff, config.dropout)
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src, tgt, permutation, src_mask=None, tgt_mask=None):
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)

        memory = self.encoder(src_embedded, src_mask)
        output = self.decoder(tgt_embedded, memory, permutation, src_mask, tgt_mask)

        output = self.linear(output)

        return output

    def generate(self, src, max_len, start_symbol, end_symbol, pad_symbol, device):
        src = src.to(device)
        batch_size = src.size(0)
        seq_len = max_len

        # Generate the random permutation
        permutation = torch.randperm(seq_len, device=device)

        # Initialize the generated sequence with start symbols
        generated = torch.full((batch_size, seq_len), pad_symbol, dtype=torch.long, device=device)
        generated[:, 0] = start_symbol

        for i in range(1, seq_len):
            # Apply the permutation to the generated sequence
            permuted_generated = generated[:, permutation]

            # Create the target mask
            tgt_mask = self.create_tgt_mask(permuted_generated, pad_symbol)

            # Get the output probabilities for the current position
            output = self.forward(src, permuted_generated, permutation, tgt_mask=tgt_mask)
            output = output[:, i-1]  # Select the output for the current position

            # Sample the next token based on the output probabilities
            next_token = torch.multinomial(F.softmax(output, dim=-1), num_samples=1)

            # Update the generated sequence at the corresponding position
            generated[:, permutation[i]] = next_token.squeeze()

            # Check if all sequences have reached the end symbol
            if (next_token == end_symbol).all():
                break

        return generated

    def create_tgt_mask(self, tgt, pad_symbol):
        tgt_mask = (tgt != pad_symbol).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & torch.tril(torch.ones_like(tgt_mask, dtype=torch.bool), diagonal=0)
        return tgt_mask