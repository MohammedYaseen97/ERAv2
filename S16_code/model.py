import torch
import torch.nn as nn
import math

class LayerNormalisation(nn.Module):
    def __init__(self, eps: float=1e-6):
        super(LayerNormalisation, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor):
        return self.embeddings(x) + math.sqrt(self.d_model)

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEmbeddings, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -9e15)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        output = attention_scores @ value
        return output, attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        batch_size = query.shape[0]
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        output, attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output

class EncoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: nn.Dropout
    ):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block.forward(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.layer_norm = LayerNormalisation()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block: MultiHeadAttentionBlock,
            cross_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: nn.Dropout
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.layer_norm = LayerNormalisation()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, d_vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_vocab_size)
    
    def forward(self, x: torch.Tensor):
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, 
            encoder: Encoder, 
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEmbeddings,
            tgt_pos: PositionalEmbeddings,
            projection_layer: ProjectionLayer
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, 
            encoder_output, 
            src_mask,
            tgt,
            tgt_mask
    ):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor):
        return self.projection_layer(x)

def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048
):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            )
        )

    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            )
        )
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer