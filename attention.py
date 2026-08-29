"""
Scripts to reimplement attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

def scaled_dot_product_attention(q, k, v, d_h, mask=False):
    """ Scaled dot product attention """
    # Q is (B, S, d_k) K is (B, S, d_k), K^T is
    # (B, d_k, S) and so Q @ K^T 
    # Q @ K.T is B x d_k @ d_k x B so B x B
    k_t = einops.rearrange(k, 'B K S -> B S K')
    qk = einops.einsum(q, k_t, 'B S K, B K L -> B S L')
    qk_dk = qk / torch.sqrt(d_h)

    if mask:
        mask_idx = torch.triu_indices(qk_dk.shape[1], qk_dk.shape[2], offset=1)
        qk_dk[:, mask_idx[0], mask_idx[1]] = float("-inf")

    qk_dk = torch.nn.functional.softmax(qk_dk, dim = 2)
    return einops.einsum(qk_dk, v, 'B S I, B I K -> B S K')


class MultiHeadAttention(nn.Module):
    """ Multi head attention """
    def __init__(self,num_heads = 8, d_model = 512, mask=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mask = mask
        # Q, K, and V are all size B, S, d_model project to different heads
        self.d_k = d_model // num_heads
        self.head_q = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(d_model, self.d_k)) for n in range(num_heads))
        self.head_k = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(d_model, self.d_k)) for n in range(num_heads))
        self.head_v = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(d_model, self.d_k)) for n in range(num_heads))

        for q, k, v in zip(self.head_q, self.head_k, self.head_v):
            torch.nn.init.xavier_uniform_(q)
            torch.nn.init.xavier_uniform_(k)
            torch.nn.init.xavier_uniform_(v)

        self.concat_mat = torch.nn.init.xavier_uniform_(torch.nn.Parameter(torch.empty(d_model, d_model)))
        

    def forward(self, q, k, v):
        concat = []
        for i in range(self.num_heads):
            attn = scaled_dot_product_attention(
                torch.matmul(q, self.head_q[i]),
                torch.matmul(k, self.head_k[i]),
                torch.matmul(v, self.head_v[i]),
                self.d_k,
                self.mask
            ) # (B, S, d_k)
            concat.append(attn) 
        
        return torch.matmul(torch.cat(concat, dim = 2), self.concat_mat)

class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin_one = torch.nn.Linear(d_model, d_ff, bias=True)
        self.lin_two = torch.nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.lin_two(torch.nn.ReLU(self.lin_one(x)))

def positional_encoding_get_val(pos, i, d_model):
    """ Fxn to create the value of the positional encoding """
    if i % 2 == 0:
        return torch.sin(pos/(10000 ** (2 * (i // 2) / d_model)))
    else:
        return torch.cos(pos/(10000 ** (2 * (i // 2) / d_model)))


def positional_encoding_get_tensor(seq_len, d_model):
    """ Creates a torch tensor """
    return torch.tensor([[positional_encoding_get_val(i, j, d_model) for j in range(d_model)] for i in range(seq_len)])



class TransformerEncoder(nn.Module):
    """ This is the transformer encoder module."""
    def __init__(self, d_model = 512, num_heads = 8, n=6, vocab_size = 10000,
                 d_ff = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = n
        self.vocab_size = vocab_size 
        self.d_ff = d_ff

        # Input to embedding is (B, S) and output is (B, S, d_E)
        self.MHA = torch.nn.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.FFN = torch.nn.ModuleList([
            FeedForwardNet(d_model = self.d_model, d_ff = self.d_ff) for i in range(n)
        ])
        
        self.LN_First = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        self.LN_Second = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        return

    def forward(self, x):
        for i in range(self.num_layers):
            attn_out = self.MHA[i](x, x, x) # (B, S, d_E)
            ln_one_out = self.LN_First[i](x + attn_out)
            
            ffn_out = self.FFN[i](ln_one_out)
            x = self.LN_Second[i](ln_one_out + ffn_out)
        return x 


class TransformerDecoder(nn.Module):
    """ Transformer Decoder Stack"""
    def __init__(self, d_model = 512, num_heads = 8, n=6, vocab_size = 10000,
                 d_ff = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = n
        self.vocab_size = vocab_size 
        self.d_ff = d_ff

        # Input to embedding is (B, S) and output is (B, S, d_E)
        self.MHA = torch.nn.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.CrossAttention = torch.nn.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.FFN = torch.nn.ModuleList([
            FeedForwardNet(d_model = self.d_model, d_ff = self.d_ff) for i in range(n)
        ])
        
        self.LN_First = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        self.LN_Second = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])
        
        self.LN_Third = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

    def forward(self, out_encoder, x):
        for i in range(self.num_layers):
            attn_out = self.MHA[i](x, x, x, mask=True) # (B, S, d_E)
            ln_one_out = self.LN_First[i](x + attn_out)

            cross_attn_out = self.CrossAttention[i](ln_one_out,out_encoder, out_encoder)
            ln_two_out = self.LN_Second[i](ln_one_out + cross_attn_out)

            
            ffn_out = self.FFN[i](ln_two_out)
            x = self.LN_Third[i](ln_two_out + ffn_out)
        return x


class Transformer(nn.Module):
    def __init__(self,d_model = 512, num_heads = 8, n=6, vocab_size = 10000,
                 d_ff = 2048):
        super().__init__()
        self.input_embedding_src = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model) 
        self.input_embedding_tgt = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model) 
        self.transformer_encoder =  TransformerEncoder()
        self.transformer_decoder = TransformerDecoder()
        self.lin_layer = torch.nn.Linear(d_model, vocab_size)

        return

    def forward(self, src_seq, target_seq):
        # Get input embedding (B, S)
        src_seq = self.input_embedding_src(src_seq)
        tgt_seq = self.input_embedding_tgt(tgt_seq)
        # Scale input by sqrt(d_model), (B, S, d_E)
        src_seq *= np.sqrt(self.d_model)
        tgt_seq *= np.sqrt(self.d_model)
        # Add positional encoding tensor (B, S, d_E or d_model)
        src_shape = src_seq.shape
        tgt_shape = tgt_seq.shape

        # Create pos encoding tensor
        pos_encoding = positional_encoding_get_tensor(src_shape[1], src_shape[2]).to(src_seq.device)
        pos_encoding = positional_encoding_get_tensor(tgt_shape[1], tgt_shape[2]).to(tgt_seq.device)

        # Add pos encoding tensor by first adding batch dim slot and then expanding B times.
        src_seq += pos_encoding.unsqueeze(0).expand(src_shape[0], *pos_encoding.shape)
        tgt_seq += pos_encoding.unsqueeze(0).expand(tgt_shape[0], *pos_encoding.shape)
        
        # Call encoder
        src_encoding = self.transformer_encoder(src_seq)

        # Call Decoder
        out = self.transformer_decoder(src_encoding, target_seq)
        out = self.lin_layer(out)

        return out
