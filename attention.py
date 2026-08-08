"""
Scripts to reimplement attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

def scaled_dot_product_attention(q, k, v, d_h):
    """ Scaled dot product attention """
    # Q is (B, S, d_k) K is (B, S, d_k), K^T is
    # (B, d_k, S) and so Q @ K^T 
    # Q @ K.T is B x d_k @ d_k x B so B x B
    k_t = einops.rearrange(k, 'B K S -> B S K')
    qk = einops.einsum(q, k_t, 'B S K, B K L -> B S L')
    qk_dk = qk / np.sqrt(d_h)
    qk_dk = torch.nn.functional.softmax(qk_dk, dim = 2)
    return einops.einsum(qk_dk, v, 'B S I, B I K -> B S K')


class MultiHeadAttention(nn.Module):
    """ Multi head attention """
    def __init__(self,num_heads = 8, d_model = 512):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
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
                self.d_k
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
    pos = torch.tensor(pos, dtype=torch.float32)
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
        self.input_embedding = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model) 
        self.MHA = self.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.FFN = self.ModuleList([
            FeedForwardNet(d_model = self.d_model, d_ff = self.d_ff) for i in range(n)
        ])
        
        self.LN_First = self.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        self.LN_Second = self.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        return

    def forward(self, x):

        # Get input embedding (B, S)
        out = self.input_embedding(x)
        # Scale input by sqrt(d_model), (B, S, d_E)
        out *= np.sqrt(d_model)

        # Add positional encoding tensor (B, S, d_E or d_model)
        shape = out.shape

        # Create pos encoding tensor
        pos_encoding = positional_encoding_get_tensor(shape[1], shape[2]).to(out.device)

        # Add pos encoding tensor by first adding batch dim slot and then expanding B times.
        out += pos_encoding.unsqueeze(0).expand(shape[0], *pos_encoding.shape)

        for i in range(self.num_layers):
            attn_out = self.MHA[i](out, out, out) # (B, S, d_E)
            ln_one_out = self.LN_First(out + attn_out)
            
            ffn_out = self.FFN[i](ln_one_out)
            out = self.LN_Second(ln_one_out + ffn_out)

        return out


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
        self.input_embedding = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model) 
        self.MHA = self.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.CrossAttention = self.ModuleList([
            MultiHeadAttention(num_heads = self.num_heads, d_model = self.d_model) for i in range(n)
        ])

        self.FFN = self.ModuleList([
            FeedForwardNet(d_model = self.d_model, d_ff = self.d_ff) for i in range(n)
        ])
        
        self.LN_First = self.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

        self.LN_Second = self.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])
        
        self.LN_Third = self.ModuleList([
            torch.nn.LayerNorm(d_model) for i in range(n)
        ])

    def forward(self, out_encoder, x):
        # Get input embedding (B, S)
        out = self.input_embedding(x)
        # Scale input by sqrt(d_model), (B, S, d_E)
        out *= np.sqrt(d_model)
        # Add positional encoding tensor (B, S, d_E or d_model)
        shape = out.shape
        # Create pos encoding tensor
        pos_encoding = positional_encoding_get_tensor(shape[1], shape[2]).to(out.device)

        # Add pos encoding tensor by first adding batch dim slot and then expanding B times.
        out += pos_encoding.unsqueeze(0).expand(shape[0], *pos_encoding.shape)

        for i in range(self.num_layers):
            attn_out = self.MHA[i](out, out, out) # (B, S, d_E)
            ln_one_out = self.LN_First(out + attn_out)

            cross_attn_out = self.CrossAttention[i](ln_one_out,out_encoder, out_encoder)
            ln_two_out = self.LN_Second(ln_one_out + cross_attn_out)

            
            ffn_out = self.FFN[i](ln_two_out)
            out = self.LN_Third(ln_two_out + ffn_out)



class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, src_seq, target_seq):

