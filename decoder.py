import torch
import torch.nn as nn
from math import sqrt

class TransformerBlock(nn.Module):
    def __init__(self, num_layers, batch_size, seq_len, hidden_size, num_heads, inner_size):
        super().__init__()
        self.N = batch_size
        self.seq_len = seq_len
        self.A = num_heads # number of attention heads
        self.num_layers = num_layers # number of layers
        self.H = hidden_size # hidden_size
        self.droput = nn.Dropout(0.1)

        ## Multi-Head Attention
        # Normal distribution with mean 0 and std 0.02 for weights
        # For residual layers, 0.02 / sqrt(2 * num_layers) to prevent accumulation of residual 
        # contributes and activations from growing too large.
        self.WQ = nn.Parameter(torch.empty(hidden_size, hidden_size ))
        torch.nn.init.normal_(self.WQ, mean=0.0, std=0.02)
        self.bQ = nn.Parameter(torch.zeros((hidden_size,))) # (d_model,)
        self.WK = nn.Parameter(torch.empty(hidden_size, hidden_size )) 
        torch.nn.init.normal_(self.WK, mean=0.0, std=0.02)
        self.bK = nn.Parameter(torch.zeros((hidden_size,))) # (d_model,)
        self.WV = nn.Parameter(torch.empty(hidden_size, hidden_size )) 
        torch.nn.init.normal_(self.WV, mean=0.0, std=0.02)
        self.bV = nn.Parameter(torch.zeros((hidden_size,))) # (d_model,)
        self.WO = nn.Parameter(torch.empty(hidden_size, hidden_size)) 
        torch.nn.init.normal_(self.WO, mean=0.0, std=0.02 / sqrt(2 * N))
        self.bO = nn.Parameter(torch.zeros((hidden_size,))) # (h * d_v), note d_v = d_model
    
        ## Multi-Layer Perceptron (MLP)
        self.W1 = nn.Parameter(torch.empty(inner_size, hidden_size))
        torch.nn.init.normal_(self.W1, mean=0.0, std=0.02)
        self.W2 = nn.Parameter(torch.empty(hidden_size, inner_size))
        torch.nn.init.normal_(self.W2, mean=0.0, std=0.02 / sqrt(2 * self.num_layers))
        self.b1 = nn.Parameter(torch.zeros((inner_size, )))
        self.b2 = nn.Parameter(torch.zeros((hidden_size, )))

        # Create mask
        self.mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=1)
        self.mask = self.mask.masked_fill(self.mask == 0, float("-inf"))
        self.mask = self.mask.masked_fill(self.mask == 1, 0)

        # Layer Normalization
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.zeros(hidden_size, ))
        self.scale = nn.Parameter(torch.ones(hidden_size, ))

    def MaskedMultiHeadSelfAttention(self, X):
        # (batch_size, seq_len, hidden_dim)
        Q = X @ self.WQ + self.bQ 
        V = X @ self.WV + self.bV 
        K = X @ self.WK + self.bK
        
        # (batch_size, num_heads, seq_len, hidden_size // num_heads)
        Q = Q.view(Q.shape[0], self.A, self.seq_len, self.H // self.A)
        K = K.view(K.shape[0], self.A, self.seq_len, self.H // self.A)
        V = V.view(V.shape[0], self.A, self.seq_len, self.H // self.A)
        
        # Scores are (batch_size, num_heads, seq_len, seq_len)
        scores = (Q @ K.T / (sqrt(self.H // self.A)))
        scores_masked = scores + self.mask
        attn_weights = torch.nn.functional.softmax(scores_masked)
        attn_weights = self.dropout(attn_weights)

        Z = attn_weights @ V
        Z = torch.view(Z.shape[0], self.seq_len, self.H)
        O = Z @ self.WO + self.bO

        return O

    def MLP(self, X):
        X = self.GeLU(X @ self.W1 + self.b1)
        X = X @ self.W2 + self.b2
        return X
        
    def GeLU(self, X):
        return 0.5 * X * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (X + 0.044715 * torch.pow(X, 3))))

    def LayerNormalization(self, X):
        # Calculate mean and variance over feature dimension
        mean = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True)

        # Normalize input
        X_normalized = (X - mean) / torch.sqrt(var + self.eps)
        
        # Apply shift and scale
        y = self.scale * X_normalized + self.shift

    def forward(self, X):
        X1 = self.MaskedMultiHeadSelfAttention(X)
        X1 = self.LayerNormalization(X + self.dropout(X1))
        X2 = self.MLP(X1)
        X2 = self.LayerNormalization(X1 + self.dropout(X2))

        return X2