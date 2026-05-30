import torch
import torch.nn as nn
from math import sqrt, pi

class TransformerBlock(nn.Module):
    def __init__(self, num_layers, batch_size, seq_len, hidden_size, num_heads, inner_size, device, pretrained_weights=None):
        super().__init__()
        self.N = batch_size
        self.seq_len = seq_len
        self.A = num_heads # number of attention heads
        self.num_layers = num_layers # number of layers
        self.H = hidden_size # hidden_size
        self.dropout = nn.Dropout(0.1).to(device)

        ## Multi-Head Attention
        # Normal distribution with mean 0 and std 0.02 for weights
        # For residual layers, 0.02 / sqrt(2 * num_layers) to prevent accumulation of residual 
        # contributes and activations from growing too large.
        self.WQ = nn.Parameter(torch.empty(hidden_size, hidden_size )).to(device)
        torch.nn.init.normal_(self.WQ, mean=0.0, std=0.02)
        self.bQ = nn.Parameter(torch.zeros((hidden_size,))).to(device) # (d_model,)
        self.WK = nn.Parameter(torch.empty(hidden_size, hidden_size )).to(device)
        torch.nn.init.normal_(self.WK, mean=0.0, std=0.02)
        self.bK = nn.Parameter(torch.zeros((hidden_size,))).to(device) # (d_model,)
        self.WV = nn.Parameter(torch.empty(hidden_size, hidden_size )).to(device)
        torch.nn.init.normal_(self.WV, mean=0.0, std=0.02).to(device)
        self.bV = nn.Parameter(torch.zeros((hidden_size,))).to(device) # (d_model,)
        self.WO = nn.Parameter(torch.empty(hidden_size, hidden_size)).to(device) 
        torch.nn.init.normal_(self.WO, mean=0.0, std=0.02 / sqrt(2 * self.num_layers)).to(device)
        self.bO = nn.Parameter(torch.zeros((hidden_size,))).to(device) # (h * d_v), note d_v = d_model
    
        ## Multi-Layer Perceptron (MLP)
        self.W1 = nn.Parameter(torch.empty(inner_size, hidden_size)).to(device)
        torch.nn.init.normal_(self.W1, mean=0.0, std=0.02)
        self.W2 = nn.Parameter(torch.empty(hidden_size, inner_size)).to(device)
        torch.nn.init.normal_(self.W2, mean=0.0, std=0.02 / sqrt(2 * self.num_layers))
        self.b1 = nn.Parameter(torch.zeros((inner_size, ))).to(device)
        self.b2 = nn.Parameter(torch.zeros((hidden_size, ))).to(device)

        self.curr_seq_len = 0

        # Causal Mask
        self.mask = None

        self.device = device

        # Layer Normalization
        self.eps = 1e-5
        self.shift_1 = nn.Parameter(torch.zeros(hidden_size, )).to(device)
        self.scale_1 = nn.Parameter(torch.ones(hidden_size, )).to(device)
        self.shift_2 = nn.Parameter(torch.zeros(hidden_size, )).to(device)
        self.scale_2 = nn.Parameter(torch.ones(hidden_size, )).to(device)

        # Load pre-trained weights
        if pretrained_weights is not None:
            # Load in Layer Norm parameters
            self.scale_1.data = pretrained_weights[0].weight.data.to(device)
            self.shift_1.data = pretrained_weights[0].bias.data.to(device)
            self.scale_2.data = pretrained_weights[6].weight.data.to(device)
            self.shift_2.data = pretrained_weights[6].bias.data.to(device)

            # Attention weights
            c_attn_weight = pretrained_weights[2].weight.data.to(device)
            c_attn_bias = pretrained_weights[2].bias.data.to(device)

            WQ, WK, WV = c_attn_weight.split(hidden_size, dim=1)
            self.WQ.data = WQ.to(device)
            self.WK.data = WK.to(device)
            self.WV.data = WV.to(device)
            self.bQ.data, self.bK.data, self.bV.data = c_attn_bias.split(hidden_size, dim=0)

            self.WO.data = pretrained_weights[3].weight.data.to(device)
            self.bO.data = pretrained_weights[3].bias.data.to(device)

            self.bQ.to(device)
            self.bK.to(device)
            self.bV.to(device)

            # MLP weights
            self.W1.data = pretrained_weights[8].weight.data.T.to(device)
            self.b1.data = pretrained_weights[8].bias.data.to(device)
            self.W2.data = pretrained_weights[9].weight.data.T.to(device)
            self.b2.data = pretrained_weights[9].bias.data.to(device)


    def MaskedMultiHeadSelfAttention(self, X):
        """ 
        Applies Masked Multi Head Self-Attention with causal masking to the input.
        """
        # (batch_size, seq_len, hidden_dim)
        Q = X @ self.WQ + self.bQ
        K = X @ self.WK + self.bK
        V = X @ self.WV + self.bV

        ## Linearly project into different subspaces (heads)
        # (batch_size, num_heads, seq_len, hidden_dim // num_heads)
        Q = Q.view(Q.shape[0], Q.shape[1], self. A, self.H // self.A).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.A, self.H // self.A).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.A, self.H // self.A).transpose(1, 2)
        
        ## Masked Multi Head Self-Attention
        # (batch_size, num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / sqrt(self.H // self.A)
        # apply causal masking to prevent tokens from attending to future positions
        scores = scores + self.mask 
        # compute attention weights
        attn_weights = self.dropout(nn.functional.softmax(scores, dim=-1)) @ V

        ## Concatenate output of each head.
        # (batch_size, seq_len, hidden_size)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(attn_weights.shape[0], self.curr_seq_len, self.H)

        O = attn_weights @ self.WO + self.bO

        return O

    def MLP(self, X):
        """
        Standard Multi-Layer Perceptron 
        """
        X = self.GeLU(X @ self.W1.T + self.b1)
        X = X @ self.W2.T + self.b2
        return X
        
    def GeLU(self, X):
        """
        Applies GELU (Gaussian Error Linear Unit) activation function 
        """
        return 0.5 * X * (1 + torch.tanh(sqrt(2 / pi) * (X + 0.044715 * torch.pow(X, 3))))

    def LayerNormalization(self, X, scale, shift):
        """
        Applies Layer Normalization to input, performing a shift and scale operation.
        """
        # Calculate mean and variance over feature dimension
        mean = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True, unbiased=False)

        # Normalize input
        X_normalized = (X - mean) / torch.sqrt(var + self.eps)
        
        # Apply shift and scale
        y = scale * X_normalized + shift

        return y
    
    def causal_mask(self, curr_seq_len):
        """
        Creates causal mask given the current sequence length.
        """
        self.curr_seq_len = curr_seq_len
        self.mask = torch.tril(torch.ones((curr_seq_len, curr_seq_len)), diagonal=0).to(self.device)
        self.mask = self.mask.masked_fill(self.mask == 0, float("-inf"))
        self.mask = self.mask.masked_fill(self.mask == 1, 0)

    def forward(self, X):
        # X is (batch_size, seq_len, hidden_size)
        self.causal_mask(X.shape[1])
        X1 = self.dropout(self.MaskedMultiHeadSelfAttention(self.LayerNormalization(X, self.scale_1, self.shift_1))) + X
        X2 = self.dropout(self.MLP(self.LayerNormalization(X1, self.scale_2, self.shift_2))) + X1

        return X2