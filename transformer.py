import torch
import torch.nn as nn
from decoder import TransformerBlock
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Embedding Matrix [vocab_size, hidden_dim]
embedding_matrix = model.transformer.wte.weight

# Position Matrix [max_seq_len, hidden_dim]
position_matrix = model.transformer.wpe.weight

# Pretrained Weights Pipeline
print(model.transformer)

class GPT2Transformer(nn.Module):
    def __init__(self, num_layers=12, attention_heads=12, hidden_size=768, vocab_size=50257):
        self.curr_len = 0
        self.N = num_layers
        self.d_model = hidden_size
        self.h = attention_heads
        self.vocab_size = vocab_size

        # Initialize Decoders
        self.transformer_blocks = {x: 0 for x in range(1, self.N + 1)}

        for i in range(self.N):
            self.transformer_blocks[i + 1] = TransformerBlock(self.d_model, self.h)

        # Get Embedding Matrix for Vocabulary
        self.embedding_matrix = embedding_matrix

        # Positional Embedding Matrix 
        self.positional_embeddings = position_matrix
        
        # Pretrained Weights into Decoders

        # Output Head
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.1)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, tokens):
        # 1. Convert input sequence into token IDs using Byte-Piece Tokenizer (BPE)

        # Token Embeddings
        token_embeddings = torch.gather(input=self.embedding_matrix, dim=0, index=tokens)

        # Add positional embeddings
        positional_embeddings = self.positional_embeddings[:tokens.shape[1]]

        # Add embeddings
        x = token_embeddings + positional_embeddings
        x = self.dropout(x)

        # 2. Pass Embeddings Through Transformer Blocks
        for i in range(self.N):
            x = self.transformer_blocks[i + 1](x)
            
        # 3. Obtain Logits by Projecting the Output to the Vocab Size.
        logits = self.layer_norm(self.fc_out(x))

        return logits
    
    
