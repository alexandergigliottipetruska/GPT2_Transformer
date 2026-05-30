import torch
import torch.nn as nn
from decoder import TransformerBlock
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import CONFIG_SMALL, CONFIG_XL

def extract_pretrained_weights(config, num_layers):
    """
    Extract pretrained weights for different model configurations.
    """
    print(num_layers)
    if config == "XL":  
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Embedding Matrix [vocab_size, hidden_dim]
    embedding_matrix = model.transformer.wte.weight

    # Position Matrix [max_seq_len, hidden_dim]
    position_matrix = model.transformer.wpe.weight

    # Pretrained Weights Pipeline
    pretrained_weights = [model.transformer.ln_f, model.lm_head]
    block_pretrained_weights = {k:[] for k in range(0, num_layers)}

    curr_block = 0
    for name, module in model.named_modules():
        if re.search(r"transformer\.h\.\d+\..+", name):
            block_pretrained_weights[curr_block].append(module)

            if re.search(r"transformer\.h\.\d+\.mlp\.dropout", name):
                curr_block += 1

    return block_pretrained_weights, embedding_matrix, position_matrix, pretrained_weights

class GPT2Transformer(nn.Module):

    def __init__(self, num_layers=12, batch_size=512, attention_heads=12, hidden_size=768, vocab_size=50257, seq_len=1024, config=None, device="cpu"):
        super().__init__()
        self.curr_len = 0
        # Default is small. 
        if config is None:
            self.N = CONFIG_SMALL["num_layers"]
            self.d_model = CONFIG_SMALL["hidden_size"]
            self.h = CONFIG_SMALL["num_heads"]
        elif config == "XL":
            self.N = CONFIG_XL["num_layers"]
            self.d_model = CONFIG_XL["hidden_size"]
            self.h = CONFIG_XL["num_heads"]

        block_pretrained_weights, embedding_matrix, position_matrix, pretrained_weights = extract_pretrained_weights(config, self.N)
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.inner_size = 4 * self.d_model

        # Initialize Decoders
        self.transformer_blocks = nn.ModuleDict({str(x): None for x in range(0, self.N)})
        
        for i in range(self.N):
            self.transformer_blocks[str(i)] = TransformerBlock(num_layers=self.N,
                                                          seq_len=self.seq_len,
                                                          hidden_size=self.d_model, 
                                                          num_heads=self.h, 
                                                          inner_size=self.inner_size,
                                                          batch_size=batch_size,
                                                          pretrained_weights=block_pretrained_weights[i],
                                                          device=device)

        # Get Embedding Matrix for Vocabulary
        self.embedding_matrix = nn.Parameter(embedding_matrix, requires_grad=False)

        # Positional Embedding Matrix 
        self.positional_embeddings = nn.Parameter(position_matrix, requires_grad=False)

        # Output Head
        self.fc_out = nn.Linear(self.d_model, vocab_size, bias=False)
        self.fc_out.load_state_dict({'weight': pretrained_weights[1].weight})
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.layer_norm.load_state_dict({'weight': pretrained_weights[0].weight,
                                         'bias': pretrained_weights[0].bias})

    def forward(self, tokens):
        # 1. Convert input sequence into token IDs using Byte-Piece Tokenizer (BPE)
        # Token Embeddings (batch_size, seq_len)
        token_embeddings = self.embedding_matrix[tokens]

        # Add positional embeddings 
        positional_embeddings = self.positional_embeddings[:tokens.shape[1]].unsqueeze(0)

        # Add embeddings
        x = token_embeddings + positional_embeddings
        x = self.dropout(x)

        # 2. Pass Embeddings Through Transformer Blocks
        for i in range(self.N):
            x = self.transformer_blocks[str(i)](x)
            
        # 3. Obtain Logits by Projecting the Output to the Vocab Size.
        logits = self.fc_out(self.layer_norm(x))

        return logits
    
  