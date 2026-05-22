import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformer import GPT2Transformer

# Byte Pair Encoding Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Transformer()
input_text = ""
vocab = tokenizer.get_vocab()
id_to_token_vocab = {v: k for k, v in vocab.items()}

def inference_pipeline(model, tokenizer, input_text, vocab, k=40):
    # Set model to evaluation mode
    model.eval()

    # Generation loop
    curr_len = len(tokens)
    end_of_sequence = False
    context_len = 1024
    
    # Get token ids from input text
    tokens = tokenizer(input_text, return_tensors='pt')

    # Terminate upon reaching context len (1024) or an <EOS> token
    while (curr_len <= context_len or end_of_sequence is False): 
        # Pass into Model architecture, output (batch_size, seq_len, vocab_size)
        logits = model(tokens)[:, -1, :] # returns logits at last position

        # Softmax for Next Token Prediction, get ID. Use Top K Random Sampling
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = nn.functional.softmax(top_k_logits, dim=-1)

        # Use Multinomial Sampling for Discrete Probabilities
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token_id = top_k_indices.gather(1, next_token_idx)

        # End if <EOS> encountered
        if next_token_id.item() == tokenizer.eos_token.id:
            break
        
        # Concatenate token
        torch.concatenate([tokens, next_token_id], dim=1)
        curr_len += 1

    output = tokenizer.decode(tokens[0], skip_special_tokens=True) # strip batch dimension

    return output

inference_pipeline(model, tokenizer, input_text, vocab)