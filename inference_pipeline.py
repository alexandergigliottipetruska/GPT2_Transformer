import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformer import GPT2Transformer

# Byte Pair Encoding Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Transformer(config="XL", device = device).to(device)
input_text = "Hello, my name is Alexander and I "
vocab = tokenizer.get_vocab()
id_to_token_vocab = {v: k for k, v in vocab.items()}


def inference_pipeline(model, tokenizer, input_text, vocab, k=40):
    # Set model to evaluation mode
    model.eval()

    # Tokenize text
    tokens = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)

    curr_len = tokens.shape[1]
    context_len = 100 # 1024
    end_of_sequence = False
    k = 40
    generated_tokens = None

    while (curr_len < context_len and end_of_sequence is False):
        # Pass into Model architecture, output (batch_size, seq_len, vocab_size)
        logits = model(tokens)[:, -1, :] # return logits at last position

        # Softmax for Next Token Prediction, get ID. Use Top K Random Sampling
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = nn.functional.softmax(top_k_logits, dim=-1)

        # Multinomial Distribution for Discrete Probabilities
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token_id = top_k_indices.gather(1, next_token_idx)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Concatenate Token to Sequence
        tokens = torch.concat([tokens, next_token_id], dim=1)
        curr_len += 1

        if generated_tokens is None:
          generated_tokens = next_token_id
        else:
          generated_tokens = torch.concat([generated_tokens, next_token_id], dim=1)

    output_text = tokenizer.decode(tokens[0], skip_special_tokens=True) # strip batch dimension
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return input_text, generated_text, output_text


prompt, generated_text, output_text = inference_pipeline(model, tokenizer, input_text, vocab)

  