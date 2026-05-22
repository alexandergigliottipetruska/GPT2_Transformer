# GPT-2

**GPT-2** is a *Decoder*-only Transformer architecture and generative model that autoregressively computes output tokens via causal masking and learns language modeling via Unsupervised Learning. It competes with supervised learning approaches while aiming to achieve zero-shot transfer on common NLP tasks.

## Transformer Architecture

The **Transformer** is a purely attention based architecture developed in the paper *Attention is All You Need* for language modeling and machine translation that avoids the sequential computation problem in RNNs which prevented parallelization.
It uses an Encoder to convert a sequence of symbol representations to continuous representations, with the output at each layer being fed to the Decoder to autoregressively predict output sequences. 

## Decoder and Causal Masking 

The Decoder consists of a Masked Multi Head Self-Attention layer and a Multi-Layer Perceptron, joined by Layer Normalization and Skip Connections.

The self-attention layers use causal masking to attend to earlier positions in the output sequence, masking future positions by setting them to $-\infty$ before the softmax layer. 

## Masked Multi Head Self-Attention

The **Attention** mechanism models long-range dependencies between inputs and outputs by allowing each token in a sequence to weigh the importance of other tokens. This enables the model to capture contextual and semantic relationships across the sequence.

In particular, attention uses three learned representations: Queries Q, Keys K, and Values V, which are obtained by linearly projecting the encoder input. The similarity between queries and keys is used to compute attention weights, which determine how much influence each token in the sequence should have when producing the representation of a given token.

$$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V $$

Multi Head Self-Attention allows the model to jointly attend to more information by using multiple different linear projections of the key, query, and value vectors. This allows each head to 'specialize' and focus on capturing different relations across the sequence.

$$\mathrm{MultiHead}(Q, K, V)=\mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\ W^O$$

$$\mathrm{where}\text{ }\mathrm{head}_i=\mathrm{Attention}(Q W_i^Q,  K W_i^K,  V W_i^V)$$

Note: Following the notation of *Attention is All You Need*, the dimensions of the weight matrices are $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

The causal mask is applied before the softmax to prevent positions from attending to future tokens. 

## Layer Normalization
**Layer Normalization** is used to stabilize and accelerate training normalizing the outputs of each layer to stay within an acceptable range. The mean and variance are computed for the features, not per batch (as in Batch Normalization).
and used to normalize the input. Note that the $\epsilon$ is used to prevent division by 0.

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Afterwards, learnable parameters ($\gamma$ and $\beta$) are applied to each feature to shift and scale the normalized activations during training.

$$y_i = \gamma \hat{x}_i + \beta$$

## GeLU Activation

**Gaussian Error Linear Unit** (**GELU**) activation function aims to smoothly weigh inputs by how likely they are to be positive under a Gaussian distribution. It appears more stochastic and smooth compare to the hard threshold applied by ReLU, with large positive values almost fully passing through, large negative values mostly suppressed, and near zero values partially passing through. It is applied to the hidden layer of the MLP.

The smoothness allows both better gradient flow and prevents the 'Dead Neuron' issue associated with ReLU activation. It is defined as 

$$\mathrm{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution. In practice, an approximation is commonly used:

$$\mathrm{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)$$

## Autoregressive Language Modeling with Top K Random Sampling

Autoregressive Language Modeling predicts an output token using an input sequence, which is then appended to the input sequence and fed back into the model to compute the next token. This is done until the context length is reached or an <EOS> ('End of Sequence') token is reached. The predicted tokens are determined by taking the logits from the output layer (which projected them from $d_model$ to vocab size), applying a softmax function to get the probabilities for every word in the vocabulary, and then randomly sampling from the top K (usually 40) highest probability tokens. Top K random sampling ensures the model produces a different output given the same input, injecting randomness and preventing the same deterministic output from occuring using a Greedy strategy. 

## Positional Embeddings

## Byte-Pair Encoding (BPE) Tokenizer



## Top K Random Sampling

## Citations
