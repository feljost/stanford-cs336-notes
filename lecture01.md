# Lecture 1: Overview and Tokenization

## Overview

### Efficiency

accuracy = efficiency x resources

Efficiency is extremely important for training of LLMs (also for smaller scales)

Over the time from 2012 to 2019 there was a 44x algorithmic efficiency increase (Hernandez et al. 2020)

--> ALWAYS benchmark and profile changes
--> Do experiments at small scale, predict hyperparameters/loss at large scale

### History

Before 2010s (non-neural net applications)
- Measuring entropy of language (Shannon 1950)
- N-Gram models ( basic autocomplete, basic speech recognition, machine translation)

Early 2010s
- First neural models: seq-to-seq models for translation
- Adam optimizers
- Attention Mechanism (only for translation)
- Transformers (only for translations
- Mixture of Experts
- Model Parallelism

Late 2010s
- ELMo (LSTM that created word-level embeddings with context)
- BERT (Transformer model that created word/token-level embeddings)
- --> ELMo and BERT were both used in different ways, for semantic understanding, classifications, NER, etc.

Modern LLMs
- Various LLMs, Closed Models, Open weights models, and Open Source Models
- ... whats happening today


### Tokenization

...

### Architecture

Variants:
- Activation functions: ReLU, SwiGLU
- Positional encodings: sinusoidal, RoPE
- Normalization: LayerNorm, RMSNorm
- Placement of normalization: pre-norm versus post-norm
- MLP: dense, mixture of experts
- Attention: full, sliding window, linear
- Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA)
- State-space models: Hyena (not doing attention, but something else, look into this)

Training:
- Optimizer: AdamW, Muon, SOAP
- LR Schedule
- Batch Size (critical batch size)
- Regularization (dropout, weight decay)
- Hyperparameters (number of heads, hidden dimensions/embedding size, grid search)

### Kernels

GPUs are very efficient in doing floating point operations, the trick is usually to minimize data movement from memory to GPU.
For example fusion will help here

### Parallelism

Data movement between GPUs is even slower, so here also we want to mminimize the movement from memory/gpu to gpu

### Inference

Inference efficiency is important as well
-> Inference generates one token at a time (not utilizing the full GPU)
-> speculative decoding
-> KV caching, batching

### Scaling

Chinchilla scaling law for TRAINING (not relevant for inference costs, etc):
TL;DR: Tokens needed = Model Parameters * 20 (e.g., 1.4B parameter model should be trained on 28B tokens)

### Data

Core question: What capabilities do we want the model to have? --> Choose data accordingly

- Perplexity: textbook evaluation for language models
- Standardized testing (e.g., MMLU, HellaSwag, GSM8K)
- Instruction following (e.g., AlpacaEval, IFEval, WildBench)
- Scaling test-time compute: chain-of-thought, ensembling
- LM-as-a-judge: evaluate generative tasks
- Full system: RAG, agents

Data Curation:
--> Just random internet data is pretty terrible, you need to curate it which takes a lot of work
--> Legally, do we have to license the data? Currently frontier models have to buy a lot of data
--> Formats of scraped data HTML, PDF, etc will have to be parsed into nice text data
--> Filtering and Deduplication (e.g. MinHash) is also a big one

### Alignment

Base model is only next token prediction
--> Get langugage model to follow instructions
--> Tune the style (format, length, tone, etc)
--> Safety tuning

Phases:
- SFT phase
- Feedback learning (RLHF, formal verifiers for math or code, LLM as a judge, etc)
- 	PPO (old, not used as much)
- 	DPO
- 	GRPO (new one from deepseek)

After SFT phase, we should already have a somewhat good model
Tokenization

Compression ratio of tokenizer: how many bytes are represented by a token? --> len(bytes(string) / len(tokens)  -> GPT4o has 1.7 compression ratio.

### Character based tokenization

Each byte gets one token --> Vocabulary of 150k unicode characters
Problem 1: Vocabulary is too large
Problem 2: Many characters are super rare (like some emojis) which is a inefficient use of model parameters
Compression rate is = 1.5 (1 character is roughly 1.5 bytes on average)

### Byte based tokenization

We don't have the sparsity problem as with Problem 2 in character based tokenization
Problem: We have super long sequences. Attention behaves quadratically in sequence length --> inefficient
Compression rate = 1

### Word based tokenization

Early NLP approach.
Problem: Vocabulary size is (almost) unbounded, we might actually see new tokens appear that we never saw before.

### Byte pair encoding (BPE)

GPT-2 uses BPE
Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.
Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.



 