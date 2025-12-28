\# Stanford CS336 Language Modelling from Scratch 



https://stanford-cs336.github.io/spring2025/

https://github.com/stanford-cs336

https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU\_MT\_



\# Lecture 1



\## Overview



\### Efficiency



accuracy = efficiency x resources



Efficiency is extremely important for training of LLMs (also for smaller scales)



Over the time from 2012 to 2019 there was a 44x algorithmic efficiency increase (Hernandez et al. 2020)



--> ALWAYS benchmark and profile changes 

--> Do experiments at small scale, predict hyperparameters/loss at large scale



\### History



Before 2010s (non-neural net applications)

&nbsp;	Measuring entropy of language (Shannon 1950)

&nbsp;	N-Gram models ( basic autocomplete, basic speech recognition, machine translation)



Early 2010s

&nbsp;	First neural models: seq-to-seq models for translation

&nbsp;	Adam optimizers

&nbsp;	Attention Mechanism (only for translation)

&nbsp;	Transformers (only for translations

&nbsp;	Mixture of Experts

&nbsp;	Model Parallelism



Late 2010s

&nbsp;	ELMo (LSTM that created word-level embeddings with context)

&nbsp;	BERT (Transformer model that created word/token-level embeddings)

&nbsp;	--> ELMo and BERT were both used in different ways, for semantic understanding, classifications, NER, etc.



Modern LLMs

&nbsp;	Various LLMs, Closed Models, Open weights models, and Open Source Models

&nbsp;	... whats happening today





\### Tokenization



...



\### Architecture



Variants:

&nbsp;	Activation functions: ReLU, SwiGLU 

&nbsp;	Positional encodings: sinusoidal, RoPE

&nbsp;	Normalization: LayerNorm, RMSNorm

&nbsp;	Placement of normalization: pre-norm versus post-norm

&nbsp;	MLP: dense, mixture of experts

&nbsp;	Attention: full, sliding window, linear

&nbsp;	Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA)

&nbsp;	State-space models: Hyena (not doing attention, but something else, look into this)



Training:

&nbsp;	Optimizer: AdamW, Muon, SOAP

&nbsp;	LR Schedule

&nbsp;	Batch Size (critical batch size)

&nbsp;	Regularization (dropout, weight decay)

&nbsp;	Hyperparameters (number of heads, hidden dimensions/embedding size, grid search)



\### Kernels



GPUs are very efficient in doing floating point operations, the trick is usually to minimize data movement from memory to GPU.

For example fusion will help here



\### Parallelism



Data movement between GPUs is even slower, so here also we want to mminimize the movement from memory/gpu to gpu



\### Inference



Inference efficiency is important as well

-> Inference generates one token at a time (not utilizing the full GPU)

-> speculative decoding

-> KV caching, batching



\### Scaling



Chinchilla scaling law for TRAINING (not relevant for inference costs, etc):

TL;DR: Tokens needed = Model Parameters \* 20 (e.g., 1.4B parameter model should be trained on 28B tokens)



\### Data



Core question: What capabilities do we want the model to have? --> Choose data accordingly



&nbsp;	Perplexity: textbook evaluation for language models

&nbsp;	Standardized testing (e.g., MMLU, HellaSwag, GSM8K)

&nbsp;	Instruction following (e.g., AlpacaEval, IFEval, WildBench)

&nbsp;	Scaling test-time compute: chain-of-thought, ensembling

&nbsp;	LM-as-a-judge: evaluate generative tasks

&nbsp;	Full system: RAG, agents



Data Curation:

--> Just random internet data is pretty terrible, you need to curate it which takes a lot of work

--> Legally, do we have to license the data? Currently frontier models have to buy a lot of data

--> Formats of scraped data HTML, PDF, etc will have to be parsed into nice text data

--> Filtering and Deduplication (e.g. MinHash) is also a big one



\### Alignment



Base model is only next token prediction

--> Get langugage model to follow instructions

--> Tune the style (format, length, tone, etc)

--> Safety tuning



Phases:

&nbsp;	SFT phase

&nbsp;	Feedback learning (RLHF, formal verifiers for math or code, LLM as a judge, etc)

&nbsp;		PPO (old, not used as much)

&nbsp;		DPO

&nbsp;		GRPO (new one from deepseek)



After SFT phase, we should already have a somewhat good model

## Tokenization



Compression ratio of tokenizer: how many bytes are represented by a token? --> len(bytes(string) / len(tokens)  -> GPT4o has 1.7 compression ratio.



\### Character based tokenization



Each byte gets one token --> Vocabulary of 150k unicode characters

Problem 1: Vocabulary is too large

Problem 2: Many characters are super rare (like some emojis) which is a inefficient use of model parameters

Compression rate is = 1.5 (1 character is roughly 1.5 bytes on average)



\### Byte based tokenization



We don't have the sparsity problem as with Problem 2 in character based tokenization

Problem: We have super long sequences. Attention behaves quadratically in sequence length --> inefficient

Compression rate = 1



\### Word based tokenization



Early NLP approach.

Problem: Vocabulary size is (almost) unbounded, we might actually see new tokens appear that we never saw before.



\### Byte pair encoding (BPE)



GPT-2 uses BPE

Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.

Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.







&nbsp;	





