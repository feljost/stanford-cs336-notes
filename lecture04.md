# Lecture 4: Mixture of Experts

## What are MoEs?

A MoE model has multiple MLP/FFN "blocks" that are used sparsely (=experts). The Attention blocks remain the same. An additional router is trained that will route the tokens accordingly to the sparse MLP/FFN blocks.

The expert's results are then added together (sometimes a weighted average) and normalized.

We have more parameters without affecting the FLOPs. There have been many papers showing that with the same amount of FLOPs, more params is better --> Same amount of compute, but better result. This is also shown in 1.3B parameter models that MoE is better than dense.

These experts are not free however, we need the router and more memory is needed.

MoE training also allows good GPU parallelism, where different Experts are trained on different devices.

On a more practical note, it only really makes sense to make MoE Models if your model is big enough that you have to split it up regardless. The training objectives are heuristic (how to train the experts) and still quite unstable.

Experts are not trained on separate data and are not interpretable as "math expert" or "poetry expert", they are just artifacts of the training.

## Architecture

### Routing Function

**Token choice topk**: Each token looks at all available experts and picks the top k experts with the highest scores for itself. Standard implementation and widely used.
**Expert choice topk**: Each expert looks at the batch of incoming tokens and picks the top k tokens it wants to process.
**Global routing via optimization**: The system looks at the entire batch of scores globally. It tries to find an assignment that maximizes the total affinity score while satisfying constraints (e.g., "Every token must be assigned" AND "No expert can be overloaded")

k is the number of experts active, and generally it should be >= 2 (having only 1 expert used at once is not as good). Mixtral has 2, Grok has 2, Qwen has 4 and Deepseek 7.

Actually there are papers showing we do not even need smart routers at all, you can just use some mapping function (without semantic knowledge) and it still gives you gains. 

### Number of Experts and Expert Size

Conventionally we had Top2 routing with few experts, then more fine grained experts became more typical and is currently what everyone does. Most recently, deepseek has fine grained experts + shared experts (experts that are always on). Usually something like >=64 routed experts and >= 2 shared experts is what is done.

--> More experts + shared experts are the way to go
--> More experts also generally mean more _activated_ experts.

### Training Objectives for Routers

If you just learn a basic routing weight matrix, generally only 1 or 2 experts will take over. This is not very good / efficient.

We need sparsity for training time efficiency. But sparse gating decisions are not differentiable.

Solutions:

#### RL for optimizing gating policies

Does not work much better than other approaches and is not that commonly used therefore at scale.

#### Stochastic Perturbations

Choose topk of a learned routing Weight matrix but add noise to give different agents some "unexpected" token. More robust, less specialized experts.

#### Heuristic 'balancing' losses

This is what mostly everyone uses. Also uses topk of a learned routing weight matrix but it balances out the tokens evenly to experts. This can happen as per-expert balancing and/or per-device balancing.

### Training MoE

#### Parallelism

MoE can really be parallelized nicely; you can do data parallelism, model parallelism, and expert parallelism. These can all be combined.

Additionally, you might have multiple experts on any given device, and modern sparse matrix multiply engines can train multiple experts at the same time with the same multiplications (Megablocks is a lib for this).

Side Issue: Sometimes the router will try to send too many tokens to one expert, and this expert (and its respective device) has a limit (memory wise), which will lead to some tokens being dropped entirely. This can be one reason for different outputs with temperature 0 of online LLM providers, as it can be that the device of the expert is currently overused.

#### Stability

Stability is always hard, we have to always look at softmaxes for stability issues. This is even worse for MoE models. Z-Loss in routing functions will help with this (it's semi-mandatory).

#### Finetuning

Overfitting can be quite problematic with smaller fine tuning data. Deepseek's solution is to have a very big SFT dataset.

#### Upcycling of Dense Models into MoE

Just take a dense model, make copies of the MLP block, then use that as a starting point for all experts. First Qwen MoE did this.

#### DeepSeek implementations

v1: standard, topk with shared and finegrained experts
v2: Same architecture & same topk selector + Top-M device routing (do not send tokens to too many devices)
v3: change of routing gate to also have sigmoid normalization + Multi-Head Latent Attention (MLA). 

MLA is projecting the QKV into lower dimensions to make KV cached smaller. However this is not compatible with RoPE.