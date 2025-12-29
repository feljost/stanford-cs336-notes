# Lecture 3: Architecture & Hyperparameters

Currently everyone uses ROPE, classical positional embeddings are done. 

## Normalization Layers

### Pre-norm vs Post-norm

Pre-norm is also by almost all big LLMs (original transformer paper implemented norm after Attention). Post-norm was much less stable than pre-norm and required a lot more warmup time to work properly.

Grok and Gemma2 use both _before_ and after the attention blocks and FFN/MLP blocks.

In every case, keep the norms out of the residual connections (the path that does not go through the transformer but bypasses it). This residual stream should be as direct as possible.

**Old Implementation**
``` 
        x_{i+1} (Output)
             ^
             |
      [ Layer Norm ]  <<<--- This is bad
             ^
             |
         (Addition) <-----------+
             ^                  |
             |                  |
      RESIDUAL PATH        [ FFN/MLP ]
      (Direct Path)             ^
             |                  |
             +------------------+
             ^
             |
      [ Layer Norm ]  <<<--- This is bad
             ^
             |
         (Addition) <-----------+
             ^                  |
             |                  |
      RESIDUAL PATH      [ Multi-Head ]
      (Direct Path)      [ Attention  ]
             |                  ^
             |                  |
             +------------------+
             ^
             |
          x_i (Input)
```

**Modern Implementation**
```
        x_{i+1} (Output)
             ^
             |
         (Addition) <-----------+
             ^                  |
             |             [ FFN/MLP ]
      RESIDUAL PATH             ^
      (Direct Path)             |
             |            [ Layer Norm ]  <<<---- Pre-norm
             |                  ^
             |                  |
             +------------------+
             ^
             |
         (Addition) <-----------+
             ^                  |
             |            [ Multi-Head ]
      RESIDUAL PATH       [ Attention  ]
      (Direct Path)             ^
             |                  |
             |            [ Layer Norm ]  <<<---- Pre-norm
             |                  ^
             |                  |
             +------------------+
             ^
             |
           x_i (Input)
```


### LayerNorm vs RMSNorm

The original transformer architecture (and GPT3/2/1) used LayerNorm: it will normalize the mean and variance. This means that we have a bias (learnable) that is added and mean (given) that is subtracted in the calculation. This gives use nice norms around the mean + bias.

The new RMSNorm simply drops the bias and mean. It is simpler and works just as well while saving FLOPs, parameters and memory.

--> FLOPs saved are not so much (as it is not a matrix multiplication we don't care too much about it)
--> Memory movement overhead (retrieving the parameters into the GPU) is however strongly reduced, which saves a bit of time (~3% in the study shown)
--> Final loss is minimally improved as well

### Dropping of Bias Terms

In a similar fashion, most modern transformers don't have bias terms at all. They are just dropped, and only trained on pure matrix multiply. For the same reasons as using RMSNorm.

## Activations

### ReLU

Most basic one, max(0,xW_1)W_2. Used by original Transformers

```
y
      ^
      |          /
      |         /  y = x
      |        /
      |       /
------+------+--------> x
      |   0
```


### GELU

Gaussian Error Linear Unit: Same but it has a slight dip below 0. GPT1,2,3 all use GELU

### Gated Linear Units *GLU (ReGLU, SwiGLU, GeGLU)

GLUs have a gating mask V (vector with entries between 0 and 1) that acts as a trainable switch for information. This means that the network can learn to "open the gate" for important information or "close the gate" for noise.

```
Output
        ^
        | (element-wise multiplication)
   +----*----+
   |         |
   |      [Sigmoid]  <-- The "Gate" (0 to 1)
   |         ^
[Linear]  [Linear]
   ^         ^
   |         |
   +---------+
        ^
      Input
```

The gate adds extra parameters, so usually the normal input dimensions are a bit smaller to ensure that the parameters of ReGLU and ReLU are the same. --> Memory wise the parameters stay the same, and FLOPs are only marginally increased (neglible).

SwiGLU is what is commonly used by new models post 2023.

## Serial vs Parallel layers

Normally Transformer blocks are serial, first we do attention, then MLP, then next attention, then next MLP ... .

There are some ways to fuse these operations together and can give a 15% performance in training increase, however it is not so commonly used in new models.

## Positional Embeddings

Sine embeddings: adds deterministic sine and cosine frequencies -> no learnable params. (Original transformer)
Absolute embeddings: adds a position vector ->  learnable params (GPT 1,2,3)
Relative embeddings: adds a vector to the attention computation -> learnable params (more complex)

### RoPE

What matters is relative embeddings, not absolute. Sine has some cross-terms and other issues with relative-ness, Absolute clearly does not have this.

We basically just rotate the embeddings of the word/token. Rotation is always done in only just 2 dimensions (it is not a classical full rotation), some are rotated quickly, some are rotated slowly. The embeddings (queries and keys) are therefore just multiplied with _fixed_ rotation matrix. (The rotation matrix can be adjusted by a factor theta but it is set at the start and not learned.) 

This adds no additional params for the model, but some minor extra compute (can be sped up with custom kernels of course).

## Hyperparameters

### Consensus on Dimensionality Feedforward Layers

d_model = dimensionality of x (i.e. the input)

The feedforward layer dimenstionality is generally:
	4          times the d_model for ReLU, GeLU, etc.
	8/3 = 2.6  times the d_model for SwiGLU, ReGLU, etc.

### Head dimension ratio

General consensus and should be used:

head_dim = model_dim / num_heads

model_dim -> embedding vector size
head_dim -> size of vector inside single attention head
num_heads -> number of attention heads per attention block

### Aspect ratios (Deep vs Wide Networks)

There is generally a sweet spot, but there are outliers. GPT2 was much wider than it was deep, but then for example GPT3/Mistral/Qwen are deeper. A somewhat wide range of aspect ratios achieve similar performance.

There are compute differences between the choices, especially with multi GPU setups. 

### Vocabulay Size

Generally trending upwards, obviously with more language support, more emoji support, more special chars needed, etc. Monolanguage usually have 30-50k vocab size, and multilingual 150-200k.

### Regularization

In pre-training, one would think we don't need regularizetion (we do  1 epoch, hard to overfit on so much data).

Dropout was generally used in the begining but it has dissapeared over time. 

Weight-decay is still used during pre-training. It does not change the val-train loss divergence, but there is some complex thing happening with the learning rate schedule, which gives us better training (and val) losses, this is important especially when the LR goes to 0.

## Stability Issues with softmax operation

Exploding / spiking gradients is often a problem with LLM training. Often can be due to softmax.

### Z Loss

Soft-max can lead to instability, this tries to satabilize it. Did not quite follow along here.

### QK Norm

Passing the Q and K through a LayerNorm / RMSNorm before going into softmax inside the attention block.

### Logit Soft-capping

Soft-cap the logits if they are over a maximum value before the final softmax calculated.



## Attention Heads

###  MHA (Multi-Head Attention) - The Standard

Structure: Every "Query" head has its own matching "Key" and "Value" head.
Ratio: 1 Query : 1 Key/Value.
Pros: Maximum expressivity.
Cons: Slowest; huge memory usage for KV cache during inference.

### MQA (Multi-Query Attention) - The Extreme

Structure: All "Query" heads share a single "Key" and "Value" head.
Ratio: Many Queries : 1 Key/Value.
Pros: Fastest inference, smallest memory footprint.
Cons: Can degrade model quality/accuracy because all heads are forced to attend to the same subspace.

### GQA (Grouped-Query Attention) - The Sweet Spot

Structure: The "Query" heads are divided into groups (e.g., 4 groups). Each group shares a single "Key" and "Value" head.
Ratio: Many Queries : Few Key/Values (e.g., 8 Queries : 1 KV).
Pros: Used in modern models (like LLaMA-2 and LLaMA-3). It offers near-MQA speed with near-MHA quality.

### Reducing Attention Head Cost

We want to keep arithmetic operations high (this is what GPUs are good at) and memory accesses low (slow process). Generally MHA (with KV-caching) need large batches and short seq length, or big model dimensions to be effective. 

GQA / MQA means we have shared K and V therefore much less memory access operations needed.

### Sparse Attention Mechanisms

There are various ways like sliding window attention that allows for longer context processing.