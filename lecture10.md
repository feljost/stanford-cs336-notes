# Lecture 10: Inference

Why inference matters: training is one time costs, inference happens many times. It is also needed for test-time compute (thinking) and training via RL.

On an efficiency note: Training can be parallelized, but inference _has to happen sequentially_.

## How to measure Inference

Time to first Token, TTFT (s): How long till any generation happens (matters for chatbot, interactive experience)
Latency (seconds/token): how fast tokens appear for user (matters for chatbot, interactive experience)
Throughput (tokens/seconds): useful for batch processing (matters for backends, infrastructure needs)

## Arithmetic Intensity for Training

Arithmetic intensity of a model is roughly B (batch size).
Accelerator intensity = FLOPs per S / memory_bandwith  <-- This is 295 for a H100

--> if arimethic intensity > accelerator intensity; we are compute limited (good)
--> if arimethic intensity < acclearator intensity; we are memory limited (bad)
--> for a H100, we want to process batch sizes of >= 295

**But** how does this work for inference? We are proceeding token by token, so our arithmetic intensity, B is gonna be 1... so this is quite an issue, we are severly memory limited.

## Arithmetic Intensity for Inference

Naive sampling: genetrate each token, feed history into the transformers. This is very intensive, and we are doing a lot of work, as the first couple of tokens stay the same.

Solution is KV cache in HBM:

Approach:

Input: _Never gonna give_  --> all KV is calculated 
Output: you

new Input: _Never gonna give_ you  --> only calculate for "you"
Output: up

--> At inference time, the prefill of the KV cache is easily compute limited. Token by token generation is slow, this is why the length of the input does not matter that much compared to token generation. TTFT is esentially entirely dependent on prefill.

MLP intensity is B (requires concurrent requests), attention intensity is always 1 (impossible to improve).

## Taking shortcuts for faster inference (lossy)

### Reducing KV Cache Size

**Grouped Query Attention GQA**: Reducing the KV cache by factor of N/K (and allowing larger batch size and/or smaller GPUs as memory needed decreases). Larger bazch size will decrease latency but increase throughput
**Multi-Head Latent Attention MLA**: This is just compressing the KV size by projecting them into smaller spaces. However this is not compatible with RoPE out of the box. It is in general better in latency and throughput.
**Cross-Layer Attention**: Use the same KV Projections accross layers. 
**Local Attntion**: Throw away stuff that is outside of the attention window: Sliding window attention, Dilated Sliding Window, Global + Sliding Window, etc. --> Much faster but hurts accuracy quite a bit. Interleaving global and local attention layers (mixed layers) kind of helps a bit.

## Alternatives to Transformers

Transformers were designed for training efficiency, not designed for heavy inference.

### State-Space Models

... didn't quite understand this part, look into it later...

Core idea: Replace attention with a fixed-size "memory" that updates as each token comes in. No KV cache needed.

- Training: Can be parallelized (unrolls into a convolution)
- Inference: O(1) memory per token, much faster than attention
- Key model: **Mamba** - makes the memory update input-dependent (selective)

Trade-off: Fixed memory size may lose info over very long contexts

--> It is pretty bad for language modelling, except under 1B model.

--> New variant with MiniMax-01 (BASED: linear attention + local attention) started to make serious STOTA models, so there is some hope for this to take over.


### Diffusion Models

Idea: generate each token in parallel, starting with random noise, then refine multiple steps
--> some new models show some hope: ultra fast tokens/second (but not beating benchmarks on performance at the moment)

### Quantization

Reducing precision of the numbers, means less memory, means lower latency and higher throuhput. This makes it however less accurate.

 
- **Quantization-aware trianing (QAT)**: Train with a quantization (means retraining)
- **Post-training quantization (PTQ)**: Take an existing model and try to quantize it without screwing things up too much.

Problem: outliers screw things up --> Solution: outliers are done in FP16 where the rest is in int8. Works well but it is a bit slower (but fits into memory)

**Activation-aware quantization (AAQ)**: select which weights to keep in precision based on activation.

### Model Pruning

Identify impprtant layers, heads, hiddem dimensions, then remove the unimportant layers, etc. Then distill the original model into the pruned model,

--> Results are pretty good according to Nvidia (not the most trustworthy company with benchmaxxing)

### Model Distillation

1. Define faster model architecture
2. Initialize weights using original model
3. Repair faster model using distillation

## Speculative Decoding (lossless)

Use a cheap draft model to generate some tokens. Then evaluate the tokens with the big model (this can be done in parallel since its pre-fill) and then check if they are good or if they have to be corrected. The output is the same as the big model but faster (acceptance rate of the cheaper model is paramount here).
