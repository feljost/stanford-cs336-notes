# Lecture 2: Pytorch & Resource Accounting

## Tensor Memory & Data Types

```python
x = torch.zeros(4, 8)  # @inspect x
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8  # number of elements in the tensor
assert x.element_size() == 4  # float32 is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```

float32 is generally the gold standard in computing / ML  --> but it requires a lot of memory!

Memory has 2 aspects: number of values and data type of each value

One matrix in the feedforward layer of GPT-3 is 2.3GB

float16 is half the element size, so it cuts down half the memory. 
However the range of numbers is not great, it can't represent very low or large numbers. For _training_ small models its probably ok, but for large models this will not work.

bfloat16 is an adjustment of float16, where the range of numbers is better but the accuracy of the numbers is worse (accuracy generally is less of an issue than range). This is an improvement for _training_ models, an it is what a lot of parameters in a model will use (but not all).

fp8 is a even cruder datatype (supported on H100 and later).


So we either have a lot of memory needed or we have instability due to limited ranges / accuracy of the numbers.

--> Solution: mixed precision training

Example: float32 for attention, but bfloat16 for more "simple" feed forward layers. Intuition: bfloat16 is going to be more transitory (change a lot) and the thing that we really want to accumulate over time (i.e. attention layers) will require float32

## Compute

GPUs are much faster in parallel matrix multiplication, but first we need to move the stuff into the GPU memory (`.to("cuda")`).

Pytorch tensors are pointers to allocated memory. The actual "numbers" will just be a long 1D array in memory. The pytorch tensor shows how the matrices are built, meaning it will point to the memory position of each matrix element in the long 1D array in memory.

--> We can have multiple tensors that have the same storage (in my fgpt example we have some shared layers).
--> it also means that some operations like slicing and assigning them to a new variable, it will not create a new _underlying tensor_, just a new pointer, etc.

Generally everything is going to be in a batch, for LLMs, this will be in the following order:
`x = (4, 8, 16, 32)`
B = 4 --> 4 batches
T = 8 --> sequence (e.g. context length)
16, 32 --> "actual matrix" (could be for example embedding dim and vocab size, or any other matrix size)
Pytorch will generally do matrix multiplications correctly if it's setup like this.

### Einops

Einops is a library that allows us to name the dimensions so we don't loose track. Jaxtyping is a type hint that will help, and there are checkers that can check if it's correctly specified.

Example:
```python
# Pytorch classic
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence 
# New (einops) way:
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

## FLOP

A FLOP is a floating point operation like x+y or x*y, etc. Hardware will tell you the number of FLOP/s (sometimes FLOPS, uppercase S) it can achieve.

GPT3 used 3e23 FLOPs. GPT4 used (speculation) 2e25 FLOPs. Models over 1e25 FLOPs for training must be reported to EU government.

FLOP/s depends on the Hardware (GPU) and what data type we use.

### FLOPs needed for common operations

For matrix multiplication (bulk of most work)

flops needed = 2 * All dimensions needed for the multiplication.

example
```python
B = 1024
D = 256
K = 64

x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w

# We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.
actual_num_flops = 2 * B * D * K
```

Element wise operation on a m x n matrix requires O(m*n) FLOPs.

Addition of two m x n matrices require m*n FLOPs.

### Heuristics for FLOPs

For a linear model, but somewhat scalable to Transformers
- Forward pass: 2 (# data points) (# parameters) FLOPs
- Backward pass: 4 (# data points) (# parameters) FLOPs (takes more operations to compute gradients)
- Total: 6 (# data points) (# parameters) FLOPs

(but we don't do optimizer.step(), this doesn't happen every forward and backward pass).

### Model FLOPs utilization (MFU)

Definition: (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]
Usually, MFU of >= 0.5 is quite good (and will be higher if matmuls dominate)



