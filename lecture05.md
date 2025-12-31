# Lecture 5: CUDA and GPUs

CPUs have alarge amount of cache and control unit, very few threads/compute units/ALUs. It's all about single (or near single) thread performance. --> Optimize for latency. I want to finnish my task as soon as possible.

GPUs have basically no cache and minimal control units, but many many many threads/compute units/ALUs. It's all about parallellism. --> Optimize for throughput --> I want all my tasks i have finish as fast as possible in aggregate.

## Memory

This is currently more important than FLOPs, it is most commonly the bottleneck. FLOPs have scaled much much faster than memory. It is hard to keep the the actual compute units fed with data.

The closer (physically) the memory is to each SM (streaming multiprocessor), the faster it will be. L1 cache is very close as it lives inside the actual SM's. L2 cache is right beside it (still on GPU chip), is still fast, but around 10x slower. DRAM outside of the chip is even slower.

## Execution model:

**Blocks**: Groups of threads, each block runs on a SM with its own shared memory.
**Threads**: Within the blocks, there are threads, that do the work in parallel. All threads execute the same instructions but with different inputs.
**Warp**: Threads always execute in a warp of 32 consecutively numbered threads.

Threads < Warps (= 32 threads) < Blocks (= Multiple Warps)

## Memory Model

**Registers**: Local Memory, Shared Memory, Global Memory.

Each computation that acts across blocks needs to go to global memory. Ideally threads only work on local and/or shared memory.

On a A100:
    Compute: streaming multiprocessors (SMs) [A100: 108]
    Memory:
    
DRAM [A100: 80GB] - big, slow
L2 cache [A100: 40MB]
L1 cache [A100: 192KB per SM] - small, fast


## How to make GPUs go brrrrr...

1. Control divergence
    - Every thread has to do the same instructions. Conditionals (if else, etc.) will be quite a slowdown.
2. Low precision computation
    - If you have fewer bits, you have fewer bits to move. Do this all the time where you can get away with it.
    - There are operations that need higher precision (softmax, normalization, etc.), and there are operations that do not (matmuls, relu, etc.)
3. Operation Fusions
    - Naive approach: one computation, one result saved in memory. repeat x times. 
    - Fused kernel: Link many compute operations together, without saving it to memory, then only at the end ship it back to memory. (Recomputation)
4. Memory coalescing and DRAM
    - DRAM is in burst mode --> each read gives you many bytes.
    - _Coalesced memory_: all threads in a warp fall within the same burst.
    - Coalescing for matrix multiplication: if you read the memory in the correct way, you will have faster matmuls
5. Tiling
    - For Matrix multiplications, cut up the input matrices into sub-matrices (tiles). Then read the tiles from memory, do the partial computation using the tiles in shared memory, and then continue with the next tiles.
    - Advantage: repeated reads now access shared, not global memory, and memory access can be coalesced
    - Tile size is complex and will be affectzed by coalesced memory access, shared memory size, divisibility of the matrix dim. Sometimes memory coalescence with tiling mightr not be possible due to dimensions of the matrixes (here we have to do padding).

## Flash Attention

Uses tiling and recomputation smartly it makes a fused kernel that does all attention operations can be done without inbetween global memory reads.

It also uses online softmax, which unlike the normal softmax, does not need all inputs at once, but can compute it tile by tile. You never have to materialize the full matrix at once to compute the softmax.

## TPUs

In short: TPUs are matrix-multiply machines, GPUs are general parallel machines.
