# FlashAttention (Metal Port)

> WARNING: The code is not finished yet. It is currently a "minimum viable product". Meaning, a complete reproduction of the Flash2 paper with reasonable performance, massive memory savings, and no bugs. Someone skilled in the art would attain meaningful insights from examining the code. The outstanding performance issues will be resolved in July&ndash;August 2024.

This repository ports the official implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) to Apple silicon. It is a minimal, maintainable set of source files that reproduces the FlashAttention algorithm.

The source tree contains a customized version of the [unified GEMM kernel](https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131), a self-contained script for reaching peak performance in matrix multiplications. The GEMM kernel is distinct from the FlashAttention kernel. The modified GEMM kernel serves a few purposes, such as testing naive attention algorithms. Code related specifically to GEMM, and its maintenance, is out of scope for `metal-flash-attention`.

## Important Information

Supports macOS and iOS. Can be compiled from within a Swift Playground on iPad.

Everything is JIT compiled at runtime. This constrasts with the previous implementation, which relied on an executable embedded in Xcode 14.2.

Everything is computed and stored in full 32-bit precision. Except for the temporary attention matrix (for algorithms that materialize it).

Async copies are used extensively, mostly to simplify the code design. Even on M3, where it harms performance.

Single-headed attention only, to focus on the core bottlenecks of different attention algorithms (arithmetic intensity, parallelism).

## Modifications to FlashAttention

TODO: Rewrite all of this.

Second, the backward pass uses less memory. The official implementation allocates scratch space for atomics and partial sums. Apple hardware lacks native FP32 atomics (`metal::atomic<float>` is emulated). While attempting to circumvent the lack of hardware support, bandwidth and parallelization bottlenecks in the FlashAttention-2 backward kernel were revealed. An alternative backward pass was designed with higher compute cost (7 GEMMs instead of 5 GEMMs). It achieves 100% parallelization efficiency across both the row and column dimensions of the attention matrix. Most importantly, it is easier to code and maintain.

### Alternatives Considered for Backward Pass

One interesting insight, is the discovery of an optimal backward kernel. The attention matrix (`dS`) should be materialized explicitly in RAM, provided the sequence length is small enough. Each inference in the batch, or head (often 8 total) can be computed sequentially. Only one $O(n^2)$ attention matrix need be held in memory at any one time. This attention matrix will consume roughly as much memory as the partial sums for `dQ` accumulation in Flash2 backward.

The minimal compute cost, maximally parallel backward pass is:
- Accumulate `dV` on-chip while writing `dS` to memory
  - Register pressure may preclude fused accumulation of `dK`
- In a second pass, use GEMM to generate `dK` and `dQ`
  - Each GEMM traverses the attention matrix along a different direction, in which it requires no synchronization between processing units

Provided the head size (`D`) is 32 or greater\*, this should use less HBM bandwidth than the kernel where `dQ` is accumulated atomically. It requires the compute cost of 5 GEMMs, just like Flash2. This variant could be limited to $O($ processor count $)$ memory instead of $O(n^2)$ memory. Doing so, without a performance regression, requires knowledge of machine-specific parameters (GPU core count, cache size, etc.). Running the algorithm in production would require a lot of hardware-specific tuning.

> \*On Apple silicon, where the optimal block size is 32x32 due to register pressure constraints.

Preliminary data supports the prediction that explicitly materializing an $O(n^2)$ matrix improves performance. Data quality was limited by register pressure bottlenecks and performance issues with compressing the attention matrix as BF16.

## TODO List

Documentation:
- Explain how the rooflines are calculated.
- Publish the performance data.
- Provide example code for encoding attention kernels.
- Explain the reason for removing the sparse blockmask, noting its recent addition to PyTorch.
  - Explain how someone would implement the sparse blockmask: two-level hierarchy of sparsity. Attention mask decomposed into block-aligned, zero-padded chunks, with one level of indirection to access via pointers.
  - PyTorch's initial implementation of FlexAttention restricted problem sizes to those that divide by 128. MFA was designed with first-class support for odd and strange block sizes. It is okay if the block size does not divide evenly into the sequence length.
  - Explain that 128x128 + 1024x1024 is not a viable pair of block sizes. The small block size should equal the native GEMM block size (e.g. 32x64, 16x96). The native GEMM block all divide evenly into 960x960, which has more prime factors.
  - 960 is divisible by 64.
  - 960 is not divisible by 128. Instead, replace 128 with 120 in all block size selection heuristics. 120 still divides by 8 (the SIMD-group matrix square side length).
  - Repo owner encourages people to try modifying this code, to defeat the quadratic scaling compute cost of attention. Due to time constraints, the author could not try this himself.

Portability:
- Support mixed precision.
- Optimize performance on M3.
- Test problems where the attention matrix is not a square.
- Support some critical features:
  - Dense mask that safely handles unaligned matrix edges.
  - Leading dimension to handle multi-head attention.
