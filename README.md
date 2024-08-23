# FlashAttention (Metal Port)

This repository ports the official implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) to Apple silicon. It is a minimal, maintainable set of source files that reproduces the FlashAttention algorithm.

## Documentation

Single-headed attention only, to focus on the core bottlenecks of different attention algorithms (register pressure, parallelism). With the basic algorithm done correctly, it should be comparatively trivial to add customizations like block sparsity.

The backward pass uses less memory than [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). The official implementation allocates scratch space for atomics and partial sums. Apple hardware lacks native FP32 atomics (`metal::atomic<float>` is emulated). While attempting to circumvent the lack of hardware support, bandwidth and parallelization bottlenecks in the FlashAttention-2 backward kernel were revealed. An alternative backward pass was designed with higher compute cost (7 GEMMs instead of 5 GEMMs). It achieves 100% parallelization efficiency across both the row and column dimensions of the attention matrix. Most importantly, it is easier to code and maintain.

Everything is JIT compiled at runtime. This constrasts with the previous implementation, which relied on an executable embedded in Xcode 14.2.

## TODO List

Document how to use (macOS, iOS testing).

Document the need for `-Xswiftc -Onone`, how to force it to be done on iOS. Cannot use release mode because that prevents incremental compilation, and now recompilation latency is a bottleneck.
