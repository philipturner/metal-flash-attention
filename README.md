# FlashAttention (Metal Port)

This repository ports the official implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) to Apple silicon. It is a minimal, maintainable set of source files that reproduces the FlashAttention algorithm.

## Documentation (In Progress)

Everything is JIT compiled at runtime. This constrasts with the previous implementation, which relied on an executable embedded in Xcode 14.2.

Single-headed attention only, to focus on the core bottlenecks of different attention algorithms (arithmetic intensity, parallelism).

## Modifications to FlashAttention

The backward pass uses less memory. The official implementation allocates scratch space for atomics and partial sums. Apple hardware lacks native FP32 atomics (`metal::atomic<float>` is emulated). While attempting to circumvent the lack of hardware support, bandwidth and parallelization bottlenecks in the FlashAttention-2 backward kernel were revealed. An alternative backward pass was designed with higher compute cost (7 GEMMs instead of 5 GEMMs). It achieves 100% parallelization efficiency across both the row and column dimensions of the attention matrix. Most importantly, it is easier to code and maintain.

## TODO List

Portability:
- Test problems where the attention matrix is not a square.
- Leading dimension to handle multi-head attention.
- Adversarial shape test to handle the above cases.
- Extend and refactor `Network.swift` to handle such problems before implementing on GPU.
- Refactor into a Swift package with a separate module for tests?
  - The benchmark needs to be documented somewhere, so somebody can copy the raw code into a new Xcodeproj for iOS.
  - Minimal examples of the benchmarks that don't actually measure problems large enough to get performance statistics?

Document the need for `-Xswiftc -Onone`, how to force it to be done on iOS.
