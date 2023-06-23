# Metal FlashAttention

A faster alternative to Metal Performance Shaders, a reference implementation of modern GPU algorithms, and a step toward defragmenting the AI ecosystem.

Algorithms:
- [ ] Attention
  - [ ] Dense FlashAttention (TBD% ALU @ 64 heads)
  - [ ] Triangular FlashAttention
- [ ] Convolution
  - [ ] ConvGEMM 1x1
  - [ ] ConvGEMM 3x3
  - [ ] Winograd w/ 2.25x speedup
  - [ ] Winograd w/ 4.0x speedup
- [x] GEMM
  - [x] FP16 (93.3% ALU)
  - [x] FP32 (87.2% ALU)
  - [ ] FP64 Emulation
  - [x] SIMD Futures
  - [x] [Stream-K](https://arxiv.org/abs/2301.03598)
- [ ] Normalization
  - [ ] Group Normalization

## Usage

Usage:
- Download Xcode 14.2 from the Apple [developer tools archive](https://developer.apple.com/download/all/?q=xcode)
- Run the Swift script to compile `libMetalFlashAttention.metallib`
- Read the [API specification](./Documentation/API.md)
- Generate Metal shader variants at runtime

Alternatively:
- Download the newest version of Xcode
- Fetch the Metal library from [GitHub releases](https://github.com/philipturner/metal-flash-attention/releases)
- Run the command-line tool from this repository that validates integrity of the Metal library
