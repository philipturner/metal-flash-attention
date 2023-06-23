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

## Performance: Overview

SGEMM, every square matrix from 1&ndash;1536:

![Max GFLOPS achieved](./Documentation/SGEMM.png)

HGEMM\*\*, every square matrix from 1&ndash;2048:

![Max GFLOPS achieved](./Documentation/HGEMM.png)

GFLOPS during general matrix multiplication:

| Large Square Sizes | 256 x 256 x 256 | 384 x 384 x 384 | 512 x 512 x 512 | 768 x 768 x 768 | 1024 x 1024 x 1024 | 1280 x 1280 x 1280 | 1440 x 1440 x 1440 |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Accelerate F64     |   333 |   622 |   616 |   696 |   442 |
| MFA F64\*          |&lt;590|&lt;590|&lt;590|&lt;590|&lt;590|&lt;590|&lt;590|
| Accelerate F32     |  1223 |  1303 |  2282 |  2679 |  2262 |
| MPS F32            |  1847 |  3216 |  6200 |  6157 |  8153 |  7771 |  6497 |
| MFA F32            |  2760 |  5202 |  7007 |  7739 |  8185 |  8111 |  8472 |
| MPS F16            |  2399 |  4066 |  5849 |  5680 |  7336 |  7102 |  6433 |
| MFA F16            |  3185 |  5945 |  7644 |  8734 |  9353 |  9109 |  9215 |

| Large Square Sizes | 2048 x 2048 x 2048 | 3072 x 3072 x 3072 | 4096 x 4096 x 4096 | 6144 x 6144 x 6144 |
| ------------------ | ----- | ----- | ----- | ----- |
| Accelerate F64     |   536 |   516 |   520 |   504 |
| MFA F64\*          |&lt;590|&lt;590|&lt;590|&lt;590|
| Accelerate F32     |  1978 |  2058 |  1957 |  1947 |
| MPS F32            |  8472 |  8482 |  8270 | Error |
| MFA F32            |  8992 |  9236 |  9247 |  9257 |
| MPS F16            |  7729 |  7824 |  7771 | Error |
| MFA F16            |  9618 |  9788 |  9778 |  9905 |
