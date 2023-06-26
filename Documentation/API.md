# Metal FlashAttention API

Metal FlashAttention is a research project, proving the feasibility of a first-principles rewrite of AI software. However, it is also an end product used in deployment scenarios. The API defines how to integrate the library into other applications.

## Programming Language

The API is in Metal Shading Language, a C++ dialect. The ABI is in Apple Intermediate Representation, an LLVM IR dialect

## Conventions

All API are in lower case, using underscores instead of camel case. This follows the naming convention of the Metal Standard Library.

Function signatures:
- Anything that replicates BLAS/LAPACK behavior: 
  - name exactly replicates the BLAS/LAPACK name
  - input types fused with the name (HGEMM, SGEMM)
  - output types fused with the name (HGEMM, SGEMM)
- Quantized variants of BLAS/LAPACK functions:
  - name is Q + the function name (QGEMM)
  - input types specified through function constants
  - output types specified through function constants
- Anything else
  - name is a one-word description of what the function does
  - input types specified through function constants
  - output types specified through function constants
  
Function variants:
  - dimensionality of convolutions specified through numerical function constants
  - different convolution windows (e.g. Conv1x1 vs Conv3x3) specified through numerical function constants
  - unsupported kernel variants simply do nothing at runtime
  
## Best Practices

List:
- Batch 8 or more compute commands into the same `MTLComputeCommandEncoder`.
  - Precludes use of MFA with eager execution engines.
  - Order of magnitude speedup over MPS for elementwise kernels.
  - Use this technique for mixed precision, as HGEMM and SGEMM don't support fused type casting yet.
- Extract all matrix shapes from your app's compute graph ahead-of-time.
  - Instantiate `MTLFunction` variants asynchronously or with explicit multithreading.
  - Order of magnitude speedup over MPSGraph, where multithreading makes initialization slower.

Examples:

```
// Mixed precision GEMM
// A (FP16)
// B (FP32)
// C (FP16)
MTLCommandQueue {
  MTLCommandBuffer {
    MTLComputeCommandEncoder {
      // custom elementwise shader converts A: FP16 -> FP32
      // `sgemm`
      // custom elementwise shader converts C: FP32 -> FP16
    }
  }
}

// In-place Winograd with 2.25x speedup
// data: FP16
// weights: 6-bit palletized FP16
// output: BF16
MTLCommandQueue {
  MTLCommandBuffer {
    MTLComputeCommandEncoder {
      // `convolution` on data: FP16 -> FP16 Winograd 4x4
      // `convolution` on weights: 6-bit palletized -> FP16 Winograd 4x4
      // custom elementwise shader converts data: FP16 -> FP32
      // custom elementwise shader converts weights: FP16 -> FP32
      // `sgemm` with `batched = true`
      // `convolution` on weights + data: FP32 Winograd 4x4 -> FP32
      // custom elementwise shader converts data: FP32 -> BF16
    }
  }
}
```
