// CURRENT STATE OF THE GIST AT JUN 4, 2024, 7:00 PM EDT
// CHECK THE SOURCE FOR ANY COMMITS SINCE THEN:
// https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131

//
//  main.swift
//  UnifiedGEMMKernel
//
//  Created by Philip Turner on 5/29/24.
//

import Metal
#if os(macOS)
import IOKit
#endif

// Single shader source that supports every hardware architecture, problem
// size, and precision.
//
// ========================================================================== //
// Introduction
// ========================================================================== //
//
// There should not be a tradeoff between:
// - Reaching theoretical maximum ALU performance (within a factor of 1.01)
// - Reaching theoretical minimum compilation latency (provided you have an
//   engine for caching shader variants)
// - Having short, legible, and portable source code
// - Supporting every GPU generation and data type
//
// The existing solutions out there compromise on one of these. For example,
// MPS and Mojo are closed source, meaning the code is not legible. Other ML
// frameworks are ergonomic and easy to use, but sacrifice performance.
// Performance is most often neglected on older chips (M1, M2) or edge cases
// (when matrix dimensions are not divisible by block size).
//
// Recently, I found a way to access SIMD async copy instructions from the JIT
// compiler. This removes the need to use Xcode 14.2 in the shader compilation
// process. Instead of encoding all arguments in function constants, some of
// them can be encoded directly into a JIT-compiled shader source. This
// freedom allows for simpler shader code and a simpler interface for running
// the kernel. For example, the client no longer needs to allocate threadgroup
// memory at runtime.
//
// The code should be robust against worst-case situations. Imagine a PyTorch
// workflow where someone explores a hyperparameter space. They try changing
// the size of an MLP from 128 neurons, to 129 neurons, 130, etc. Each call
// into a matrix multiplication has a different problem size. If a new kernel
// was cached/compiled from scratch for each call, the latency would become a
// bottleneck. Yet, knowing the problem size beforehand is critical to reaching
// maximum performance (and outperforming MPS). The code should meet the
// standards of something that replaces MPS across millions of end-user
// workflows.
//
// The first attempt will not be perfect. I will learn things, try again with
// new implementations written from scratch. Continue distilling, simplifying,
// making the code more robust. Compare against alternatives (MPS, MLX) with a
// data-driven, evidence-driven approach.
//
// ========================================================================== //
// Methods
// ========================================================================== //
//
// The following design specifications were drafted for the source file.
//
// Anything with a fixed number of options that would require adding control
// flow blocks to the shader.
// - Device architecture
// - Blocking setup, threadgroup memory allocation
// - Operand precisions, accumulator precision
// Convention: Injected into shader source.
//
// Remaining parts that must be known ahead of time for reasonable performance.
// - Remainder of the integer division: problem size / block size
// - Transpose state of operands
// Convention: Choice between injection into shader source, or function
//             constants.
//
// Parts where inlining a constant into assembly would maximize performance.
// - Actual problem size
// - Value of non-1.0 alpha, non-0.0 beta
// Convention: Choice between injection into shader source, function constants,
//             or runtime data structure.
//
// Draft a kernel that does all of these things. It doesn't need optimal
// performance; just needs to implement them correctly.
//
// ========================================================================== //
// Lab Notes
// ========================================================================== //
//
// First issue:
// - The optimized BF16 decoding function from a previous experiment does not
//   support transposed representations in memory.
// - It also didn't support loading/storing from threadgroup memory.
// - The previous experiment didn't implement all the components of the BF16 ->
//   FP32 decoding optimization for M1.
//
// Second issue:
// - simdgroup_matrix_storage.load/store doesn't support unaligned inputs.
// - Previous design delegated that to the SIMD async copy unit in hardware.
// - Likely need a better alternative on M3/M4, where async copy is emulated
//   and hence very slow.
//
// Solution:
// - Three code paths for loading/storing from memory, until I prove the Apple
//   GPU 'device_load' instruction natively supports alignments smaller than
//   the vector size (e.g. alignment is the scalar size).
//   - Transposed (2 instructions)
//   - Untransposed, not aligned to multiple of 2 (2 instructions)
//   - Untransposed, aligned to multiple of 2 (1 instruction)
// - Use code generation to spawn the header with compact Swift code.
//
// Third issue:
// - While trying to get rid of async copies, I kept finding edge cases. For
//   example, when the matrix dimension is an odd number, and RAM accesses go
//   out of bounds. Fixing these requires either incurring immense overhead at
//   matrix edges, or baking if/else statements into source code.
//
// I found something interesting. If I avoid async copies for most of the inner
// loop iterations, I can get decent performance on M4. This is a slightly
// modified version of the M1 kernel, which divides the "k" accumulator into
// two sections. The first section reads the inputs directly from device memory.
// The last section reads from threadgroup memory.
//
// This modified kernel frequently causes IOCommandBuffer errors on M1. I need
// to understand why that is happening.
//
// ========================================================================== //
// Tuning Performance on M4
// ========================================================================== //
//
// Avoiding regressions on M1 Max:
//
// The kernel must achieve 8100 GFLOPS @ 1535x1535, 48x48x24.
// The kernel must achieve 8150 GFLOPS @ 1536x1536, 48x48x24.
// The kernel must achieve 7530 GFLOPS @ 1537x1537, 48x48x24.
//
// Reference statistics for M1 Max:
//
// - problemSize = 256  |  913 GFLOPS (32x32x8)
// - problemSize = 384  | 2931 GFLOPS (32x32x8)
// - problemSize = 512  | 5342 GFLOPS (32x32x8)
// - problemSize = 640  | 5463 GFLOPS (32x32x8) 6440 GFLOPS (async copy)
// - problemSize = 768  | 6160 GFLOPS (48x48x8) 7017 GFLOPS (async copy)
// - problemSize = 896  | 6643 GFLOPS (48x48x8) 7136 GFLOPS (async copy)
// - problemSize = 1024 | 7596 GFLOPS (48x48x8) 6966 GFLOPS (async copy)
// - problemSize = 1152 | 7676 GFLOPS (48x48x8) 8144 GFLOPS (async copy)
// - problemSize = 1280 | 7712 GFLOPS (48x48x8) 7813 GFLOPS (async copy)
// - problemSize = 1408 | 7747 GFLOPS (48x48x8)
// - problemSize = 1536 | 8392 GFLOPS (48x48x8)
//
// Performance target on M4:
//
// - problemSize = 256  | 1195 GFLOPS (32x32x8)  590 GFLOPS (MPS)
// - problemSize = 384  | 1729 GFLOPS (32x32x8) 1105 GFLOPS (MPS)
// - problemSize = 512  | 2549 GFLOPS (32x32x8) 2051 GFLOPS (MPS)
// - problemSize = 640  | 2983 GFLOPS (32x32x8) 3028 GFLOPS (MPS)
// - problemSize = 768  | 3036 GFLOPS (32x32x8) 3087 GFLOPS (MPS)
// - problemSize = 896  | 3044 GFLOPS (32x32x8) 3086 GFLOPS (MPS)
// - problemSize = 1024 | 3074 GFLOPS (32x32x8) 3125 GFLOPS (MPS)
// - problemSize = 1152 | 3123 GFLOPS (32x32x8) 3152 GFLOPS (MPS)
// - problemSize = 1280 | 3134 GFLOPS (32x32x8) 3134 GFLOPS (MPS)
// - problemSize = 1408 | 3167 GFLOPS (32x32x8) 3150 GFLOPS (MPS)
// - problemSize = 1536 | 3174 GFLOPS (32x32x8) 3129 GFLOPS (MPS)
//
// Performance deterioration for odd problem sizes:
// M1 Max (32x32, async copy), M4 (32x32, no async copy)
//
// - problemSize = 254 | 1888 GFLOPS  950 GFLOPS
// - problemSize = 255 | 1950 GFLOPS  971 GFLOPS
// - problemSize = 256 | 2087 GFLOPS 1210 GFLOPS
// - problemSize = 257 | 1744 GFLOPS  907 GFLOPS
// - problemSize = 258 | 1754 GFLOPS  921 GFLOPS
//
// - problemSize = 510 | 5296 GFLOPS 2614 GFLOPS
// - problemSize = 511 | 5266 GFLOPS 2624 GFLOPS
// - problemSize = 512 | 5390 GFLOPS 2765 GFLOPS
// - problemSize = 513 | 5180 GFLOPS 2365 GFLOPS
// - problemSize = 514 | 5208 GFLOPS 2377 GFLOPS
//
// - problemSize = 1022 | 5989 GFLOPS 3054 GFLOPS
// - problemSize = 1023 | 5905 GFLOPS 3059 GFLOPS
// - problemSize = 1024 | 7164 GFLOPS 3051 GFLOPS
// - problemSize = 1025 | 5618 GFLOPS 2880 GFLOPS
// - problemSize = 1026 | 5770 GFLOPS 2905 GFLOPS
//
// Overall scaling for aligned problem sizes:
// M4 (32x32, no async copy) vs M4 (MPS)
//
// - problemSize = 256  | 1205 GFLOPS vs 590 GFLOPS (MPS)
// - problemSize = 384  | 1711 GFLOPS vs 1105 GFLOPS (MPS)
// - problemSize = 512  | 2747 GFLOPS vs 2051 GFLOPS (MPS)
// - problemSize = 640  | 2939 GFLOPS vs 3028 GFLOPS (MPS)
// - problemSize = 768  | 3010 GFLOPS vs 3087 GFLOPS (MPS)
// - problemSize = 896  | 3024 GFLOPS vs 3086 GFLOPS (MPS)
// - problemSize = 1024 | 3040 GFLOPS vs 3125 GFLOPS (MPS)
// - problemSize = 1152 | 3101 GFLOPS vs 3152 GFLOPS (MPS)
// - problemSize = 1280 | 3101 GFLOPS vs 3134 GFLOPS (MPS)
// - problemSize = 1408 | 3130 GFLOPS vs 3150 GFLOPS (MPS)
// - problemSize = 1536 | 3140 GFLOPS vs 3129 GFLOPS (MPS)
//
// The above results were without some potential optimizations. For example,
// fusing device_load instructions for consecutive matrix elements when the
// matrix dimension is not divisible by 2. In addition, eliding the async
// copies when writing the C matrix to memory on M4.
//
// ========================================================================== //
// Tuning Precisions
// ========================================================================== //
//
// ## M1 Max
//
// Configuration:
// - maximum of 32x32x32 and 48x48x24/32
// - inputs are not transposed
//
// memA | memB | memC | regA | regB | regC |  512 |  768 | 1024 | 1280 | 1536 |
// ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
// FP16 | FP32 | FP32 | FP16 | FP32 | FP32 | 6754 | 7274 | 7604 | 7891 | 8300 |
// FP16 | FP32 | FP32 | FP32 | FP32 | FP32 | 6794 | 7244 | 7578 | 7864 | 8299 |
// FP32 | FP16 | FP32 | FP32 | FP16 | FP32 | 6651 | 7224 | 7630 | 7920 | 8322 |
// FP32 | FP16 | FP32 | FP32 | FP32 | FP32 | 6560 | 7231 | 7632 | 7924 | 8318 |
// BF16 | FP32 | FP32 | BF16 | FP32 | FP32 | 6118 | 6624 | 6912 | 7381 | 7590 |
// BF16 | FP32 | FP32 | FP32 | FP32 | FP32 | 6398 | 7159 | 7031 | 7632 | 8232 |
// FP32 | BF16 | FP32 | FP32 | BF16 | FP32 | 5556 | 6779 | 6529 | 7297 | 7680 |
// FP32 | BF16 | FP32 | FP32 | FP32 | FP32 | 5223 | 7149 | 6898 | 7753 | 8112 |
//
// FP16 | FP16 | FP32 | FP16 | FP16 | FP32 | 6796 | 7879 | 8331 | 8396 | 8492 |
// FP16 | FP16 | FP32 | FP32 | FP32 | FP32 | 6727 | 7848 | 8306 | 8396 | 8490 |
// FP16 | BF16 | FP32 | FP16 | BF16 | FP32 | 6156 | 6935 | 7221 | 7535 | 7841 |
// FP16 | BF16 | FP32 | FP16 | FP32 | FP32 | 6304 | 7157 | 7617 | 7894 | 8142 |
// FP16 | BF16 | FP32 | FP32 | FP32 | FP32 | 6347 | 7130 | 7628 | 7904 | 8152 |
// BF16 | FP16 | FP32 | BF16 | FP16 | FP32 | 6246 | 7348 | 7722 | 7758 | 7878 |
// BF16 | FP16 | FP32 | FP32 | FP16 | FP32 | 6435 | 7328 | 7324 | 7865 | 8260 |
// BF16 | FP16 | FP32 | FP32 | FP32 | FP32 | 6352 | 7339 | 7312 | 7865 | 8259 |
// BF16 | BF16 | FP32 | BF16 | BF16 | FP32 | 5787 | 6598 | 6993 | 6967 | 7207 |
// BF16 | BF16 | FP32 | FP32 | FP32 | FP32 | 6075 | 6967 | 6955 | 7515 | 7888 |
//
// FP16 | FP16 | FP16 | FP16 | FP16 | FP16 | 7077 | 8535 | 8096 | 8660 | 9136 |
// FP16 | FP16 | FP16 | FP16 | FP16 | FP32 | 6946 | 8561 | 8322 | 8696 | 9103 |
// FP16 | FP16 | FP16 | FP16 | FP32 | FP16 | 6384 | 7742 | 7496 | 7879 | 8254 |
// FP16 | FP16 | FP16 | FP32 | FP16 | FP16 | 6350 | 7747 | 7476 | 7875 | 8263 |
// FP16 | FP16 | FP16 | FP32 | FP32 | FP32 | 7124 | 8505 | 8330 | 8702 | 9091 |
//
// BF16 | BF16 | BF16 | BF16 | BF16 | FP32 | 5861 | 7356 | 7084 | 7426 | 7720 |
// BF16 | BF16 | BF16 | BF16 | FP32 | FP32 | 5676 | 7805 | 7415 | 7724 | 8250 |
// BF16 | BF16 | BF16 | FP32 | BF16 | FP32 | 6243 | 6998 | 7031 | 7355 | 7724 |
// BF16 | BF16 | BF16 | FP32 | FP32 | FP32 | 6367 | 7210 | 7086 | 7544 | 7930 |
//
// FP32 | FP32 | FP16 | FP32 | FP32 | FP16 | 5738 | 6450 | 6130 | 6741 | 7312 |
// FP32 | FP32 | FP16 | FP32 | FP32 | FP32 | 5420 | 7084 | 7171 | 7739 | 8223 |
// FP32 | FP32 | BF16 | FP32 | FP32 | FP32 | 6452 | 7173 | 7200 | 7804 | 8243 |
// FP32 | FP32 | FP32 | FP32 | FP32 | FP32 | 5368 | 7074 | 7165 | 7740 | 8225 |
//
// FP16 | BF16 | FP16 | FP16 | FP32 | FP16 | 5908 | 6873 | 7076 | 7417 | 7745 |
// FP16 | BF16 | FP16 | FP16 | FP32 | FP32 | 6566 | 7598 | 7617 | 7993 | 8489 |
// FP16 | BF16 | FP16 | FP32 | FP32 | FP16 | 5896 | 6873 | 7070 | 7419 | 7739 |
// FP16 | BF16 | FP16 | FP32 | FP32 | FP32 | 5891 | 7602 | 7616 | 8011 | 8494 |
// FP16 | BF16 | BF16 | FP16 | FP32 | FP32 | 6650 | 7581 | 7572 | 7987 | 8462 |
// FP16 | BF16 | BF16 | FP32 | FP32 | FP32 | 6809 | 7534 | 7562 | 8027 | 8467 |
//
// BF16 | FP16 | FP16 | FP32 | FP16 | FP16 | 5937 | 6923 | 7108 | 7527 | 7930 |
// BF16 | FP16 | FP16 | FP32 | FP16 | FP32 | 6414 | 7554 | 7333 | 7981 | 8459 |
// BF16 | FP16 | FP16 | FP32 | FP32 | FP16 | 5976 | 6924 | 7083 | 7525 | 7929 |
// BF16 | FP16 | FP16 | FP32 | FP32 | FP32 | 5383 | 7559 | 7310 | 8008 | 8456 |
// BF16 | FP16 | BF16 | FP32 | FP16 | FP32 | 6575 | 7573 | 7512 | 8028 | 8469 |
// BF16 | FP16 | BF16 | FP32 | FP32 | FP32 | 6699 | 7530 | 7525 | 8000 | 8470 |
//
// BF16 | BF16 | FP16 | FP32 | FP32 | FP16 | 5756 | 6721 | 6780 | 7220 | 7583 |
// BF16 | BF16 | FP16 | FP32 | FP32 | FP32 | 6424 | 7206 | 6954 | 7547 | 7923 |
//
// Optimal register precisions for each memory precision.
//
// Truth Table:
//
// memA | memB | memC | regA | regB | regC |
// ---- | ---- | ---- | ---- | ---- | ---- |
// FP16 | FP16 | FP16 | FP16 | FP16 | FP16 |
// FP16 | FP16 | BF16 | FP16 | FP16 | FP32 |
// FP16 | FP16 | FP32 | FP16 | FP16 | FP32 |
// FP16 | BF16 | FP16 | FP32 | FP32 | FP32 |
// FP16 | BF16 | BF16 | FP32 | FP32 | FP32 |
// FP16 | BF16 | FP32 | FP32 | FP32 | FP32 |
// FP16 | FP32 | FP16 | FP32 | FP32 | FP32 |
// FP16 | FP32 | BF16 | FP32 | FP32 | FP32 |
// FP16 | FP32 | FP32 | FP32 | FP32 | FP32 |
//
// BF16 | FP16 | FP16 | FP32 | FP32 | FP32 |
// BF16 | FP16 | BF16 | FP32 | FP32 | FP32 |
// BF16 | FP16 | FP32 | FP32 | FP32 | FP32 |
// BF16 | BF16 | FP16 | FP32 | FP32 | FP32 |
// BF16 | BF16 | BF16 | FP32 | FP32 | FP32 |
// BF16 | BF16 | FP32 | FP32 | FP32 | FP32 |
// BF16 | FP32 | FP16 | FP32 | FP32 | FP32 |
// BF16 | FP32 | BF16 | FP32 | FP32 | FP32 |
// BF16 | FP32 | FP32 | FP32 | FP32 | FP32 |
//
// FP32 | FP16 | FP16 | FP32 | FP32 | FP32 |
// FP32 | FP16 | BF16 | FP32 | FP32 | FP32 |
// FP32 | FP16 | FP32 | FP32 | FP32 | FP32 |
// FP32 | BF16 | FP16 | FP32 | FP32 | FP32 |
// FP32 | BF16 | BF16 | FP32 | FP32 | FP32 |
// FP32 | BF16 | FP32 | FP32 | FP32 | FP32 |
// FP32 | FP32 | FP16 | FP32 | FP32 | FP32 |
// FP32 | FP32 | BF16 | FP32 | FP32 | FP32 |
// FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
//
// Optimized form of the logic:
//
// If memA and memB are FP16,
//   regA is FP16
//   regB is FP16
// else
//   regA is FP32
//   regB is FP32
// If memA, memB, and memC are FP16,
//   regC is FP16
// else
//   regC is FP32
//
// ## M4
//
// Configuration:
// - 32x32x28
// - inputs are not tranposed
//
// memA | memB | memC | regA | regB | regC |  512 |  768 | 1024 | 1280 | 1536 |
// ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
// FP16 | FP32 | FP32 | FP16 | FP32 | FP32 | 2864 | 3093 | 3128 | 3201 | 3237 |
// FP16 | FP32 | FP32 | FP32 | FP32 | FP32 | 2823 | 3093 | 3128 | 3197 | 3236 |
// FP32 | FP16 | FP32 | FP32 | FP16 | FP32 | 2871 | 3136 | 3180 | 3238 | 3276 |
// FP32 | FP16 | FP32 | FP32 | FP32 | FP32 | 2853 | 3133 | 3180 | 3239 | 3275 |
// BF16 | FP32 | FP32 | BF16 | FP32 | FP32 | 2832 | 3089 | 3122 | 3199 | 3237 |
// BF16 | FP32 | FP32 | FP32 | FP32 | FP32 | 2735 | 2981 | 2999 | 3067 | 3102 |
// FP32 | BF16 | FP32 | FP32 | BF16 | FP32 | 2852 | 3134 | 3189 | 3241 | 3276 |
// FP32 | BF16 | FP32 | FP32 | FP32 | FP32 | 2742 | 3007 | 3053 | 3104 | 3139 |
//
// FP16 | FP16 | FP32 | FP16 | FP16 | FP32 | 2985 | 3285 | 3342 | 3417 | 3458 |
// FP16 | FP16 | FP32 | FP32 | FP32 | FP32 | 2987 | 3285 | 3348 | 3416 | 3458 |
// FP16 | BF16 | FP32 | FP16 | BF16 | FP32 | 3017 | 3290 | 3344 | 3417 | 3460 |
// FP16 | BF16 | FP32 | FP16 | FP32 | FP32 | 2861 | 3130 | 3179 | 3244 | 3282 |
// FP16 | BF16 | FP32 | FP32 | BF16 | FP32 | 2988 | 3281 | 3349 | 3419 | 3459 |
// FP16 | BF16 | FP32 | FP32 | FP32 | FP32 | 2887 | 3127 | 3177 | 3244 | 3281 |
// BF16 | FP16 | FP32 | BF16 | FP16 | FP32 | 2990 | 3284 | 3339 | 3418 | 3458 |
// BF16 | FP16 | FP32 | BF16 | FP32 | FP32 | 2712 | 3277 | 3332 | 3413 | 3459 |
// BF16 | FP16 | FP32 | FP32 | FP16 | FP32 | 2836 | 3108 | 3168 | 3221 | 3258 |
// BF16 | FP16 | FP32 | FP32 | FP32 | FP32 | 2855 | 3108 | 3162 | 3220 | 3257 |
// BF16 | BF16 | FP32 | BF16 | BF16 | FP32 | 2978 | 3285 | 3347 | 3417 | 3458 |
// BF16 | BF16 | FP32 | BF16 | FP32 | FP32 | 2868 | 3119 | 3182 | 3245 | 3281 |
// BF16 | BF16 | FP32 | FP32 | BF16 | FP32 | 2850 | 3114 | 3164 | 3218 | 3257 |
// BF16 | BF16 | FP32 | FP32 | FP32 | FP32 | 2707 | 2943 | 2991 | 3038 | 3070 |
//
// FP16 | FP16 | FP16 | FP16 | FP16 | FP16 | 3069 | 3334 | 3410 | 3471 | 3512 |
// FP16 | FP16 | FP16 | FP16 | FP16 | FP32 | 3025 | 3264 | 3318 | 3394 | 3430 |
// FP16 | FP16 | FP16 | FP16 | FP32 | FP16 | 2708 | 2932 | 2990 | 3039 | 3072 |
// FP16 | FP16 | FP16 | FP32 | FP16 | FP16 | 2714 | 2928 | 2987 | 3041 | 3074 |
// FP16 | FP16 | FP16 | FP32 | FP32 | FP32 | 3016 | 3260 | 3315 | 3392 | 3432 |
//
// BF16 | BF16 | BF16 | BF16 | BF16 | FP32 | 3025 | 3280 | 3324 | 3409 | 3445 |
// BF16 | BF16 | BF16 | BF16 | FP32 | FP32 | 2876 | 3111 | 3164 | 3233 | 3268 |
// BF16 | BF16 | BF16 | FP32 | BF16 | FP32 | 2857 | 3093 | 3138 | 3206 | 3241 |
// BF16 | BF16 | BF16 | FP32 | FP32 | FP32 | 2707 | 2883 | 2939 | 2982 | 3014 |
//
// FP32 | FP32 | FP16 | FP32 | FP32 | FP16 | 2535 | 2791 | 2769 | 2828 | 2861 |
// FP32 | FP32 | FP16 | FP32 | FP32 | FP32 | 2781 | 2999 | 3022 | 3077 | 3115 |
// FP32 | FP32 | BF16 | FP32 | FP32 | FP32 | 2809 | 3052 | 3090 | 3141 | 3178 |
// FP32 | FP32 | FP32 | FP32 | FP32 | FP32 | 2772 | 3017 | 3044 | 3101 | 3141 |
//
// FP16 | BF16 | FP16 | FP16 | BF16 | FP16 | 2702 | 2937 | 2984 | 3040 | 3072 |
// FP16 | BF16 | FP16 | FP16 | BF16 | FP32 | 3010 | 3258 | 3317 | 3392 | 3431 |
// FP16 | BF16 | FP16 | FP32 | BF16 | FP16 | 2712 | 2938 | 2990 | 3040 | 3073 |
// FP16 | BF16 | FP16 | FP32 | BF16 | FP32 | 2997 | 3260 | 3314 | 3394 | 3431 |
// FP16 | BF16 | BF16 | FP16 | BF16 | FP32 | 3028 | 3275 | 3324 | 3403 | 3447 |
// FP16 | BF16 | BF16 | FP32 | BF16 | FP32 | 3008 | 3274 | 3322 | 3407 | 3443 |
//
// BF16 | FP16 | FP16 | BF16 | FP16 | FP16 | 2705 | 2938 | 2987 | 3042 | 3072 |
// BF16 | FP16 | FP16 | BF16 | FP16 | FP32 | 3011 | 3255 | 3310 | 3392 | 3433 |
// BF16 | FP16 | FP16 | BF16 | FP32 | FP16 | 2698 | 2939 | 2988 | 3039 | 3073 |
// BF16 | FP16 | FP16 | BF16 | FP32 | FP32 | 3015 | 3261 | 3313 | 3391 | 3431 |
// BF16 | FP16 | BF16 | BF16 | FP16 | FP32 | 3027 | 3276 | 3323 | 3406 | 3445 |
// BF16 | FP16 | BF16 | BF16 | FP32 | FP32 | 3040 | 3275 | 3327 | 3407 | 3445 |
//
// BF16 | BF16 | FP16 | BF16 | BF16 | FP16 | 2709 | 2938 | 2987 | 3038 | 3073 |
// BF16 | BF16 | FP16 | BF16 | BF16 | FP32 | 3005 | 3260 | 3320 | 3394 | 3433 |
//
// Truth Table:
//
// memA | memB | memC | regA | regB | regC |
// ---- | ---- | ---- | ---- | ---- | ---- |
// FP16 | FP16 | FP16 | FP16 | FP16 | FP16 |
// FP16 | FP16 | BF16 | FP16 | FP16 | FP32 |
// FP16 | FP16 | FP32 | FP16 | FP16 | FP32 |
// FP16 | BF16 | FP16 | FP16 | BF16 | FP32 |
// FP16 | BF16 | BF16 | FP16 | BF16 | FP32 |
// FP16 | BF16 | FP32 | FP16 | BF16 | FP32 |
// FP16 | FP32 | FP16 | FP16 | FP32 | FP32 |
// FP16 | FP32 | BF16 | FP16 | FP32 | FP32 |
// FP16 | FP32 | FP32 | FP16 | FP32 | FP32 |
//
// BF16 | FP16 | FP16 | BF16 | FP16 | FP32 |
// BF16 | FP16 | BF16 | BF16 | FP16 | FP32 |
// BF16 | FP16 | FP32 | BF16 | FP16 | FP32 |
// BF16 | BF16 | FP16 | BF16 | BF16 | FP32 |
// BF16 | BF16 | BF16 | BF16 | BF16 | FP32 |
// BF16 | BF16 | FP32 | BF16 | BF16 | FP32 |
// BF16 | FP32 | FP16 | BF16 | FP32 | FP32 |
// BF16 | FP32 | BF16 | BF16 | FP32 | FP32 |
// BF16 | FP32 | FP32 | BF16 | FP32 | FP32 |
//
// FP32 | FP16 | FP16 | FP32 | FP16 | FP32 |
// FP32 | FP16 | BF16 | FP32 | FP16 | FP32 |
// FP32 | FP16 | FP32 | FP32 | FP16 | FP32 |
// FP32 | BF16 | FP16 | FP32 | BF16 | FP32 |
// FP32 | BF16 | BF16 | FP32 | BF16 | FP32 |
// FP32 | BF16 | FP32 | FP32 | BF16 | FP32 |
// FP32 | FP32 | FP16 | FP32 | FP32 | FP32 |
// FP32 | FP32 | BF16 | FP32 | FP32 | FP32 |
// FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
//
// Optimized form of the logic:
//
// regA is identical to memA
// regB is identical to memB
// If memA, memB, and memC are FP16,
//   regC is FP16
// else
//   regC is FP32
//
// ## Conclusion
//
// Use a common set of logic for both architectures:
//
// ```
// regA is identical to memA
// regB is identical to memB
// If memA, memB, and memC are FP16,
//   regC is FP16
// else
//   regC is FP32
//
// If earlier than M3
//   If memA is BF16,
//     regA is FP32
//   If memB is BF16,
//     regB is FP32
// ```
//
// ========================================================================== //
// Tuning Transposes
// ========================================================================== //
//
//

// MARK: - GEMM Kernel

/// An enumeration of the precisions supported by the kernel.
///
/// If you wish to support quantized precisions, copy/translate the source code
/// and integrate a modified version into your app. Something similar to a Swift
/// `enum` (e.g. C++ `enum class`) could enumerate the quantization formats
/// used by application code. An exemplary set could be:
/// - FP32
/// - FP16
/// - BF16
/// - signed 8-bit integer
/// - s1ezm7
/// - FP8
/// - palletized
///
/// If you support non-floating-point formats, you have the responsibility of
/// authoring correct and performant GPU code for them. A general rule of thumb,
/// is keep the data compressed in `device` or `threadgroup` memory. Transform
/// into a floating point type while loading into the registers. Keep the
/// accumulator in floating point until the output needs to be written.
/// If the output is quantized, it will be compressed when writing back to
/// `device` memory (or `threadgroup` before the async copy in edge cases).
///
/// For example, the reference implementation treats BF16 like a quantized
/// integer type on Apple7 and Apple8 GPUs. It is decompressed to FP32 in
/// registers.
enum GEMMOperandPrecision {
  case FP32
  case FP16
  case BF16
}

/// A configuration for a GEMM kernel.
///
/// The information in this data structure is enough to uniquely identify the
/// kernel. It can be used as a key in a key-value cache.
///
/// ## Usage
///
/// The code for generating the GEMM kernel does not include any assumptions
/// about performance. It should only be responsible for correctly generating
/// a shader source, provided a configuration. The user is responsible for
/// choosing that configuration.
///
/// TODO: Provide a legible Swift implementaton of the selection heuristics.
struct GEMMKernelDescriptor {
  /// Required. The number of matrix elements spanned by each threadgroup.
  /// - Parameter M: Number of output columns spanned.
  /// - Parameter N: Number of output rows spanned.
  /// - Parameter K: Number of loop iterations unrolled.
  ///
  /// Optimal values:
  /// - Apple7 and Apple8: 48x48x24
  /// - Apple9 and later: 32x32x8
  ///
  /// To reach optimal performance on Apple7 and Apple8, the recommended default
  /// value needs to be modified conditionally. When all three operands have
  /// 16-bit memory precisions, change `K` to 32. When the matrix is too small
  /// to saturate all of the GPU cores, change all dimensions to 32x32x32. Even
  /// smaller blocks can be exploited in low-occupancy cases, but 32x32 and
  /// 48x48 are sufficient for general use.
  ///
  /// For simplicity or an out-of-the-box performance test, one can assume
  /// occupancy is always high. But to match the performance of MPS, one must
  /// optimize for small problem sizes on large GPUs.
  ///
  /// ## Choosing Block Size by Precision
  ///
  /// Legend:
  /// - memA: precision for left input matrix, in memory
  /// - memB: precision for right input matrix, in memory
  /// - memC: precision for output matrix, in memory
  /// - regA: precision for left input matrix, in registers
  /// - regB: precision for right input matrix, in registers
  /// - regC: precision for output matrix, in registers
  /// - M1: optimal block size on Apple7 and Apple8
  /// - M3: optimal block size on Apple9 and later
  ///
  /// memA | memB | memC | regA | regB | regC | M1       | M3      |
  /// ---- | ---- | ---- | ---- | ---- | ---- | -------- | ------- |
  /// FP16 | FP16 | FP16 | any  | any  | any  | 48x48x32 | 32x32x8 |
  /// BF16 | BF16 | BF16 | any  | any  | any  | 48x48x32 | 32x32x8 |
  /// FP16 | FP16 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// BF16 | BF16 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// FP16 | FP32 | FP16 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// BF16 | FP32 | BF16 | any  | any  | any  | 48x48x24 | 32x32x8 |
  /// FP32 | FP32 | FP32 | any  | any  | any  | 48x48x24 | 32x32x8 |
  ///
  /// ## Detecting Low-Occupancy Cases
  ///
  /// To determine whether the matrix saturates the GPU, divide the output
  /// matrix's dimensions by 48x48. Round up to the nearest integer. Then,
  /// multiply the number of row blocks by the number of column blocks. The
  /// result is the number of threadgroups dispatched. For example, a C matrix
  /// with dimensions 768x768 would dispatch 256 threadgroups. If you are
  /// batching multiple matrix multiplications into one shader call, multiply
  /// the number of threadgroups by the batch count.
  ///
  /// Next, calculate the target occupancy. Start by finding the GPU core count.
  /// This can be accomplished in many ways; there is a heavily tested reference
  /// implementation [here](https://github.com/philipturner/applegpuinfo). On
  /// macOS, you can query the core count through IORegistry. On iOS, go with a
  /// conservative (meaning more likely to overestimate) estimate of 5 cores on
  /// A14 - A16, 10 cores on M1 - M2.
  ///
  /// When one of the operands is 32-bit, the target occupancy is 6 threadgroups
  /// per core. When all three operands are 16-bit, the target increases to 9
  /// per core. Multiply the number of cores by the number of threadgroups per
  /// core. If the total GPU occupancy is greater than or equal to the number of
  /// matrix blocks, use the smaller blocking scheme.
  ///
  /// For example, the following decision tree would be used on an M1 Max
  /// (32 cores).
  ///
  /// ```
  /// is device Apple9 or later?
  /// yes: use block size 32x32x8
  /// no: continue decision tree [selected decision]
  /// unsure: use block size 48x48x24-32
  ///
  /// compute number of matrix blocks
  /// 768x768 / 48x48 = 16.0 x 16.0
  ///   round floating point (16.0 x 16.0)
  ///   to next greatest integer (16 x 16)
  ///  16 x 16 x (batch size of 1) = 256 threadgroups
  ///
  /// compute target occupancies with 48x48 scheme
  /// 32 x 6 = 192 [selected when A, B, or C is FP32]
  /// 32 x 9 = 288 [selected when every matrix is FP16/BF16]
  ///
  /// prefer 32x32 when 48x48 has low occupancy
  /// if 256 ≤ 192
  ///    choose small block size (32x32x32xFP32)
  /// else
  ///    choose large block size (48x48x24xFP32) [selected]
  /// if 256 ≤ 288
  ///   choose small block size (32x32x32xFP16) [selected]
  /// else
  ///   choose large block size (48x48x32xFP16)
  /// ```
  var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)?
  
  var memoryPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)?
  
  /// Required. Whether async copies will improve performance during inner loop
  /// iterations.
  ///
  /// The default value is `true`. Async copies improve performance on Apple7
  /// and Apple8, but harm performance on Apple9 and later. However, they are
  /// essential for correctness when reading from the edges of unaligned
  /// matrices. Setting the value to `false` means skipping async copies when
  /// doing so will not change the final result.
  ///
  /// > WARNING: Setting this to the wrong value could cause IOCommandBuffer
  /// errors in edge cases. Specifically, on Apple7 or Apple8 when both
  /// `preferAsyncCopy` is `false` and `splits` is not equal to 2x2. The failure
  /// mode causes an infinite loop, which means the GPU becomes unresponsive
  /// until the computer is forced to reboot. If you are uncertain, stay with a
  /// conservative default value of `true`.
  var preferAsyncCopy: Bool = true
  
  /// Set the register precision based on the GPU architecture, and your choice
  /// for memory precision. The following set of logic statements should provide
  /// optimal performance for all permutations of operand precisions.
  ///
  /// ```
  /// regA is identical to memA
  /// regB is identical to memB
  /// If memA, memB, and memC are FP16,
  ///   regC is FP16
  /// else
  ///   regC is FP32
  ///
  /// If earlier than M3
  ///   If memA is BF16,
  ///     regA is FP32
  ///   If memB is BF16,
  ///     regB is FP32
  /// ```
  var registerPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)?
  
  /// Required. The array of SIMDs to divide the threadgroup into.
  ///
  /// Optimal values:
  /// - Apple7 and Apple8: 2x2
  /// - Apple9 and later: 1x1
  var splits: (M: UInt16, N: UInt16)?
  
  /// Required. Whether each of the inputs deviates from row-major order.
  ///
  /// The default value is `(false, false)`.
  var transposeState: (A: Bool, B: Bool) = (false, false)
}

struct GEMMKernel {
  var source: String = ""
  
  // A copy of the block dimensions from the descriptor.
  var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  init(descriptor: GEMMKernelDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let registerPrecisions = descriptor.registerPrecisions,
          let splits = descriptor.splits else {
      fatalError("Descriptor was incomplete.")
    }
    self.blockDimensions = blockDimensions
    self.threadgroupSize = 32 * splits.M * splits.N
    
    // Validate the correctness of register precisions.
    func checkOperandPair(
      memory: GEMMOperandPrecision,
      register: GEMMOperandPrecision
    ) -> Bool {
      // Truth table:
      //
      // memory | register | valid |
      // ------ | -------- | ----- |
      // FP32   | FP32     | yes   |
      // FP32   | FP16     | no    |
      // FP32   | BF16     | no    |
      // FP16   | FP32     | yes   |
      // FP16   | FP16     | yes   |
      // FP16   | BF16     | no    |
      // BF16   | FP32     | yes   |
      // BF16   | FP16     | no    |
      // BF16   | BF16     | yes   |
      //
      // Optimized form of the logic:
      //
      // If the register precision matches the memory precision,
      //   return true
      // If the register precision equals FP32,
      //   return true
      // Otherwise,
      //   return false
      //
      // The logic statements will change if you introduce custom quantized
      // formats. The truth table will grow exponentially. You'll need to
      // add more restrictions on accepted pairs to overcome the combinatorial
      // explosion.
      if register == memory {
        return true
      } else if register == .FP32 {
        return true
      } else {
        return false
      }
    }
    
    guard checkOperandPair(
      memory: memoryPrecisions.A, register: registerPrecisions.A) else {
      fatalError("Operand A had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.B, register: registerPrecisions.B) else {
      fatalError("Operand B had an invalid register precision.")
    }
    guard checkOperandPair(
      memory: memoryPrecisions.C, register: registerPrecisions.C) else {
      fatalError("Operand C had an invalid register precision.")
    }
    if registerPrecisions.C == .BF16 {
      // BF16 has too few mantissa bits to be an accurate accumulator. In
      // addition, switching from FP32 accumulator to BF16 accumulator slows
      // down execution speed on both M1/M2 and M3+.
      fatalError("BF16 cannot be used as the register precision for C.")
    }
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

"""
    
    // Async copies are required for correct behavior in edge cases. We attempt to
    // execute most iterations without async copy, and only the necessary ones with
    // async copy.
    var asyncIterationsStart: String
    if descriptor.preferAsyncCopy {
      asyncIterationsStart = "0"
    } else {
      asyncIterationsStart = "K - (K % K_group)"
    }
    
    // Add the function constants.
    source += """

// Dimensions of each matrix.
// - Limitations to matrix size:
//    - 2^32 in each dimension (M/N/K).
//    - TODO: Test whether the maximum dimension with correct execution is
//      actually 2^16. This will require a testing setup with non-square
//      matrices, as 65536^3 is uncomputable.
//    - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//      good chance this will significantly degrade performance, and require
//      changing the data type of several variables that process addresses. The
//      client is responsible for ensuring correctness and performance with
//      matrices spanning several billion elements in one direction.
//    - The matrix dimensions must be known at compile time, via function
//      constants. Dynamic matrix shapes are beyond the scope of this reference
//      implementation. Dynamic shapes cause a non-negligible regression to
//      shader execution speed. However, they could minimize a compilation
//      latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Alpha and beta constants from BLAS.
constant float alpha [[function_constant(10)]];
constant float beta [[function_constant(11)]];
constant ushort M_simd = \(blockDimensions.M / splits.M);
constant ushort N_simd = \(blockDimensions.N / splits.N);

// Whether each matrix is transposed.
constant bool A_trans = \(descriptor.transposeState.A);
constant bool B_trans = \(descriptor.transposeState.B);
constant uint A_leading_dim = A_trans ? M : K;
constant uint B_leading_dim = B_trans ? K : N;

// Elide work on the edge when matrix dimension < SRAM block dimension.
constant ushort M_modulo = (M % M_simd == 0) ? M_simd : (M % M_simd);
constant ushort N_modulo = (N % N_simd == 0) ? N_simd : (N % N_simd);
constant ushort M_padded = (M < M_simd) ? (M_modulo + 7) / 8 * 8 : M_simd;
constant ushort N_padded = (N < N_simd) ? (N_modulo + 7) / 8 * 8 : N_simd;
constant ushort M_group = \(blockDimensions.M);
constant ushort N_group = \(blockDimensions.N);
constant ushort K_group = \(blockDimensions.K);
constant ushort A_block_leading_dim = (A_trans ? M_group : K_group);
constant ushort B_block_leading_dim = (B_trans ? K_group : N_group);

// There is no padding for M reads/writes.
// There is no padding for N reads/writes.
constant ushort K_group_unpadded = (K % K_group == 0) ? K_group : (K % K_group);
constant ushort K_group_padded = (K_group_unpadded + 7) / 8 * 8;

constant uint K_async_copy = \(asyncIterationsStart);
"""
    
    // Generate names for load/store functions. This code exists because the
    // function for BF16 <-> FP32 coding is different from the others.
    var loadFunctionA: String = "load"
    var loadFunctionB: String = "load"
    var loadFunctionC: String = "store"
    var storeFunctionC: String = "store"
    
    if memoryPrecisions.A == .BF16 {
      switch registerPrecisions.A {
      case .FP32: loadFunctionA = "load_bfloat"
      case .BF16: loadFunctionA = "load"
      default: fatalError(
        "This is an invalid input, and should have been caught earlier.")
      }
    }
    if memoryPrecisions.B == .BF16 {
      switch registerPrecisions.B {
      case .FP32: loadFunctionB = "load_bfloat"
      case .BF16: loadFunctionB = "load"
      default: fatalError(
        "This is an invalid input, and should have been caught earlier.")
      }
    }
    if memoryPrecisions.C == .BF16 {
      switch registerPrecisions.C {
      case .FP32: loadFunctionC = "load_bfloat"
      case .BF16: loadFunctionC = "load"
      default: fatalError(
        "This is an invalid input, and should have been caught earlier.")
      }
      switch registerPrecisions.C {
      case .FP32: storeFunctionC = "store_bfloat"
      case .BF16: storeFunctionC = "store"
      default: fatalError(
        "This is an invalid input, and should have been caught earlier.")
      }
    }
    
    // Allocate threadgroup memory, using the 'memory precision'. This memory
    // is allocated at runtime, either by the user (explicit API call) or by
    // the driver (behind the scenes).
    func createPrecisionSize(_ precision: GEMMOperandPrecision) -> UInt16 {
      // NOTE: Exotic precisions like some LLaMA quantization formats and ezm8
      // have the exponent deinterleaved from the mantissa. Such precisions
      // would require careful consideration of the meaning of per-scalar
      // memory footprint.
      switch precision {
      case .FP32: return 4
      case .FP16: return 2
      case .BF16: return 2
      }
    }
    
    // Allocate thread memory, using the 'register precision'. This memory
    // is allocated by embedding the precision into the assembly code.
    func createPrecisionName(_ precision: GEMMOperandPrecision) -> String {
      // Exotic precisions would not require any special handling here. Good
      // practices dictate that you decode to floating point while filling
      // up the registers. Therefore, the registers will always be floating
      // point.
      switch precision {
      case .FP32: return "float"
      case .FP16: return "half"
      case .BF16: return "bfloat"
      }
    }
    
    // Allocate threadgroup blocks.
    //
    // Dynamic caching works to our advantage here. On M1/M2, this heuristic is
    // needed for optimal performance. On M3+, it is not needed (in many cases,
    // the matrix is aligned and async copies are not used). Dynamic caching
    // means the excessive threadgroup memory allocation won't harm occupancy
    // on M3+.
    var blockBytesA = blockDimensions.M * blockDimensions.K
    var blockBytesB = blockDimensions.K * blockDimensions.N
    var blockBytesC = blockDimensions.M * blockDimensions.N
    blockBytesA *= createPrecisionSize(memoryPrecisions.A)
    blockBytesB *= createPrecisionSize(memoryPrecisions.B)
    blockBytesC *= createPrecisionSize(memoryPrecisions.C)
    threadgroupMemoryAllocation = max(blockBytesA + blockBytesB, blockBytesC)
    
    // Determine the names of the operands.
    let memoryNameA = createPrecisionName(memoryPrecisions.A)
    let memoryNameB = createPrecisionName(memoryPrecisions.B)
    let memoryNameC = createPrecisionName(memoryPrecisions.C)
    let registerNameA = createPrecisionName(registerPrecisions.A)
    let registerNameB = createPrecisionName(registerPrecisions.B)
    let registerNameC = createPrecisionName(registerPrecisions.C)
    
    // Add the utility functions for accessing memory.
    source += """

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

METAL_FUNC void prefetch(
  threadgroup \(memoryNameA) *A_block,
  device \(memoryNameA) *A,
  ushort2 A_tile_src,
  uint2 A_offset,
  threadgroup \(memoryNameB) *B_block,
  device \(memoryNameB) *B,
  ushort2 B_tile_src,
  uint2 B_offset,
  uint k
) {
  A_tile_src.x = min(uint(K_group), K - k);
  B_tile_src.y = min(uint(K_group), K - k);
  auto A_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
    A, A_leading_dim, A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
    B, B_leading_dim, B_offset, B_trans);
  
  // Rounded-up ceiling for the threadgroup block.
  const uint K_edge_floor = K - K_group_unpadded;
  const uint K_edge_ceil = K_edge_floor + K_group_padded;
  ushort K_padded;
  if (K_edge_floor == K_group) {
    K_padded = K_group;
  } else {
    K_padded = min(uint(K_group), K_edge_ceil - k);
  }
  ushort2 A_tile_dst(K_padded, A_tile_src.y);
  ushort2 B_tile_dst(B_tile_src.x, K_padded);
  
  simdgroup_event events[2];
  events[0].async_copy(A_block, A_block_leading_dim, A_tile_dst, A_src,
                       A_leading_dim, A_tile_src, A_trans);
  events[1].async_copy(B_block, B_block_leading_dim, B_tile_dst, B_src,
                       B_leading_dim, B_tile_src, B_trans);
  simdgroup_event::wait(2, events);
}

// Convention: allocate all memory, including register memory, in the top-level
// function. Do not allow utility functions to allocate any memory internally,
// even if temporary. It complicates estimates of occupancy.
//
// This convention was used to decide where 'previous_sram' is declared and
// allocated.
METAL_FUNC void partial_accumulate(
  thread simdgroup_matrix_storage<\(registerNameC)> *previous_sram,
  thread simdgroup_matrix_storage<\(registerNameC)> *current_sram,
  threadgroup \(memoryNameC) *C_block
) {
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto previous = get_sram(previous_sram, N_simd, ushort2(n, 0));
      previous->\(loadFunctionC)(C_block, N_group, ushort2(n, m));
    }
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto previous = get_sram(previous_sram, N_simd, ushort2(n, 0));
      auto current = get_sram(current_sram, N_simd, ushort2(n, m));
      float2 C_old = float2(previous->thread_elements()[0]);
      float2 C_new = float2(current->thread_elements()[0]);
      current->thread_elements()[0] = vec<\(registerNameC), 2>(
        C_old * beta + C_new);
    }
  }
}

METAL_FUNC void async_access_accumulator(
  threadgroup \(memoryNameC) *C_block,
  device \(memoryNameC) *C,
  uint2 C_offset,
  bool is_store
) {
  ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                 min(uint(M_group), M - C_offset.y));
  auto C_src = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
    C, N, C_offset);
  
  simdgroup_event event;
  if (is_store) {
    event.async_copy(C_src, N, C_tile, C_block, N_group, C_tile);
  } else {
    event.async_copy(C_block, N_group, C_tile, C_src, N, C_tile);
    simdgroup_event::wait(1, &event);
  }
}

METAL_FUNC void store_accumulator(
  thread simdgroup_matrix_storage<\(registerNameC)> *C_sram,
  device \(memoryNameC) *C_dst,
  bool m_is_edge,
  bool n_is_edge
) {
  const ushort m_start = (m_is_edge) ? M_modulo : 0;
  const ushort n_start = (n_is_edge) ? N_modulo : 0;
  const ushort m_end = (m_is_edge) ? M_simd : M_modulo;
  const ushort n_end = (n_is_edge) ? N_simd : N_modulo;
  
#pragma clang loop unroll(full)
  for (ushort m = m_start; m < m_end; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = n_start; n < n_end; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, N_simd, origin);
      C->\(storeFunctionC)(C_dst, N, origin);
    }
  }
}

"""
    
    // Add the multiply-accumulate inner loop.
    source += """

// One iteration of the MACC loop, effectively k=8 iterations.
METAL_FUNC void multiply_accumulate(
  thread simdgroup_matrix_storage<\(registerNameA)> *A_sram,
  thread simdgroup_matrix_storage<\(registerNameB)> *B_sram,
  thread simdgroup_matrix_storage<\(registerNameC)> *C_sram,
  const device \(memoryNameA) *A_src,
  const device \(memoryNameB) *B_src,
  bool accumulate = true
) {
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
    A->\(loadFunctionA)(A_src, A_leading_dim, origin, A_trans);
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < N_padded; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, N_simd, origin);
    B->\(loadFunctionB)(B_src, B_leading_dim, origin, B_trans);
  }
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto A = get_sram(A_sram, 8, ushort2(0, m));
      auto B = get_sram(B_sram, N_simd, ushort2(n, 0));
      auto C = get_sram(C_sram, N_simd, ushort2(n, m));
      C->multiply(*A, *B, accumulate);
    }
  }
}

// One iteration of the MACC loop, effectively k=8 iterations.
METAL_FUNC void multiply_accumulate(
  thread simdgroup_matrix_storage<\(registerNameA)> *A_sram,
  thread simdgroup_matrix_storage<\(registerNameB)> *B_sram,
  thread simdgroup_matrix_storage<\(registerNameC)> *C_sram,
  const threadgroup \(memoryNameA) *A_block,
  const threadgroup \(memoryNameB) *B_block,
  bool accumulate = true
) {
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
    A->\(loadFunctionA)(A_block, A_block_leading_dim, origin, A_trans);
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < N_padded; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, N_simd, origin);
    B->\(loadFunctionB)(B_block, B_block_leading_dim, origin, B_trans);
  }
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto A = get_sram(A_sram, 8, ushort2(0, m));
      auto B = get_sram(B_sram, N_simd, ushort2(n, 0));
      auto C = get_sram(C_sram, N_simd, ushort2(n, m));
      C->multiply(*A, *B, accumulate);
    }
  }
}

"""
    
    // Add the setup portion where the addresses are prepared.
    source += """

kernel void gemm(device \(memoryNameA) *A [[buffer(0)]],
                 device \(memoryNameB) *B [[buffer(1)]],
                 device \(memoryNameC) *C [[buffer(2)]],
                 
                 threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                 
                 uint3 gid [[threadgroup_position_in_grid]],
                 ushort sidx [[simdgroup_index_in_threadgroup]],
                 ushort lane_id [[thread_index_in_simdgroup]])
{
  // The compiler optimizes away the >100 registers allocated in the source
  // code. Only a tiny fraction of these will be used, even on M3 with giant
  // block sizes. For example, a 48x48 accumulator on M3 would require 36
  // array elements. 512 is simply a conservative upper bound to minimize the
  // cost of processing the registers during compilation.
  simdgroup_matrix_storage<\(registerNameA)> A_sram[512];
  simdgroup_matrix_storage<\(registerNameB)> B_sram[512];
  simdgroup_matrix_storage<\(registerNameC)> C_sram[512];
  auto A_block = (threadgroup \(memoryNameA)*)(threadgroup_block);
  auto B_block = (threadgroup \(memoryNameB)*)(threadgroup_block + \(blockBytesA));
  ushort2 sid(sidx % \(splits.N), sidx / \(splits.N));
  ushort2 offset_in_simd = simdgroup_matrix_storage<float>::offset(lane_id);
  
  uint2 A_offset(K_async_copy, gid.y * M_group);
  uint2 B_offset(gid.x * N_group, K_async_copy);
  {
    uint C_base_offset_x = B_offset.x + sid.x * N_simd;
    uint C_base_offset_y = A_offset.y + sid.y * M_simd;
    if (C_base_offset_x >= N || C_base_offset_y >= M) {
      return;
    }
  }
  
  ushort2 offset_in_group(sid.x * N_simd + offset_in_simd.x,
                          sid.y * M_simd + offset_in_simd.y);
  
  ushort2 A_tile_src;
  ushort2 B_tile_src;
  if (sidx == 0) {
    uint k = K_async_copy;
    A_tile_src.y = min(uint(M_group), M - A_offset.y);
    B_tile_src.x = min(uint(N_group), N - B_offset.x);
    prefetch(A_block, A, A_tile_src, A_offset,
             B_block, B, B_tile_src, B_offset, k);
  }
  A_offset.x = 0;
  B_offset.y = 0;
  
  if (K > K_group) {
#pragma clang loop unroll(full)
    for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N_padded; n += 8) {
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, N_simd, origin);
        *C = simdgroup_matrix_storage<\(registerNameC)>(0);
      }
    }
  }

"""
    
    // Add the matrix multiplication iterations.
    source += """

  for (ushort k = 0; k < K_async_copy; k += 8) {
    uint2 A_src_offset(offset_in_simd.x, offset_in_group.y);
    uint2 B_src_offset(offset_in_group.x, offset_in_simd.y);
    auto A_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
      A, A_leading_dim, A_offset + A_src_offset, A_trans);
    auto B_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
      B, B_leading_dim, B_offset + B_src_offset, B_trans);

    bool accumulate = !(K <= K_group && k == 0);
    multiply_accumulate(A_sram, B_sram, C_sram, A_src, B_src, accumulate);

    A_offset.x += 8;
    B_offset.y += 8;
  }

  for (uint K_floor = K_async_copy; K_floor < K; K_floor += K_group) {
    ushort2 A_block_offset(offset_in_simd.x, offset_in_group.y);
    ushort2 B_block_offset(offset_in_group.x, offset_in_simd.y);
    auto A_block_src = simdgroup_matrix_storage<\(memoryNameA)>::apply_offset(
      A_block, A_block_leading_dim, A_block_offset, A_trans);
    auto B_block_src = simdgroup_matrix_storage<\(memoryNameB)>::apply_offset(
      B_block, B_block_leading_dim, B_block_offset, B_trans);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  
#pragma clang loop unroll(full)
    for (ushort k = 0; k < K_group_padded; k += 8) {
      bool accumulate = !(K <= K_group && k == 0);
      multiply_accumulate(A_sram, B_sram, C_sram,
                          A_block_src, B_block_src, accumulate);
      A_block_src += A_trans ? 8 * A_block_leading_dim : 8;
      B_block_src += B_trans ? 8 : 8 * B_block_leading_dim;
    }
    
    if (K_floor + K_group < K) {
#pragma clang loop unroll(full)
      for (ushort k = K_group_padded; k < K_group; k += 8) {
        multiply_accumulate(A_sram, B_sram, C_sram,
                            A_block_src, B_block_src);
        A_block_src += A_trans ? 8 * A_block_leading_dim : 8;
        B_block_src += B_trans ? 8 : 8 * B_block_leading_dim;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        uint K_next = K_floor + K_group;
        A_offset.x = K_next;
        B_offset.y = K_next;
        prefetch(A_block, A, A_tile_src, A_offset,
                 B_block, B, B_tile_src, B_offset, K_next);
      }
    }
  }

"""
    
    // Add the cleanup portion where the accumulator is stored.
    source += """

  if (alpha != 1) {
#pragma clang loop unroll(full)
    for (int m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
      for (int n = 0; n < N_padded; n += 8) {
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, N_simd, origin);
        *(C->thread_elements()) *= alpha;
      }
    }
  }
  
  auto C_block = (threadgroup \(memoryNameC)*)(threadgroup_block);
  uint2 C_offset(B_offset.x, A_offset.y);
  ushort2 C_block_offset = offset_in_group.xy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (beta != 0) {
    simdgroup_matrix_storage<\(registerNameC)> previous_sram[512];
    if (sidx == 0) {
      async_access_accumulator(C_block, C, C_offset, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    auto C_block_src = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
      C_block, N_group, C_block_offset);
    partial_accumulate(previous_sram, C_sram, C_block_src);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  if ((M % 8 != 0) || (N % 8 != 0)) {
    auto C_block_src = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
      C_block, N_group, C_block_offset);
#pragma clang loop unroll(full)
    for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N_padded; n += 8) {
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, N_simd, origin);
        C->\(storeFunctionC)(C_block_src, N_group, origin);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      async_access_accumulator(C_block, C, C_offset, true);
    }
  } else {
    uint2 matrix_origin = C_offset + uint2(C_block_offset);
    auto C_src = simdgroup_matrix_storage<\(memoryNameC)>::apply_offset(
      C, N, matrix_origin);
    store_accumulator(C_sram, C_src, false, false);
    
    const uint M_edge_floor = M - M % M_simd;
    const uint N_edge_floor = N - N % N_simd;
    if (matrix_origin.y < M_edge_floor) {
      store_accumulator(C_sram, C_src, true, false);
    }
    if (matrix_origin.x < N_edge_floor) {
      store_accumulator(C_sram, C_src, false, true);
      if (matrix_origin.y < M_edge_floor) {
        store_accumulator(C_sram, C_src, true, true);
      }
    }
  }
}

"""
  }
}

// MARK: - Header Sources

// Create the source code for the 'metal_simdgroup_event' header.
func createMetalSimdgroupEvent() -> String {
  // Return the source string.
  return """
// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

// Invoking the generation of LLVM bitcode for async copies.
//
//   %struct._simdgroup_event_t = type opaque
//
struct _simdgroup_event_t;

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p3i8.p1i8(
//       i64, i64, i8 addrspace(3)* nocapture writeonly,
//       i64, i64, <2 x i64>, i8 addrspace(1)* nocapture readonly,
//       i64, i64, <2 x i64>, <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong, threadgroup void *,
  ulong, ulong, ulong2, const device void *,
  ulong, ulong, ulong2, long2, int)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p1i8.p3i8(
//       i64, i64, i8 addrspace(1)* nocapture writeonly,
//       i64, i64, <2 x i64>, i8 addrspace(3)* nocapture readonly,
//       i64, i64, <2 x i64>, <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong, device void *,
  ulong, ulong, ulong2, const threadgroup void *,
  ulong, ulong, ulong2, long2, int)
  __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: convergent nounwind
//   declare void
//     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
//     local_unnamed_addr #3
//
void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };
  
  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}
    
    template <typename T>
    METAL_FUNC void async_copy(threadgroup T *dst, ushort dst_elements_per_row, ushort2 dst_tile_dimensions, const device T *src, uint src_elements_per_row, ushort2 src_tile_dimensions, bool transpose_matrix = false, simdgroup_async_copy_clamp_mode clamp_mode = simdgroup_async_copy_clamp_mode::clamp_to_zero) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(sizeof(T), alignof(T), reinterpret_cast<threadgroup void *>(dst), ushort(dst_elements_per_row), 1, ulong2(dst_tile_dimensions), reinterpret_cast<const device void *>(src), uint(src_elements_per_row), 1, ulong2(src_tile_dimensions), long2(0), static_cast<int>(clamp_mode));
    }
    
    template <typename T>
    METAL_FUNC void async_copy(device T *dst, uint dst_elements_per_row, ushort2 dst_tile_dimensions, const threadgroup T *src, ushort src_elements_per_row, ushort2 src_tile_dimensions, bool transpose_matrix = false) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(sizeof(T), alignof(T), reinterpret_cast<device void *>(dst), uint(dst_elements_per_row), 1, ulong2(dst_tile_dimensions), reinterpret_cast<const threadgroup void *>(src), ushort(src_elements_per_row), 1, ulong2(src_tile_dimensions), long2(0), 0);
    }
    
    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }
    
  private:
    // Invoking the generation of LLVM bitcode for async copies.
    //
    //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
    //
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT
"""
}

// Create the source code for the 'metal_simdgroup_matrix_storage' header.
func createMetalSimdgroupMatrixStorage() -> String {
  // Find the patterns between the load/store functions:
  // - device has 'uint' elements_per_row
  // - threadgroup has 'ushort' elements_per_row
  // - both have 'ushort2' matrix_origin
  //
  // The origin is 'ushort2' because the 32-bit part of the address should have
  // been applied previously during 'apply_offset'. The 16-bit part should be
  // hard-coded into the assembly when the GEMM loop is unrolled.
  //
  // Transpose path:
  // - load: reads two values; should split each one onto a separate line.
  //   - overwrites the value of *thread_elements() with a new vec<T, 2>
  // - store: the two instructions are on two separate lines.
  //   - fetches from lane 0 or 1 of thread_elements()[0]
  // - adds 0 or 1 to the hard-coded matrix_origin.x
  //
  // Address generation:
  // - casts some intermediate address fragments to 'ulong' for 'device'
  // - keeps all address fragments in 'ushort' for 'threadgroup'
  // - inconsistent conventions for transposed and non-transposed
  //   - should default to 32-bit in the new code
  //   - treat them as if they're guaranteed to be compiled away
  //   - this may harm performance for dynamic shapes; fix later while tuning
  //     kernel performance
  
  enum AddressSpace {
    case device
    case threadgroup
    
    var keyword: String {
      switch self {
      case .device: return "device"
      case .threadgroup: return "threadgroup"
      }
    }
    
    var offsetType: String {
      switch self {
      case .device: return "uint"
      case .threadgroup: return "ushort"
      }
    }
  }
  
  enum Action {
    case load
    case store
  }
  
  struct MemoryAccessDescriptor {
    var action: Action?
    var addressSpace: AddressSpace?
    var decodingBF16: Bool?
    var indentationSpaceCount: Int = .zero
  }
  
  func createMemoryAccess(
    descriptor: MemoryAccessDescriptor
  ) -> String {
    guard let action = descriptor.action,
          let addressSpace = descriptor.addressSpace,
          let decodingBF16 = descriptor.decodingBF16 else {
      fatalError("Descriptor was incomplete.")
    }
    let indentation = String(
      repeating: " ", count: descriptor.indentationSpaceCount)
    
    // Determine the arguments.
    var arguments: [String] = []
    func addPointerArgument(dataType: String) {
      if action == .load {
        arguments.append("const \(addressSpace.keyword) \(dataType) *src")
      } else {
        arguments.append("\(addressSpace.keyword) \(dataType) *dst")
      }
    }
    if decodingBF16 {
      addPointerArgument(dataType: "bfloat")
    } else {
      addPointerArgument(dataType: "U")
    }
    arguments.append("\(addressSpace.offsetType) elements_per_row")
    arguments.append("ushort2 matrix_origin")
    arguments.append("bool transpose_matrix = false")
    
    // Create the warning comment.
    var output: String = ""
    if decodingBF16 {
      output += "\(indentation)// WARNING: 'T' must be 'float'.\n"
    }
    
    // Create the function signature.
    if !decodingBF16 {
      output += "\(indentation)template <typename U>\n"
    }
    output += "\(indentation)METAL_FUNC void"
    if action == .load {
      output += " load"
    } else {
      output += " store"
    }
    if decodingBF16 {
      output += "_bfloat"
    }
    output += "("
    for argumentID in arguments.indices {
      let argument = arguments[argumentID]
      output += argument
      if argumentID < arguments.count - 1 {
        output += ", "
      }
    }
    output += ") {\n"
    
    func createAddress(transposed: Bool, offset: Int) -> String {
      let lineY = "\(addressSpace.offsetType)(matrix_origin.y)"
      var lineX = "matrix_origin.x + \(offset)"
      lineX = "\(addressSpace.offsetType)(\(lineX))"
      
      if transposed {
        return "\(lineX) * elements_per_row + \(lineY)"
      } else {
        return "\(lineY) * elements_per_row + \(lineX)"
      }
    }
    
    // NOTE: This function seems to not be optimized for BF16. It could be
    // an unnoticed area of dismal performance on M1. Investigate this issue
    // before shipping the kernel in production.
    func createTwoPartAccess(transposed: Bool) -> [String] {
      // Generate the addresses.
      var lines: [String] = []
      for laneID in 0..<2 {
        lines.append(
          "\(addressSpace.offsetType) address\(laneID) = " +
          createAddress(transposed: transposed, offset: laneID))
      }
      
      // Determine the best way to explicitly denote the scalar's data type.
      var U: String
      if decodingBF16 {
        U = "bfloat"
      } else {
        U = "U"
      }
      
      if action == .load {
        for laneID in 0..<2 {
          lines.append("\(U) lane\(laneID) = src[address\(laneID)]")
        }
      }
      
      for laneID in 0..<2 {
        var projectedLaneID: Int
        if decodingBF16 {
          projectedLaneID = laneID * 2 + 1
        } else {
          projectedLaneID = laneID
        }
        if action == .load {
          lines.append(
            "((thread \(U)*)thread_elements())[\(projectedLaneID)] = " +
            "\(U)(lane\(laneID))")
        } else {
          lines.append(
            "\(U) lane\(laneID) = " +
            "((thread \(U)*)thread_elements())[\(projectedLaneID)]")
        }
      }
      
      if action == .store {
        for laneID in 0..<2 {
          lines.append("dst[address\(laneID)] = lane\(laneID)")
        }
      }
      return lines
    }
    
    func createOnePartAccess() -> [String] {
      var lines: [String] = []
      do {
        let address = createAddress(transposed: false, offset: 0)
        lines.append("auto combinedAddress = \(address)")
      }
      if action == .load {
        if decodingBF16 {
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "float memoryForm = " +
            "*(const \(addressSpace.keyword) float*)(src + combinedAddress)")
          lines.append(
            "((thread float*)&registerForm)[1] = memoryForm")
          lines.append(
            "((thread bfloat*)&registerForm)[1] = " +
            "((thread bfloat*)&memoryForm)[0]")
          lines.append(
            "((thread bfloat4*)thread_elements())[0] = registerForm")
        } else {
          lines.append(
            "vec<U, 2> memoryForm = " +
            "*(const \(addressSpace.keyword) vec<U, 2>*)(src + combinedAddress)")
          lines.append(
            "*(thread_elements()) = vec<T, 2>(memoryForm)")
        }
      } else {
        if decodingBF16 {
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "registerForm[2] = registerForm[1]")
          lines.append(
            "float memoryForm = ((thread float*)&registerForm)[1]")
          lines.append(
            "*(\(addressSpace.keyword) float*)(dst + combinedAddress) = " +
            "memoryForm")
        } else {
          lines.append(
            "vec<T, 2> registerForm = *(thread_elements())")
          lines.append(
            "*(\(addressSpace.keyword) vec<U, 2>*)(dst + combinedAddress) = " +
            "vec<U, 2>(registerForm)")
        }
      }
      return lines
    }
    
    func addBlockContents(_ block: [String]) -> [String] {
      block.map { "  \($0);" }
    }
    
    // Determine the lines of the 'if' block.
    var body: [String] = []
    body.append("if (transpose_matrix) {")
    body += addBlockContents(createTwoPartAccess(transposed: true))
    
    // Determine the lines of the 'else if' block.
    body.append("} else if (elements_per_row % 2 != 0) {")
    body += addBlockContents(createTwoPartAccess(transposed: false))
    
    // Determine the lines of the 'else' block.
    body.append("} else {")
    body += addBlockContents(createOnePartAccess())
    body.append("}")
    
    // Create the function body.
    for line in body {
      output += "\(indentation)  \(line)\n"
    }
    output += "\(indentation)}\n"
    return output
  }
  
  // Add the first section of the shader.
  var output: String = ""
  output += """
// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    
    storage_type t;
    
    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }
    
    METAL_FUNC simdgroup_matrix_storage() thread = default;
    
    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }
    
    METAL_FUNC static ushort2 offset(ushort thread_index_in_simdgroup) {
      // https://patents.google.com/patent/US11256518B2
      ushort lane_id = thread_index_in_simdgroup;
      ushort quad_id = lane_id / 4;
      
      constexpr ushort QUADRANT_SPAN_M = 4;
      constexpr ushort THREADS_PER_QUADRANT = 8;
      ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
      ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
      ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;
      
      ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
      ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
      ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;
      
      return ushort2(N_in_simd, M_in_simd);
    }
    
    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }
    
    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }


"""
  
  var desc = MemoryAccessDescriptor()
  desc.indentationSpaceCount = 4
  
  for action in [Action.load, .store] {
    for addressSpace in [AddressSpace.device, .threadgroup] {
      for decodingBF16 in [false, true] {
        desc.action = action
        desc.addressSpace = addressSpace
        
        desc.decodingBF16 = decodingBF16
        output += createMemoryAccess(descriptor: desc)
        output += "\n"
      }
    }
  }
  
  // Add the last section of the header.
  output += """
    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE

"""
  return output
}

// A description of a matrix multiplication.
struct ProblemDescriptor {
  // The number of equally sized multiplications that run in parallel. Batching
  // is out of scope for the reference implementation. However, there should
  // be a guide for clients that wish to modify the shader, in ways that
  // increase the compute workload. For example, by batching the multiplication
  // of (sub)matrices located at arbitrary pointers in memory (with potentially
  // nonuniform stride or noncontiguous padding).
  var batchDimensions: Int = 1
  
  var matrixDimensions: (Int, Int, Int)?
}

// Implementation of the selection heuristics.
//
// TODO: Copy the code from AppleGPUInfo that reads from IORegistry, instead
// of introducing a package dependecy on AppleGPUInfo. Not all clients will be
// able to import the AppleGPUInfo library.
func createOptimalKernelDescriptor(
  problemDescriptor: ProblemDescriptor
) -> GEMMKernelDescriptor {
  fatalError("Not implemented.")
}

#if os(macOS)
// Finds the core count on macOS devices, using IORegistry.
func findCoreCount() -> Int {
  // Create a matching dictionary with "AGXAccelerator" class name
  let matchingDict = IOServiceMatching("AGXAccelerator")
  
  // Get an iterator for matching services
  var iterator: io_iterator_t = 0
  do {
    let io_registry_error =
    IOServiceGetMatchingServices(
      kIOMainPortDefault, matchingDict, &iterator)
    guard io_registry_error == 0 else {
      fatalError(
        "Encountered IORegistry error code \(io_registry_error)")
    }
  }
  
  // Get the first (and only) GPU entry from the iterator
  let gpuEntry = IOIteratorNext(iterator)
  
  // Check if the entry is valid
  if gpuEntry == MACH_PORT_NULL {
    fatalError(
      "Error getting GPU entry at \(#file):\(#line - 5)")
  }
  
  // Release the iterator
  IOObjectRelease(iterator)
  
  // Get the "gpu-core-count" property from gpuEntry
  let key = "gpu-core-count"
  let options: IOOptionBits = 0 // No options needed
  let gpuCoreCount = IORegistryEntrySearchCFProperty(
    gpuEntry, kIOServicePlane, key as CFString, nil, options)
  
  // Check if the property is valid
  if gpuCoreCount == nil {
    fatalError(
      "Error getting gpu-core-count property at \(#file):\(#line - 6)")
  }
  
  // Cast the property to CFNumberRef
  let gpuCoreCountNumber = gpuCoreCount as! CFNumber
  
  // Check if the number type is sInt64
  let type = CFNumberGetType(gpuCoreCountNumber)
  if type != .sInt64Type {
    fatalError(
      "Error: gpu-core-count is not sInt64 at \(#file):\(#line - 3)")
  }
  
  // Get the value of the number as Int64
  var value: Int64 = 0
  let result = CFNumberGetValue(gpuCoreCountNumber, type, &value)
  
  // Check for errors
  if result == false {
    fatalError(
      " Error getting value of gpu-core-count at \(#file):\(#line - 5)")
  }
  
  return Int(value)
}
#endif

// MARK: - Profiling

func runApplication() {
  print("Hello, console.")
  
  var gemmKernelDesc = GEMMKernelDescriptor()
  gemmKernelDesc.memoryPrecisions = (.FP32, .FP32, .FP32)
  gemmKernelDesc.registerPrecisions = (.FP32, .FP32, .FP32)
  gemmKernelDesc.blockDimensions = (M: 48, N: 48, K: 24)
  gemmKernelDesc.splits = (M: 2, N: 2)
  gemmKernelDesc.preferAsyncCopy = true
  
  // profileProblemSize(64, descriptor: gemmKernelDesc)
}

// Set up a continuous correctness test, before making further changes to the
// kernel.
func profileProblemSize(
  _ problemSize: Int,
  descriptor: GEMMKernelDescriptor
) {
  // Allocate FP32 memory for the operands.
  var A = [Float](repeating: .zero, count: problemSize * problemSize)
  var B = [Float](repeating: .zero, count: problemSize * problemSize)
  var C = [Float](repeating: .zero, count: problemSize * problemSize)
  
  // Initialize A as the 2nd-order periodic Laplacian.
  for diagonalID in 0..<problemSize {
    let diagonalAddress = diagonalID * problemSize + diagonalID
    A[diagonalAddress] = -2
    
    let leftColumnID = (diagonalID + problemSize - 1) % problemSize
    let leftSubDiagonalAddress = diagonalID * problemSize + leftColumnID
    A[leftSubDiagonalAddress] = 1
    
    let rightColumnID = (diagonalID + problemSize + 1) % problemSize
    let rightSubDiagonalAddress = diagonalID * problemSize + rightColumnID
    A[rightSubDiagonalAddress] = 1
  }
  
  // Initialize B to random numbers.
  for rowID in 0..<problemSize {
    for columnID in 0..<problemSize {
      let address = rowID * problemSize + columnID
      let entry = Float.random(in: 0..<1)
      B[address] = entry
    }
  }
  
  // Initialize the context.
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let context = (device: device, commandQueue: commandQueue)
  
  func createBuffer(
    _ originalData: [Float],
    _ precision: GEMMOperandPrecision
  ) -> MTLBuffer {
    // Add random numbers to expose out-of-bounds accesses.
    var augmentedData = originalData
    for _ in 0..<originalData.count {
      let randomNumber = Float.random(in: -2...2)
      augmentedData.append(randomNumber)
    }
    
    // Allocate enough memory to store everything in Float32.
    let bufferSize = augmentedData.count * 4
    let buffer = context.device.makeBuffer(length: bufferSize)!
    
    // Copy the data into the buffer.
    switch precision {
    case .FP32:
      let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
      for i in augmentedData.indices {
        pointer[i] = augmentedData[i]
      }
    case .FP16:
      let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
      for i in augmentedData.indices {
        pointer[i] = Float16(augmentedData[i])
      }
    case .BF16:
      let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
      for i in augmentedData.indices {
        let value32 = augmentedData[i].bitPattern
        let value16 = unsafeBitCast(value32, to: SIMD2<UInt16>.self)[1]
        pointer[i] = value16
      }
    }
    return buffer
  }
  
  // Multiply A with B.
  do {
    // Generate the kernel.
    let gemmKernel = GEMMKernel(descriptor: descriptor)
    let library = try! context.device.makeLibrary(
      source: gemmKernel.source, options: nil)
    
    // Set the function constants.
    let constants = MTLFunctionConstantValues()
    var M: Int = problemSize
    var N: Int = problemSize
    var K: Int = problemSize
    var alpha: Float = 1
    var beta: Float = 0
    constants.setConstantValue(&M, type: .uint, index: 0)
    constants.setConstantValue(&N, type: .uint, index: 1)
    constants.setConstantValue(&K, type: .uint, index: 2)
    constants.setConstantValue(&alpha, type: .float, index: 10)
    constants.setConstantValue(&beta, type: .float, index: 11)
    
    let function = try! library.makeFunction(
      name: "gemm", constantValues: constants)
    let pipeline = try! context.device
      .makeComputePipelineState(function: function)
    
    // Create the buffers.
    let bufferA = createBuffer(A, descriptor.memoryPrecisions!.A)
    let bufferB = createBuffer(B, descriptor.memoryPrecisions!.B)
    let bufferC = createBuffer(C, descriptor.memoryPrecisions!.C)
    
    // Profile the latency of matrix multiplication.
    for _ in 0..<15 {
      let duplicatedCommandCount: Int = 20
      
      // Encode the GPU command.
      let commandBuffer = context.commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(gemmKernel.threadgroupMemoryAllocation), index: 0)
      encoder.setBuffer(bufferA, offset: 0, index: 0)
      encoder.setBuffer(bufferB, offset: 0, index: 1)
      encoder.setBuffer(bufferC, offset: 0, index: 2)
      for _ in 0..<duplicatedCommandCount {
        func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
          (target + Int(granularity) - 1) / Int(granularity)
        }
        let gridSize = MTLSize(
          width: ceilDivide(N, gemmKernel.blockDimensions.N),
          height: ceilDivide(M, gemmKernel.blockDimensions.M),
          depth: 1)
        let groupSize = MTLSize(
          width: Int(gemmKernel.threadgroupSize),
          height: 1,
          depth: 1)
        encoder.dispatchThreadgroups(
          gridSize, threadsPerThreadgroup: groupSize)
      }
      encoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      
      // Determine the time taken.
      let start = commandBuffer.gpuStartTime
      let end = commandBuffer.gpuEndTime
      let latency = end - start
      let latencyMicroseconds = Int(latency / 1e-6)
      
      // Determine the amount of work done.
      var operations = 2 * problemSize * problemSize * problemSize
      operations = operations * duplicatedCommandCount
      let gflops = Int(Double(operations) / Double(latency) / 1e9)
      
      // Report the results.
      print(latencyMicroseconds, "μs", gflops, "GFLOPS")
    }
    
    // Copy the results to C.
    do {
      let precision = descriptor.memoryPrecisions!.C
      let raw = bufferC.contents()
      for rowID in 0..<problemSize {
        for columnID in 0..<problemSize {
          let address = rowID * problemSize + columnID
          var entry32: Float
          
          switch precision {
          case .FP32:
            let casted = raw.assumingMemoryBound(to: Float.self)
            entry32 = casted[address]
          case .FP16:
            let casted = raw.assumingMemoryBound(to: Float16.self)
            let entry16 = casted[address]
            entry32 = Float(entry16)
          case .BF16:
            let casted = raw.assumingMemoryBound(to: UInt16.self)
            let entry16 = casted[address]
            let entry16x2 = SIMD2<UInt16>(.zero, entry16)
            entry32 = unsafeBitCast(entry16x2, to: Float.self)
          }
          C[address] = entry32
        }
      }
    }
  }
  
  // Choose an error threshold.
  func createErrorThreshold(precision: GEMMOperandPrecision) -> Float {
    switch precision {
    case .FP32: return 1e-5
    case .FP16: return 5e-3
    case .BF16: return 5e-2
    }
  }
  var errorThreshold: Float = 0
  do {
    let memoryPrecisions = descriptor.memoryPrecisions!
    let thresholdA = createErrorThreshold(precision: memoryPrecisions.A)
    let thresholdB = createErrorThreshold(precision: memoryPrecisions.B)
    let thresholdC = createErrorThreshold(precision: memoryPrecisions.C)
    errorThreshold = max(errorThreshold, thresholdA)
    errorThreshold = max(errorThreshold, thresholdB)
    errorThreshold = max(errorThreshold, thresholdC)
  }
  
  // Check the results.
  for m in 0..<problemSize {
    for n in 0..<problemSize {
      // Find the source row IDs.
      let leftRowID = (m + problemSize - 1) % problemSize
      let centerRowID = m
      let rightRowID = (m + problemSize + 1) % problemSize
      
      // Find the source values.
      let leftSource = B[leftRowID * problemSize + n]
      let centerSource = B[centerRowID * problemSize + n]
      let rightSource = B[rightRowID * problemSize + n]
      
      // Find the expected and actual values.
      let expected = leftSource - 2 * centerSource + rightSource
      let actual = C[m * problemSize + n]
      
      // Report the results.
      let error = (expected - actual).magnitude
      if error > errorThreshold {
        print("error: \(error) / ~1.000")
      }
    }
  }
}
