//
//  GEMMKernelDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import protocol Metal.MTLDevice

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
  
  /// Optional. The layout of elements in threadgroup memory.
  ///
  /// If not specified, the default value matches the actual block dimensions.
  ///
  /// This property can be used to avoid bank conflicts. For example, of one
  /// operand will have 16 FP32 elements per row, there is good chance of
  /// increased bank conflicts on M1. One may pad that threadgroup memory
  /// allocation to 20 FP32 elements per row.
  var leadingBlockDimensions: (A: UInt16, B: UInt16, C: UInt16)?
  
  var memoryPrecisions: (
    A: GEMMOperandPrecision, B: GEMMOperandPrecision, C: GEMMOperandPrecision)?
  
  /// Required. Whether async copies will improve performance during the
  /// matrix multiplication loop.
  ///
  /// The default value is `true`. Async copies improve performance on Apple7
  /// and Apple8, but harm performance on Apple9 and later. However, they are
  /// essential for correctness when reading from the edges of unaligned
  /// matrices. Setting the value to `false` means skipping async copies when
  /// doing so will not change the final result.
  var preferAsyncLoad: Bool = true
  
  /// Required. Whether async copies will improve performance when storing the
  /// accumulator to main memory.
  ///
  /// There is no default value that will reliably yield consistent performance.
  var preferAsyncStore: Bool?
  
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
  var transposeState: (A: Bool, B: Bool)?
}

struct GEMMKernelKey: Equatable, Hashable {
  var blockDimensions: SIMD3<UInt16>
  var leadingBlockDimensions: SIMD3<UInt16>
  var memoryPrecisions: SIMD3<UInt16>
  var preferAsyncLoad: UInt8
  var preferAsyncStore: UInt8
  var registerPrecisions: SIMD3<UInt16>
  var splits: SIMD2<UInt16>
  var transposeState: SIMD2<UInt8>
  
  init(copying source: GEMMKernelDescriptor) {
    blockDimensions = Self.createBlockDimensions(source.blockDimensions)
    leadingBlockDimensions = Self.createBlockDimensions(
      source.leadingBlockDimensions)
    memoryPrecisions = Self.createPrecisions(source.memoryPrecisions)
    preferAsyncLoad = Self.createBoolean(source.preferAsyncLoad)
    preferAsyncStore = Self.createBoolean(source.preferAsyncStore)
    registerPrecisions = Self.createPrecisions(source.registerPrecisions)
    
    splits = SIMD2(repeating: .max)
    if let (M, N) = source.splits {
      splits[0] = M
      splits[1] = N
    }
    transposeState = Self.createTransposeState(source.transposeState)
  }
  
  @_transparent // performance in -Ounchecked
  static func createBlockDimensions(
    _ input: (UInt16, UInt16, UInt16)?
  ) -> SIMD3<UInt16> {
    if let input {
      return SIMD3(input.0, input.1, input.2)
    } else {
      return SIMD3(repeating: .max)
    }
  }
  
  @_transparent // performance in -Ounchecked
  static func createBoolean(
    _ input: Bool?
  ) -> UInt8 {
    if let input {
      return input ? 1 : 0
    } else {
      return UInt8.max
    }
  }
  
  @_transparent // performance in -Ounchecked
  static func createPrecisions(
    _ input: (
      GEMMOperandPrecision, GEMMOperandPrecision, GEMMOperandPrecision)?
  ) -> SIMD3<UInt16> {
    if let input {
      return SIMD3(input.0.rawValue, input.1.rawValue, input.2.rawValue)
    } else {
      return SIMD3(repeating: .max)
    }
  }
  
  @_transparent // performance in -Ounchecked
  static func createTransposeState(
    _ input: (Bool, Bool)?
  ) -> SIMD2<UInt8> {
    if let input {
      return SIMD2(input.0 ? 1 : 0,
                   input.1 ? 1 : 0)
    } else {
      return SIMD2(repeating: .max)
    }
  }
}

extension GEMMKernelDescriptor: Hashable, Equatable {
  static func == (lhs: GEMMKernelDescriptor, rhs: GEMMKernelDescriptor) -> Bool {
    let lhsKey = GEMMKernelKey(copying: lhs)
    let rhsKey = GEMMKernelKey(copying: rhs)
    return lhsKey == rhsKey
  }
  
  func hash(into hasher: inout Hasher) {
    let key = GEMMKernelKey(copying: self)
    hasher.combine(key)
  }
}
