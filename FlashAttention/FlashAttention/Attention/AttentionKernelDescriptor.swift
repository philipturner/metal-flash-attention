//
//  AttentionKernelDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//


struct AttentionKernelDescriptor {
  var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)?
  
  /// Whether each operand is cached in registers.
  var cacheState: [AttentionOperand: Bool] = [:]
  
  /// Required. The problem size along the head dimension.
  var headDimension: UInt16?
  
  var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
  
  /// Reads with a one-to-one mapping to threads (like GEMM store) and writes.
  var preferAsyncCache: Bool?
  
  /// Reads that are shared among threads (like GEMM load).
  var preferAsyncLoad: Bool?
  
  var registerPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
  
  /// Whether each operand is transposed in RAM.
  ///
  /// If the layout is row-major, where a row spans D contiguous elements in
  /// memory, enter `false`. If the layout is column-major, where a row spans
  /// D widely separated elements in memory, enter `true`.
  ///
  /// The transpose state of a derivative (e.g. dQ for Q) must match the
  /// corresponding input from the forward pass.
  ///
  /// > NOTE: To implement multi-head attention, clients may need to modify
  /// the stride of matrix elements in memory. If and only if the transpose
  /// state is `false`, change the stride from `D` to `D * H`. Ensure the
  /// value of H is known at compile time, so the product `D * H` can be
  /// embedded into the GPU assembly code.
  var transposeState: [AttentionOperand: Bool] = [:]
  
  var type: AttentionKernelType?
}
