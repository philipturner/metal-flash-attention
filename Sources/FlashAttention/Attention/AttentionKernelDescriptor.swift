//
//  AttentionKernelDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

public struct AttentionKernelDescriptor {
  public var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)?
  
  /// Whether each operand is cached in registers.
  public var cacheState: [AttentionOperand: Bool] = [:]
  
  /// Required. The problem size along the head dimension.
  public var headDimension: UInt16?
  
  public var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
  
  /// Reads with a one-to-one mapping to threads (like GEMM store) and writes.
  public var preferAsyncCache: Bool?
  
  /// Reads that are shared among threads (like GEMM load).
  public var preferAsyncLoad: Bool?
  
  public var registerPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
  
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
  public var transposeState: [AttentionOperand: Bool] = [:]
  
  public var type: AttentionKernelType?
  
  public init() {
    
  }
}
