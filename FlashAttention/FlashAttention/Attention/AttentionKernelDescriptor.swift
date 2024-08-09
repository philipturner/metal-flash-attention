//
//  AttentionKernelDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

/// The three kernels of the FlashAttention algorithm for devices without
/// hardware acceleration for floating-point atomics.
///
/// For the forward pass, enter `.forward(false)` if you are only running
/// inference. Never bind the "L terms" (softmax log-sum-exp)  when encoding the
/// forward command. If you will compute the gradient in another pass (e.g.
/// training), enter `.forward(true)`. Always bind the "L terms" when encoding
/// the forward command. The rule regarding presence/absence of buffer binding
/// is a way to detect accidental computation of L terms during forward pass.
/// Although small for some problem configurations, such accidental computation
/// is a provable performance regression of nonzero magnitude. Omitting the
/// L buffer binding will cause a Metal API error when an offending forward
/// kernel is encoded.
///
/// Originally, there was intended to be two pathways for backward. One path
/// computed everything online without materializing the attention matrix:
/// `.backwardQuery(true)`, `.backwardKeyValue(true)`. The other skipped the
/// computation of dQ and dK, deferring that to two GEMM calls after the
/// attention submatrix was stored in RAM:
/// `.backwardQuery(false)`, `.backwardKeyValue(false)`. The latter, faster
/// pathway was not implemented due to time constraints. Therefore, the
/// associated value on the dQ and dK/dV cases has now been removed.
enum AttentionKernelType {
  /// Forward attention, computing O and optionally L[i].
  ///
  /// Variants:
  /// - `false`: compute only O
  /// - `true`: compute both O and L[i]
  case forward(Bool)
  
  /// Backward attention, computing D[i] and dQ.
  ///
  /// Depends on: L[i]
  case backwardQuery
  
  /// Backward attention, computing dK and dV.
  ///
  /// Depends on: L[i], D[i]
  case backwardKeyValue
}

struct AttentionKernelDescriptor {
  var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)?
  
  /// Whether each operand is cached in registers.
  var cacheState: [AttentionOperand: Bool] = [:]
  
  var headDimension: UInt16?
  
  /// Reads with a one-to-one mapping to threads (like GEMM store) and writes.
  var preferAsyncCache: Bool = true
  
  /// Reads that are shared among threads (like GEMM load).
  var preferAsyncLoad: Bool = true
  
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
