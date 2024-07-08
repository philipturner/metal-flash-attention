//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

enum AttentionKernelType {
  /// Forward attention, computing O and L[i].
  ///
  /// Variants:
  /// - `false`: compute O
  /// - `true`: compute O and L[i]
  case forward(Bool)
  
  /// Backward attention, computing D[i] and dQ.
  ///
  /// Variants:
  /// - `false`: compute D[i]
  /// - `true`: compute D[i] and dQ
  ///
  /// Depends on: L[i]
  case backwardQuery(Bool)
  
  /// Backward attention, computing dK and dV.
  ///
  /// Variants:
  /// - `false`: compute dV, store the intermediate dS
  /// - `true`: compute dV and dK
  ///
  /// Depends on: L[i], D[i]
  case backwardKeyValue(Bool)
}

struct AttentionDescriptor {
  /// The dimensions of the input and output matrices.
  /// - Parameters R: Number of rows in the attention matrix.
  /// - Parameters C: Number of columns in the attention matrix.
  /// - Parameters D: Size of the head.
  var matrixDimensions: (R: UInt32, C: UInt32, D: UInt16)?
  
  /// Currently ignored.
  var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)?
  
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
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?
  
  var type: AttentionKernelType?
}
