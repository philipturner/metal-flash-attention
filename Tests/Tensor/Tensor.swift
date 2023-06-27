//
//  Tensor.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

struct Tensor<Element: TensorElement> {
  var buffer: TensorBuffer
  var shape: [Int] { buffer.shape }
  var count: Int { buffer.count }
  
  init(randomUniform distribution: Range<Float>, backend: TensorBackend) {
    buffer = backend.typeObject.init(randomUniform: distribution)
  }
  init(zerosLike shape: [Int], backend: TensorBackend) {
    buffer = backend.typeObject.init(zerosLike: shape)
  }
  init(copying other: Tensor, backend: TensorBackend) {
    buffer = backend.typeObject.init(copying: other.buffer)
  }
}

extension Tensor {
  // Sets this tensor's data to the product of the inputs.
  mutating func matmul(
    _ a: Tensor<Element>, _ b: Tensor<Element>,
    transposeA: Bool = false, transposeB: Bool = false,
    alpha: Float = 1.0, beta: Float = 0.0
  ) {
    precondition(transposeA == false)
    precondition(transposeB == false)
    precondition(alpha == 1.0)
    precondition(beta == 1.0)
  }
  
  // Inputs Q, K, V and outputs O in what would typically be called
  // Q^T, K^T, V^T, O^T. Performs softmax(Q^T^T K^T / sqrt(D)) * V^T^T.
  
  // TODO: Ensure all other functions are mutating, because they overwrite the
  // contents of the destination tensor.
}
