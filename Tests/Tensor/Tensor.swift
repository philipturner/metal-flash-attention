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
  
  private init(unsafeUninitializedShape shape: [Int], backend: TensorBackend) {
    buffer = backend.typeObject.init(
      unsafeUninitializedShape: shape, dataType: Element.mtlDataType)
  }
  
  init(
    shape: [Int],
    randomUniform distribution: Range<Float>,
    backend: TensorBackend
  ) {
    self.init(unsafeUninitializedShape: shape, backend: backend)
    RandomNumberGenerator.global.fillBuffer(
      buffer.pointer, range: distribution, elements: count,
      dataType: Element.mtlDataType)
  }
  
  init(zerosLike shape: [Int], backend: TensorBackend) {
    self.init(unsafeUninitializedShape: shape, backend: backend)
    memset(buffer.pointer, 0, buffer.allocatedSize)
  }
  
  init(copying other: Tensor, backend: TensorBackend) {
    self.init(unsafeUninitializedShape: other.shape, backend: backend)
    memcpy(buffer.pointer, other.buffer.pointer, buffer.allocatedSize)
  }
}

extension Tensor {
  // Inputs Q, K, V and outputs O in what would typically be called
  // Q^T, K^T, V^T, O^T. Performs softmax(Q^T^T K^T / sqrt(D)) * V^T^T.
  // func attention(...)
  
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
  
  // TODO: Ensure all other functions are mutating, because they overwrite the
  // contents of the destination tensor.
}
