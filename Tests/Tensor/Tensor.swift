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
  func dispatchCompatible(_ other: Tensor<Element>) -> Bool {
    return self.buffer.backend == other.buffer.backend
  }
  
  // Inputs Q, K, V and outputs O in what would typically be called
  // Q^T, K^T, V^T, O^T. Performs softmax(Q^T^T K^T / sqrt(D)) * V^T^T.
  //
  // Queries, keys and values are already stored as one mega-tensor, of shape
  // 3 x H x D x N.
  mutating func attention(
    queriesKeysValues: Tensor<Element>
  ) {
    assert(self.dispatchCompatible(queriesKeysValues))
    
    let inShape = queriesKeysValues.shape
    let outShape = self.shape
    assert(inShape.count >= 4)
    assert(outShape.count >= 4)
    
    let li = inShape.endIndex - 1
    let lo = outShape.endIndex - 1
    assert(inShape[li - 0] == outShape[lo - 0]) // N
    assert(inShape[li - 1] == outShape[lo - 1]) // D
    assert(inShape[li - 2] == outShape[lo - 2]) // H
    assert(inShape[li - 3] == 3)
    assert(inShape[lo - 3] == 1)
    
    if inShape.count >= 5 || outShape.count >= 5 {
      assert(inShape.count == 5)
      assert(outShape.count == 5)
      assert(inShape[li - 4] == outShape[lo - 4])
    }
  }
  
  // Sets this tensor's data to the product of the inputs.
  mutating func matmul(
    _ a: Tensor<Element>, _ b: Tensor<Element>,
    transposeA: Bool = false, transposeB: Bool = false,
    alpha: Float = 1.0, beta: Float = 0.0
  ) {
    assert(self.dispatchCompatible(a))
    assert(self.dispatchCompatible(b))
    assert(transposeA == false)
    assert(transposeB == false)
    assert(alpha == 1.0)
    assert(beta == 1.0)
  }
  
  // TODO: Ensure all other functions are mutating, because they overwrite the
  // contents of the destination tensor.
}
