//
//  TensorBuffer.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

protocol TensorBuffer {
  var shape: [Int] { get }
  var pointer: UnsafeMutableRawPointer { get }
  var dataType: MTLDataType { get }
  var backend: TensorBackend { get }
  
  // Number of elements.
  //
  // Cache the count because we use it a lot. Shapes should never change.
  var count: Int { get } // shape.reduce(1, *)
  
  // A simple set of initializers; none deal with pointers directly.
  init(randomUniform distribution: Range<Float>)
  init(zerosLike shape: [Int])
  init(copying other: TensorBuffer)
}

extension TensorBuffer {
  // Number of bytes in memory.
  var allocatedSize: Int {
    self.count * dataType.size
  }
  
  // The conversion process makes multiple copies, but at least it's safe.
  func numpy() -> PythonObject {
    let ctx = PythonContext.global
    let data = Data(bytes: pointer, count: allocatedSize)
    let bytes = PythonBytes(data)
    return ctx.np.frombuffer(bytes, dtype: dataType.numpy)
  }
}

extension TensorBuffer {
  // TODO: Provide functionality for checking this inside the GEMM type object.
  func dispatchCompatible(_ other: TensorBuffer) -> Bool {
    // Data type must be the same because mixed precision not supported yet.
    return self.dataType == other.dataType &&
           self.backend == other.backend
  }
  
  func matmul(_ a: TensorBuffer, _ b: TensorBuffer, _ c: TensorBuffer) {
    precondition(a.dispatchCompatible(b))
    precondition(a.dispatchCompatible(c))
  }
}
