//
//  Tensor.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

protocol Tensor {
  var shape: [Int] { get }
  var pointer: UnsafeMutableRawPointer { get }
  var dataType: MTLDataType { get }
  
  // A simple set of initializers; none deal with pointers directly.
  init(randomUniform distribution: Range<Float>)
  init(zerosLike shape: [Int])
  init(copying other: Tensor)
}

extension Tensor {
  // Number of elements.
  var count: Int {
    shape.reduce(1, *)
  }
  
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
