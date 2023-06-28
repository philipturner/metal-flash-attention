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
  var dataType: MTLDataType { get }
  var backend: TensorBackend { get }
  var pointer: UnsafeMutableRawPointer { get }
  
  // Number of elements. Cache the count because we use it a lot.
  // Shapes should never change.
  var count: Int { get }
  
  init(unsafeUninitializedShape shape: [Int], dataType: MTLDataType)
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

