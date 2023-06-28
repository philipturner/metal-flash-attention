//
//  NumPy.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

struct Py_Backend: _TensorBackend {
  typealias _GEMM = Py_GEMM
}

final class Py_TensorBuffer: TensorBuffer {
  var shape: [Int]
  var dataType: MTLDataType
  var backend: TensorBackend { .numpy }
  
  var ndarray: PythonObject
  var pointer: UnsafeMutableRawPointer
  private(set) var count: Int
  
  init(unsafeUninitializedShape shape: [Int], dataType: MTLDataType) {
    self.shape = shape
    self.dataType = dataType
    self.count = shape.reduce(1, *)
    
    let np = PythonContext.global.np
    self.ndarray = np.empty(shape, dtype: dataType.numpy)
    
    let pyInt = x.ctypes.data
    let uinteger = UInt(Int(pyInt)!)
    self.pointer = UnsafeMutableRawPointer(bitPattern: uinteger)!
  }
}

protocol Py_Operation {
  typealias Backend = Py_Backend
  associatedtype Tensors
  
  func execute(tensors: Tensors)
}
