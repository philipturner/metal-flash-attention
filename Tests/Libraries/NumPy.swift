//
//  NumPy.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

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

// class Py_Attention

// class Py_Convolution

// class Py_GEMM

// class Py_Normalization

// class Py_Tensor
