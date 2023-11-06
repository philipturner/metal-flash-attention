//
//  NumPy.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit
import QuartzCore

final class Py_Backend: _TensorBackend {
  typealias _Attention = Py_Attention
  typealias _GEMM = Py_GEMM
  static let global = Py_Backend()
  
  var context: _ExecutionContext = _ExecutionContext()
  var usesCustomProfiler: Bool { false }
  
  var timerStart: Double = -1
  var timerEnd: Double = -1
  
  func markFirstCommand() {
    timerStart = CACurrentMediaTime()
  }
  
  func markLastCommand() {
    timerEnd = CACurrentMediaTime()
  }
  
  func synchronize() -> Double {
    return timerEnd - timerStart
  }
}

extension Py_Backend {
  func dispatch(
    parameters: Attention_Parameters, tensors: Attention_Tensors
  ) {
    let operation = _Attention(parameters: parameters)
    if context.ghost {
      // do nothing
    } else {
      operation.execute(tensors: tensors)
    }
  }
  
  func dispatch(
    parameters: GEMM_Parameters, tensors: GEMM_Tensors
  ) {
    let operation = _GEMM(parameters: parameters)
    if context.ghost {
      // do nothing
    } else {
      operation.execute(tensors: tensors)
    }
  }
}

final class Py_TensorBuffer: TensorBuffer {
  var shape: [Int]
  var dataType: MTLDataType
  var backend: TensorBackend { .numpy }
  
  var ndarray: PythonObject
  var pointer: UnsafeMutableRawPointer
  private(set) var count: Int
  
  init(unsafeUninitializedShape shape: [Int], dataType: MTLDataType) {
    if _ExecutionContext.logTensorCreation {
      print("NumPy tensor created: \(shape)")
    }
    self.shape = shape
    self.dataType = dataType
    self.count = shape.reduce(1, *)
    
    let np = PythonContext.global.np
    self.ndarray = np.empty(shape, dtype: dataType.numpy)
    
    let pyInt = ndarray.ctypes.data
    let uinteger = UInt(Int(pyInt)!)
    self.pointer = UnsafeMutableRawPointer(bitPattern: uinteger)!
  }
  
  func release() {
    // ¯\_(ツ)_/¯
  }
}

protocol Py_Operation {
  typealias Backend = Py_Backend
  associatedtype Tensors
  
  func execute(tensors: Tensors)
}
