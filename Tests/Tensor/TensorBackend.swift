//
//  TensorBackend.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit
import QuartzCore

// Stores state information for the backend and contains references to type
// objects for dispatching operations.
protocol _TensorBackend: AnyObject {
  // associatedtype _Attention: Attention
  // associatedtype _Convolution: Convolution
  associatedtype _GEMM: GEMM
  // associatedtype _Normalization: Normalization
  
  static var global: Self { get }
  
  var context: _ExecutionContext { get set }
  
  var usesCustomProfiler: Bool { get }
  
  func markFirstCommand()
  
  func markLastCommand()
  
  func synchronize() -> Double
  
  func dispatch(parameters: GEMM_Parameters, tensors: GEMM_Tensors)
}

struct _ExecutionContext {
  static let logTensorCreation: Bool = false
  
  // Code besides the backend will modify these properties.
  var ghost: Bool = false
  var executionWillStart: Bool = false
  var executionWillEnd: Bool = false
  
  static var defaultBackend: TensorBackend = .numpy
  
  // First makes a ghost pass through the expression, then actually executes it.
  static func executeExpression<R>(_ closure: () throws -> R) rethrows -> R {
    _ = try defaultBackend.withGhostExecution(closure)
    return try closure()
  }
  
  @discardableResult
  static func profileCommands(_ closure: () throws -> Void) rethrows -> Double {
    return try executeExpression {
      defaultBackend.markFirstCommand()
      try closure()
      defaultBackend.markLastCommand()
      return defaultBackend.synchronize()
    }
  }
  
  @discardableResult
  static func withDefaultBackend<R>(
    _ backend: TensorBackend,
    _ closure: () throws -> R
  ) rethrows -> R {
    let previous = _ExecutionContext.defaultBackend
    _ExecutionContext.defaultBackend = backend
    let output = try closure()
    _ExecutionContext.defaultBackend = previous
    return output
  }
}

enum TensorBackend {
  case mfa
  case mps
  case numpy
  
  static var `default`: TensorBackend {
    _ExecutionContext.defaultBackend
  }
  
  var backendObject: any _TensorBackend {
    switch self {
    case .mfa: return MFA_Backend.global
    case .mps: return MPS_Backend.global
    case .numpy: return Py_Backend.global
    }
  }
  
  var bufferObject: TensorBuffer.Type {
    switch self {
    case .mfa: return MFA_TensorBuffer.self
    case .mps: return MPS_TensorBuffer.self
    case .numpy: return Py_TensorBuffer.self
    }
  }
  
  @inline(__always)
  func withContextParameter<R>(
    setOnCall: Bool,
    unsetOnReturn: Bool,
    _ parameter1: WritableKeyPath<_ExecutionContext, Bool>,
    _ parameter2: WritableKeyPath<_ExecutionContext, Bool>? = nil,
    _ closure: (any _TensorBackend) throws -> R
  ) rethrows -> R {
    let backend = self.backendObject
    if setOnCall {
      var newContext = backend.context
      precondition(newContext[keyPath: parameter1] == false)
      newContext[keyPath: parameter1] = true
      if let parameter2 {
        precondition(newContext[keyPath: parameter2] == false)
        newContext[keyPath: parameter2] = true
      }
      backend.context = newContext
    }
    
    func cleanup() {
      if unsetOnReturn {
        var newContext = backend.context
        precondition(newContext[keyPath: parameter1] == true)
        newContext[keyPath: parameter1] = false
        if let parameter2 {
          precondition(newContext[keyPath: parameter2] == true)
          newContext[keyPath: parameter2] = false
        }
        backend.context = newContext
      }
    }
    
    var output: R
    do {
      output = try closure(backend)
    } catch {
      cleanup()
      throw error
    }
    cleanup()
    return output
  }
  
  // Perform a ghost execution, just to async create the graphs or pipelines.
  func withGhostExecution<R>(_ closure: () throws -> R) rethrows -> R {
    try withContextParameter(
      setOnCall: true, unsetOnReturn: true,
      \.ghost
    ) { _ in
      try closure()
    }
  }
  
  // Wrapper around `withGhostExecution` for de-duplicating code.
  @inline(__always)
  func withExecution<R>(ghost: Bool, _ closure: () throws -> R) rethrows -> R {
    if ghost {
      try withGhostExecution(closure)
    } else {
      try closure()
    }
  }
  
  // Some backends do not eagerly dispatch the commands, so you must manually
  // commit them. This method tells the internal profiling timer to start.
  func markFirstCommand() {
    withContextParameter(
      setOnCall: true, unsetOnReturn: false,
      \.executionWillStart
    ) {
      $0.markFirstCommand()
    }
  }
  
  // Tells the backend to dispatch the buffered up commands. This will signal
  // the profiling timer to stop.
  func markLastCommand() {
    withContextParameter(
      setOnCall: true, unsetOnReturn: false,
      \.executionWillEnd
    ) {
      $0.markLastCommand()
    }
  }
  
  // Synchronizes with the backend and reports the execution time. You must call
  // this before calling `markFirstCommand` again.
  func synchronize() -> Double {
    return withContextParameter(
      setOnCall: false, unsetOnReturn: true,
      \.executionWillStart, \.executionWillEnd
    ) {
      $0.synchronize()
    }
  }
  
  func dispatch(parameters: GEMM_Parameters, tensors: GEMM_Tensors) {
    self.backendObject.dispatch(parameters: parameters, tensors: tensors)
  }
}
