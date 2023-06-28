//
//  TensorBackend.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal
import PythonKit

// Stores state information for the backend and contains references to type
// objects for dispatching operations.
protocol _TensorBackend {
  // TODO: Properties for whether you're currently doing ghost executions or
  // running a profiling timer.
  
  // associatedtype _Attention: Attention
  // associatedtype _Convolution: Convolution
  associatedtype _GEMM: GEMM
  // associatedtype _Normalization: Normalization
  
  // TODO: Function for the backend to dispatch an attention, dispatch a GEMM
  
  // static var global: Self { get }
  // func markFirstCommand()
  // func markLastCommand()
  // func synchronize()
}

enum TensorBackend {
  case mfa
  case mps
  case numpy
  
  var typeObject: TensorBuffer.Type {
    fatalError()
  }
  
  // Perform a ghost execution, just to async create the graphs or pipelines.
  func withGhostExecution<R>(_ closure: () throws -> R) rethrows -> R {
    // TODO: Signal a flag in Metal/PythonContext for ghost execution. If a
    // pipeline was not previously dispatched during a ghost execution, throw an
    // error. We do not want to accidentally trigger bad app design, which
    // eagerly stalls on each resource generated.
    fatalError()
  }
  
  // Wrapper around `withGhostExecution` for de-duplicating code.
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
    // self.typeObject.markFirstCommand()
  }
  
  // Tells the backend to dispatch the buffered up commands. This will signal
  // the profiling timer to stop.
  func markLastCommand() {
    // self.typeObject.markLastCommand()
  }
  
  // Synchronizes with the backend and reports the execution time.
  func synchronize() -> Double {
    // self.typeObject.synchronize()
    fatalError()
  }
}
