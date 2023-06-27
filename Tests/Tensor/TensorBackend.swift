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
    
  }
  
  // Tells the backend that the next command must be synchronous. This may be
  // called immediately after `markFirstCommand` to dispatch only one command.
  func markPenultimateCommand() {
    
  }
  
  // Returns the total execution time and resets the profiling timer.
  func blockingExecute() -> Double {
    // TODO: Assert that one, and only one, command was dispatched between the
    // penultimate command and now.
    fatalError("Not implemented.")
  }
}
