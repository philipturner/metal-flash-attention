//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.

// Workspace for drafting the auto-parsing code.
func executeScript() {
  let sequenceDimension: Int = 1024
  let headDimension: Int = 97
  
  var attentionDesc = AttentionDescriptor()
  attentionDesc.lowPrecisionInputs = true
  attentionDesc.lowPrecisionIntermediates = true
  attentionDesc.matrixDimensions = (
    R: UInt32(sequenceDimension),
    C: UInt32(sequenceDimension),
    D: UInt16(headDimension))
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  
  /*
   
   maximum head dimension: 16
   block dimension (R): 32
   block dimension (C): 32
   block dimension (D): 16
   cached operands: [V, dV, dK]
   */
  let kernelDesc = attentionDesc.kernelDescriptor(type: .forward)
  
  // Display the selected parameters.
  print()
  print("block dimension (R):", kernelDesc.blockDimensions?.parallelization)
  print("block dimension (C):", kernelDesc.blockDimensions?.traversal)
  print("block dimension (D):", kernelDesc.blockDimensions?.head)
  print("cached operands:", kernelDesc.cacheState)
  print()
  print("memory precisions:")
  for (key, value) in kernelDesc.memoryPrecisions {
    print("- \(key) : \(value)")
  }
  print("register precisions:")
  for (key, value) in kernelDesc.registerPrecisions {
    print("- \(key) : \(value)")
  }
  print()
}
