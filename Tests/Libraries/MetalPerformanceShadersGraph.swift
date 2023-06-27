//
//  MetalPerformanceShadersGraph.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import MetalPerformanceShadersGraph

// Load MPSGraph, but use it as an eager execution engine like many real-world
// libraries do. This is not the way MPSGraph is intended to be used, and incurs
// a heavy sequential throughput bottleneck.

class MPS_GEMM: Operation {
  
}

class MPS_Tensor/*: Tensor */ {
  var buffer: MTLBuffer
  var tensorData: MPSGraphTensorData
  
  init() {
    fatalError()
  }
}
