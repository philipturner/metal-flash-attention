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

struct MPS_Backend: MetalBackend {
  typealias _AsyncResource = AsyncGraph
  typealias _GEMM = MPS_GEMM
}

class AsyncGraph: AsyncResource {
  typealias Resource = MPSGraph
  
  func finish(resource: MPSGraph) {
    fatalError("Not implemented.")
  }
  
  var resource: MPSGraph {
    fatalError("Not implemented.")
  }
}

class MPS_TensorBuffer/*: Tensor */ {
  var buffer: MTLBuffer
  var tensorData: MPSGraphTensorData
  
  init() {
    fatalError()
  }
}

class MPS_GEMM: GEMM {
  var parameters: GEMM_Parameters
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
}

