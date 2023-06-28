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

final class MPS_TensorBuffer: TensorBuffer {
  var shape: [Int]
  var nsShape: [NSNumber] { tensorData.shape }
  var dataType: MTLDataType
  var backend: TensorBackend { .mps }
  
  var buffer: MTLBuffer
  var tensorData: MPSGraphTensorData
  var pointer: UnsafeMutableRawPointer { buffer.contents() }
  private(set) var count: Int
  
  init(unsafeUninitializedShape shape: [Int], dataType: MTLDataType) {
    self.shape = shape
    self.dataType = dataType
    self.count = shape.reduce(1, *)
    
    let bufferSize = count * dataType.size
    let device = MetalContext.global.device
    self.buffer = device.makeBuffer(length: bufferSize)!
    
    let nsShape = shape.map(NSNumber.init)
    self.tensorData = MPSGraphTensorData(
      buffer, shape: nsShape, dataType: dataType.mps)
  }
}

protocol MPS_Operation: Operation {
  
}
