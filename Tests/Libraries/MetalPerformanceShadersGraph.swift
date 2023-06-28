//
//  MetalPerformanceShadersGraph.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import MetalPerformanceShadersGraph

struct MPS_Backend: MetalBackend {
  typealias _AsyncResource = AsyncGraph
  typealias _GEMM = MPS_GEMM
}

class AsyncGraph: AsyncResource {
  private var _executable: MPSGraphExecutable?
  private var _semaphore: DispatchSemaphore
  private var _finished = false
  
  init(
    graph: MPSGraph,
    feeds: [MPSGraphTensor : MPSGraphShapedType],
    targetTensors: [MPSGraphTensor]
  ) {
    self._semaphore = DispatchSemaphore(value: 0)
    
    let compileDesc = MPSGraphCompilationDescriptor()
    compileDesc.optimizationLevel = .level1
    compileDesc.waitForCompilationCompletion = false
    compileDesc.compilationCompletionHandler = { executable, error in
      if let error {
        fatalError(error.localizedDescription)
      }
      self.finish(resource: executable)
    }
    
    let graphDevice = MetalContext.global.graphDevice
    graph.compile(
      with: graphDevice, feeds: feeds, targetTensors: targetTensors,
      targetOperations: nil, compilationDescriptor: compileDesc)
  }
  
  private func _blockingWait() {
    self._semaphore.wait()
    self._finished = true
  }
  
  func finish(resource: MPSGraphExecutable) {
    self._executable = resource
    self._semaphore.signal()
  }
  
  var resource: MPSGraphExecutable {
    if !_finished {
      _blockingWait()
    }
    return _executable!
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
