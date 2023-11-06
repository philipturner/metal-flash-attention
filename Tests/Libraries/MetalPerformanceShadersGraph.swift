//
//  MetalPerformanceShadersGraph.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import MetalPerformanceShadersGraph
import QuartzCore

final class MPS_Backend: MetalBackend {
  typealias _Attention = MPS_Attention
  typealias _GEMM = MPS_GEMM
  
  typealias Resource = AsyncGraph
  static let global = MPS_Backend()
  static let dynamicBatch: Bool = false
  
  var context: _ExecutionContext = _ExecutionContext()
  var usesCustomProfiler: Bool { false }
  var encoder: MPSCommandBuffer { commandBuffer! }
  
  var cache: OperationCache<MPS_Backend> = .init()
  var commandBuffer: MPSCommandBuffer?
  var timerStart: Double = -1
  var timerEnd: Double = -1
  
  func markFirstCommand() {
    precondition(commandBuffer == nil)
    if !context.ghost {
      let ctx = MetalContext.global
      commandBuffer = MPSCommandBuffer(from: ctx.commandQueue)
      timerStart = CACurrentMediaTime()
    }
  }
  
  func markLastCommand() {
    if !context.ghost {
      commandBuffer!.commit()
    }
  }
  
  func synchronize() -> Double {
    if context.ghost {
      return 0
    } else {
      // MPS has to end the timer here (instead of during `markLastCommand()`)
      // because it must synchronize beforehand.
      commandBuffer!.waitUntilCompleted()
      commandBuffer = nil
      timerEnd = CACurrentMediaTime()
      return timerEnd - timerStart
    }
  }
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
    compileDesc.optimizationLevel = .level0
    compileDesc.waitForCompilationCompletion = false
    compileDesc.compilationCompletionHandler = { executable, error in
      if let error {
        fatalError(error.localizedDescription)
      }
      self.finish(resource: executable, index: 0)
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
  
  func finish(resource: MPSGraphExecutable, index: Int) {
    self._executable = resource
    self._semaphore.signal()
  }
  
  func resource(index: Int) -> MPSGraphExecutable {
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
    if _ExecutionContext.logTensorCreation {
      print("MPS tensor created: \(shape)")
    }
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
  
  func release() {
    self.buffer.setPurgeableState(MTLPurgeableState.empty)
  }
}

protocol _Has_MPS_Backend {
  typealias Backend = MPS_Backend
}

protocol MPS_Operation: MetalOperation, _Has_MPS_Backend {
  
}
