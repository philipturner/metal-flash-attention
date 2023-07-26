//
//  MetalFlashAttention.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import AppleGPUInfo
import Metal

final class MFA_Backend: MetalBackend {
  typealias _Attention = MFA_Attention
  typealias _GEMM = MFA_GEMM
  
  typealias Resource = AsyncPipeline
  static let global = MFA_Backend()
  static let dynamicBatch: Bool = true
  
  var context: _ExecutionContext = _ExecutionContext()
  var usesCustomProfiler: Bool { true }
  var encoder: MTLComputeCommandEncoder { _encoder! }
  
  var cache: OperationCache<MFA_Backend> = .init()
  var commandBuffer: MTLCommandBuffer?
  var _encoder: MTLComputeCommandEncoder?
  var gpuTime: Double = -1 // set by the command buffer's completion handler
  
  func markFirstCommand() {
    precondition(commandBuffer == nil)
    precondition(_encoder == nil)
    if !context.ghost {
      let ctx = MetalContext.global
      commandBuffer = ctx.commandQueue.makeCommandBuffer()!
      _encoder = commandBuffer!.makeComputeCommandEncoder()!
    }
  }
  
  func markLastCommand() {
    if !context.ghost {
      _encoder!.endEncoding()
      commandBuffer!.addCompletedHandler { commandBuffer in
        self.gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
      }
      commandBuffer!.commit()
    }
  }
  
  func synchronize() -> Double {
    if context.ghost {
      return 0
    } else {
      commandBuffer!.waitUntilCompleted()
      commandBuffer = nil
      _encoder = nil
      return gpuTime
    }
  }
}

class AsyncPipeline: AsyncResource {
  private var _pipelines: [MTLComputePipelineState?]
  private var _semaphore: DispatchSemaphore
  private var _finished = false
  
  // Pre-compute some of the dispatch metadata to speed up encoding. Some
  // functions will ignore the metadata or overwrite some of its values.
  var flags: UInt32
  
  // Device memory lengths are for an entire matrix plane, not a single TG.
  // Locks are not included in device memory length; every threadgroup gets one
  // lock for kernels that opt in.
  var deviceMemoryLengths: [UInt64]
  var threadgroupMemoryLengths: [UInt16]
  var gridSizes: [MTLSize]
  var groupSizes: [MTLSize]
  
  init(
    functions: [MTLFunction],
    flags: UInt32,
    deviceMemoryLengths: [UInt64],
    threadgroupMemoryLengths: [UInt16],
    gridSizes: [MTLSize],
    groupSizes: [MTLSize]
  ) {
    self._pipelines = Array(repeating: nil, count: functions.count)
    self._semaphore = DispatchSemaphore(value: 0)
    
    self.flags = flags
    self.deviceMemoryLengths = deviceMemoryLengths
    self.threadgroupMemoryLengths = threadgroupMemoryLengths
    self.gridSizes = gridSizes
    self.groupSizes = groupSizes
    
    let device = MetalContext.global.device
    for i in functions.indices {
      let function = functions[i]
      device.makeComputePipelineState(function: function) { pipeline, error in
        if let error {
          fatalError(error.localizedDescription)
        }
        self.finish(resource: pipeline!, index: i)
      }
    }
  }
  
  private func _blockingWait() {
    for _ in 0..<_pipelines.count {
      self._semaphore.wait()
    }
    self._finished = true
  }
  
  func finish(resource: MTLComputePipelineState, index: Int) {
    self._pipelines[index] = resource
    self._semaphore.signal()
  }
  
  func resource(index: Int) -> MTLComputePipelineState {
    if !_finished {
      _blockingWait()
    }
    return _pipelines[index]!
  }
}

final class MFA_TensorBuffer: TensorBuffer {
  var shape: [Int]
  var dataType: MTLDataType
  var backend: TensorBackend { .mfa }
  
  var buffer: MTLBuffer
  var pointer: UnsafeMutableRawPointer { buffer.contents() }
  private(set) var count: Int
  
  init(unsafeUninitializedShape shape: [Int], dataType: MTLDataType) {
    if _ExecutionContext.logTensorCreation {
      print("MFA tensor created: \(shape)")
    }
    self.shape = shape
    self.dataType = dataType
    self.count = shape.reduce(1, *)
    
    let bufferSize = count * dataType.size
    let device = MetalContext.global.device
    self.buffer = device.makeBuffer(length: bufferSize)!
  }
}

protocol _Has_MFA_Backend {
  typealias Backend = MFA_Backend
}

protocol MFA_Operation: MetalOperation, _Has_MFA_Backend {
  // Shader configuration that the main script can modify, to measure the
  // performance difference.
  static var functionConstants: [String: MTLConvertible] { get }
}
