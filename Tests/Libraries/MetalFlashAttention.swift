//
//  MetalFlashAttention.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import AppleGPUInfo
import Metal

struct MFA_Backend: MetalBackend {
  typealias _AsyncResource = AsyncPipeline
  typealias _GEMM = MFA_GEMM
}

class AsyncPipeline: AsyncResource {
  private var _pipeline: MTLComputePipelineState?
  private var _semaphore: DispatchSemaphore
  private var _finished = false
  
  init(function: MTLFunction) {
    self._semaphore = DispatchSemaphore(value: 0)
    
    let device = MetalContext.global.device
    device.makeComputePipelineState(function: function) { pipeline, error in
      if let error {
        fatalError(error.localizedDescription)
      }
      self.finish(resource: pipeline!)
    }
  }
  
  private func _blockingWait() {
    self._semaphore.wait()
    self._finished = true
  }
  
  func finish(resource: MTLComputePipelineState) {
    self._pipeline = resource
    self._semaphore.signal()
  }
  
  var resource: MTLComputePipelineState {
    if !_finished {
      _blockingWait()
    }
    return _pipeline!
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
    self.shape = shape
    self.dataType = dataType
    self.count = shape.reduce(1, *)
    
    let bufferSize = count * dataType.size
    let device = MetalContext.global.device
    self.buffer = device.makeBuffer(length: bufferSize)!
  }
}

protocol MFA_Operation: Operation {
  // Shader configuration that the main script can modify, to measure the
  // performance difference.
  static var functionConstants: [String: MTLConvertible] { get }
  
  // Make an async pipeline if the cache doesn't already contain it.
  func makeAsyncPipeline() -> AsyncPipeline
}


