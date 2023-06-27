//
//  MetalFlashAttention.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import AppleGPUInfo
import Metal

// TODO: Utility to asynchronously generate compute pipeline states, only
// waiting on their completion when encoding the GPU command.

class AsyncPipeline {
  private var _pipeline: MTLComputePipelineState?
  private var _semaphore: DispatchSemaphore
  private var _finished = false
  
  init(device: MTLDevice, function: MTLFunction) {
    self._semaphore = DispatchSemaphore(value: 0)
    device.makeComputePipelineState(function: function) { pipeline, error in
      if let error {
        fatalError(error.localizedDescription)
      }
      self.finish(pipeline: pipeline!)
    }
  }
  
  private func _blockingWait() {
    self._semaphore.wait()
    self._finished = true
  }
  
  func finish(pipeline: MTLComputePipelineState) {
    self._pipeline = pipeline
    self._semaphore.signal()
  }
  
  var pipeline: MTLComputePipelineState {
    if !_finished {
      _blockingWait()
    }
    return _pipeline!
  }
}

class MFA_GEMM: Operation {
  
}
