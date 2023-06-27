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
  
  init(device: MTLDevice, function: MTLFunction) {
    self._semaphore = DispatchSemaphore(value: 0)
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

class MFA_TensorBuffer/*: TensorBuffer*/ {
  
}

protocol MFA_Operation {
  // Shader configuration that the main script can modify, to measure the
  // performance difference.
  static var functionConstants: [String: MTLConvertible] { get }
  
  // Make an async pipeline if the cache doesn't already contain it.
  func makeAsyncPipeline() -> AsyncPipeline
}

class MFA_GEMM: GEMM, MFA_Operation {
  var parameters: GEMM_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    "M_simd": UInt16(16), // 24
    "N_simd": UInt16(16), // 24
    "K_simd": UInt16(32), // 24-32
    "M_splits": UInt16(2),
    "N_splits": UInt16(2),
    "K_splits": UInt16(1),
  ]
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncPipeline() -> AsyncPipeline {
    fatalError()
  }
}
