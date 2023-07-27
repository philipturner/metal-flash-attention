//
//  Metal.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import AppleGPUInfo
import MetalPerformanceShadersGraph

struct MetalContext {
  static let global = MetalContext()
  
  var device: MTLDevice
  var infoDevice: GPUInfoDevice
  var graphDevice: MPSGraphDevice
  
  var library: MTLLibrary
  var commandQueue: MTLCommandQueue
  
  init() {
    self.device = MTLCopyAllDevices().first!
    device.shouldMaximizeConcurrentCompilation = true
    self.infoDevice = try! GPUInfoDevice()
    self.graphDevice = MPSGraphDevice(mtlDevice: device)
    
    var libraryURL = Bundle.main.resourceURL!
    libraryURL.append(component: "lib")
    libraryURL.append(component: "libMetalFlashAttention.metallib")
    self.library = try! device.makeLibrary(URL: libraryURL)
    self.commandQueue = device.makeCommandQueue()!
  }
}

protocol MetalBackend: _TensorBackend
where
_Attention: MetalOperation, _Attention.Backend == Self,
_GEMM: MetalOperation, _GEMM.Backend == Self
{
  associatedtype Encoder
  associatedtype Resource: AsyncResource
  
  var encoder: Encoder { get }
  
  var cache: OperationCache<Self> { get }
  
  static var dynamicBatch: Bool { get }
}

extension MetalBackend {
  func dispatch(
    parameters: Attention_Parameters, tensors: Attention_Tensors
  ) {
    let operation = _Attention(parameters: parameters)
    if context.ghost {
      cache.cache(operation: operation)
    } else {
      cache.encode(operation: operation, encoder: encoder, tensors: tensors)
    }
  }
  
  func dispatch(
    parameters: GEMM_Parameters, tensors: GEMM_Tensors
  ) {
    let operation = _GEMM(parameters: parameters)
    if context.ghost {
      cache.cache(operation: operation)
    } else {
      cache.encode(operation: operation, encoder: encoder, tensors: tensors)
    }
  }
}

protocol AsyncResource {
  associatedtype Resource
  
  // A background thread calls this to write the finished resoure.
  func finish(resource: Resource, index: Int)
  
  // Lazily blocks until the background thread finishes.
  func resource(index: Int) -> Resource
}

protocol MetalOperation {
  associatedtype Backend: MetalBackend
  typealias Encoder = Backend.Encoder
  associatedtype Tensors
  
  // Make an async resource if the cache doesn't already contain it.
  func makeAsyncResource() -> Backend.Resource
  
  // Never called during ghost execution.
  func encode(encoder: Encoder, tensors: Tensors, resource: Backend.Resource)
}

class OperationCache<Backend: MetalBackend> {
  fileprivate var attention: [
    Attention_Parameters: Backend.Resource] = [:]
  fileprivate var gemm: [
    GEMM_Parameters: Backend.Resource] = [:]
  
  var scratchBuffer: MTLBuffer?
  
  var partialsBuffer: MTLBuffer?
  
  var locksBuffer: MTLBuffer?
  
  func requestScratchBuffer(size: Int) -> MTLBuffer {
    requestBuffer(size: size, keyPath: \.scratchBuffer)
  }
  
  func requestPartialsBuffer(size: Int) -> MTLBuffer {
    requestBuffer(size: size, keyPath: \.partialsBuffer)
  }
  
  func requestLocksBuffer(size: Int) -> MTLBuffer {
    requestBuffer(size: size, keyPath: \.locksBuffer)
  }
  
  private func requestBuffer(
    size: Int, keyPath: ReferenceWritableKeyPath<OperationCache, MTLBuffer?>
  ) -> MTLBuffer {
    let device = MetalContext.global.device
    var regenerateBuffer = (self[keyPath: keyPath] == nil)
    if !regenerateBuffer {
      regenerateBuffer = (self[keyPath: keyPath]!.length < size)
    }
    if regenerateBuffer {
      func roundUpToPowerOf2(_ input: Int) -> Int {
          1 << (Int.bitWidth - max(0, input - 1).leadingZeroBitCount)
      }
      let roundedSize = roundUpToPowerOf2(size)
      self[keyPath: keyPath] = device.makeBuffer(length: roundedSize)
    }
    return self[keyPath: keyPath]!
  }
  
  func clear() {
    gemm.removeAll()
    attention.removeAll()
  }
  
  func cache(operation _operation: Backend._Attention) {
    var reducedOperation = _operation
    if Backend.dynamicBatch {
      reducedOperation.parameters.batchDimensionsQ = nil
      reducedOperation.parameters.batchDimensionsMask = nil
    }
    guard attention[reducedOperation.parameters] == nil else {
      return
    }
    attention[reducedOperation.parameters] = _operation.makeAsyncResource()
  }
  
  func cache(operation _operation: Backend._GEMM) {
    var reducedOperation = _operation
    if Backend.dynamicBatch {
      reducedOperation.parameters.batchDimensionsA = nil
      reducedOperation.parameters.batchDimensionsB = nil
    }
    guard gemm[reducedOperation.parameters] == nil else {
      return
    }
    gemm[reducedOperation.parameters] = _operation.makeAsyncResource()
  }
  
  func encode(
    operation _operation: Backend._Attention,
    encoder: Backend.Encoder,
    tensors: Attention_Tensors
  ) {
    var reducedOperation = _operation
    if Backend.dynamicBatch {
      reducedOperation.parameters.batchDimensionsQ = nil
      reducedOperation.parameters.batchDimensionsMask = nil
    }
    guard let resource = attention[reducedOperation.parameters] else {
      fatalError("Forgot ghost pass.")
    }
    _operation.encode(encoder: encoder, tensors: tensors, resource: resource)
  }
  
  func encode(
    operation _operation: Backend._GEMM,
    encoder: Backend.Encoder,
    tensors: GEMM_Tensors
  ) {
    var reducedOperation = _operation
    if Backend.dynamicBatch {
      reducedOperation.parameters.batchDimensionsA = nil
      reducedOperation.parameters.batchDimensionsB = nil
    }
    guard let resource = gemm[reducedOperation.parameters] else {
      fatalError("Forgot ghost pass.")
    }
    _operation.encode(encoder: encoder, tensors: tensors, resource: resource)
  }
}
