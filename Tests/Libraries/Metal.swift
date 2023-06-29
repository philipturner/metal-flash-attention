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
_GEMM: MetalOperation, _GEMM.Backend == Self
{
  associatedtype Encoder
  associatedtype Resource: AsyncResource
  
  var encoder: Encoder { get }
  
  var cache: OperationCache<Self> { get }
}

extension MetalBackend {
  func dispatch(parameters: GEMM_Parameters, tensors: GEMM_Tensors) {
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
  func finish(resource: Resource)
  
  // Lazily blocks until the background thread finishes.
  var resource: Resource { get }
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
  var gemm: [GEMM_Parameters: Backend.Resource] = [:]
  
  func clear() {
    gemm.removeAll()
  }
  
  func cache(operation: Backend._GEMM) {
    guard gemm[operation.parameters] == nil else {
      return
    }
    gemm[operation.parameters] = operation.makeAsyncResource()
  }
  
  func encode(
    operation: Backend._GEMM,
    encoder: Backend.Encoder,
    tensors: GEMM_Tensors
  ) {
    guard let resource = gemm[operation.parameters] else {
      fatalError("Forgot ghost pass.")
    }
    operation.encode(encoder: encoder, tensors: tensors, resource: resource)
  }
}
